#!/usr/bin/env python
"""Evaluation runner for VeriRay (Blueprint 26 Nov §9).

Supports:
- ``--demo-synthetic``: runs on synthetic records to verify metrics, tables, and plots.
- Real NIH manifest mode (``--manifest data/manifests/nih_manifest.csv --images-dir ...``),
  defaulting to the ``val`` split and locking ``test`` unless ``--unlock-test --confirm-frozen``
  is explicitly passed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.ablations import compute_fill_sensitivity, run_ablations  # noqa: E402
from eval.metrics import (  # noqa: E402
    compute_localization_metrics,
    compute_reader_metrics,
    compute_subgroup_metrics,
    summarize_latencies,
)
from eval.risk_coverage import compare_reader_vs_judge, plot_risk_coverage  # noqa: E402
from veriray.contracts import Verdict  # noqa: E402
from veriray.pipeline import run_verification  # noqa: E402
from veriray.reader import Reader  # noqa: E402
from veriray.splits import load_manifest, select_split  # noqa: E402


def _synthetic_records(n: int = 60) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(20261126)
    records = []
    fill_cases = []
    for i in range(n):
        y = int(i % 2 == 0)
        score = float(np.clip(0.65 + 0.2 * rng.normal() if y else 0.35 + 0.2 * rng.normal(), 0.01, 0.99))
        is_shortcut = (y == 0) and (score > 0.6)
        t_drop = (
            float(max(0.0, 0.18 + 0.04 * rng.normal()))
            if (y == 1 and not is_shortcut)
            else float(max(0.0, 0.01 + 0.02 * rng.normal()))
        )
        c_med = float(max(0.0, 0.015 + 0.01 * rng.normal()))
        j_acc = (t_drop >= 0.03) and ((t_drop - c_med) >= 0.02) and not is_shortcut
        records.append({
            "patient_id": f"SYN_{i // 2:04d}",
            "y_true": y,
            "score": score,
            "judge_accepted": j_acc,
            "verdict": "MODEL-SUPPORTED" if j_acc else ("SHORTCUT RISK" if is_shortcut else "NOT SUPPORTED"),
            "target_drop": t_drop,
            "control_median_drop": c_med,
            "random_control_median_drop": c_med + 0.01,
            "sex": "M" if i % 3 else "F",
            "age_band": "40-60" if i % 2 else "60+",
            "latencies_ms": {"gate": 8.0, "reader": 24.0, "locator": 18.0, "judge": 140.0},
        })
        fill_cases.append({
            "per_method_score_drops": {
                "telea": t_drop,
                "gaussian_blur": max(0.0, t_drop - 0.01),
                "mean_fill": max(0.0, t_drop - 0.02),
            }
        })
    return records, fill_cases


def _age_band(age_str: str) -> str:
    try:
        age = int(float(age_str))
    except (TypeError, ValueError):
        return "UNKNOWN"
    if age < 30:
        return "<30"
    if age < 50:
        return "30-49"
    if age < 70:
        return "50-69"
    return "70+"


def _load_bboxes(bbox_csv: Path | None) -> dict[str, list[tuple[float, float, float, float]]]:
    if bbox_csv is None or not bbox_csv.is_file():
        return {}
    out: dict[str, list[tuple[float, float, float, float]]] = {}
    with bbox_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) < 6 or row[1].strip() != "Pneumothorax":
                continue
            x, y, w, h = (float(row[2]), float(row[3]), float(row[4]), float(row[5]))
            # Scale from 1024x1024 NIH coordinates to 224x224 model space
            s = 224.0 / 1024.0
            out.setdefault(row[0].strip(), []).append((x * s, y * s, (x + w) * s, (y + h) * s))
    return out


def _index_images(images_dir: Path) -> dict[str, Path]:
    """Index PNGs in flat ``images_dir`` or Kaggle ``images_*/images/*.png`` subfolders."""
    index: dict[str, Path] = {}
    for p in images_dir.glob("*.png"):
        index[p.name] = p
    if not index:
        for p in images_dir.glob("images_*/images/*.png"):
            index[p.name] = p
    if not index:
        for p in images_dir.rglob("*.png"):
            index[p.name] = p
    return index


def _evaluate_real_split(
    manifest_path: Path,
    images_dir: Path,
    *,
    split: str,
    allow_test: bool,
    bbox_csv: Path | None,
    max_cases: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = load_manifest(manifest_path)
    selected = select_split(rows, split, allow_test=allow_test)
    if max_cases is not None and len(selected) > max_cases:
        # Deterministic patient-level balanced subset so small validation runs include positives & negatives
        pos = [r for r in selected if int(r["pneumothorax"]) == 1]
        neg = [r for r in selected if int(r["pneumothorax"]) == 0]
        rng = np.random.default_rng(20261126)
        rng.shuffle(pos)
        rng.shuffle(neg)
        n_pos = min(len(pos), max_cases // 2)
        n_neg = min(len(neg), max_cases - n_pos)
        selected = pos[:n_pos] + neg[:n_neg]

    image_index = _index_images(images_dir)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reader = Reader.load(ROOT / "models", device=device)
    boxes = _load_bboxes(bbox_csv)
    records: list[dict[str, Any]] = []
    fill_cases: list[dict[str, Any]] = []
    loc_cases: list[dict[str, Any]] = []

    for row in selected:
        # Respect bbox tuning/reporting boundary: never tune on 'reporting' boxes
        img_path = image_index.get(row["image"])
        if img_path is None or not img_path.is_file():
            continue
        res = run_verification(
            img_path,
            reader=reader,
            manifest_dir=None,
            raw_metadata={"view": row.get("view", "UNKNOWN"), "source_dataset": "nih_chestxray14", "split": split},
            dataset_split_id=f"nih_{split}",
        )
        score = res.reader.score_raw if res.reader is not None else 0.0
        te = dict(res.judge_outcome.contract.target_effects) if res.judge_outcome else {}
        ce = dict(res.judge_outcome.contract.control_effects) if res.judge_outcome else {}
        records.append({
            "image": row["image"],
            "patient_id": row["patient_id"],
            "y_true": int(row["pneumothorax"]),
            "score": score,
            "judge_accepted": res.verdict is Verdict.MODEL_SUPPORTED,
            "verdict": res.verdict.value,
            "target_drop": float(te.get("primary_score_drop", 0.0) or 0.0),
            "control_median_drop": float(ce.get("median_score_drop", 0.0) or 0.0),
            "sex": row.get("sex") or "UNKNOWN",
            "age_band": _age_band(row.get("age", "")),
            "latencies_ms": res.latencies_ms,
        })
        if "per_method_score_drops" in te:
            fill_cases.append({"per_method_score_drops": te["per_method_score_drops"]})
        if res.locator is not None and row["image"] in boxes:
            if split == "val" or row.get("bbox_role") == "reporting":
                loc_cases.append({"heatmap": res.locator.heatmap, "mask": res.locator.target_mask, "gt_boxes_xyxy": boxes[row["image"]]})

    return records, fill_cases, loc_cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-synthetic", action="store_true", help="run on synthetic validation records (no NIH data)")
    ap.add_argument("--manifest", type=Path, help="path to nih_manifest.csv")
    ap.add_argument("--images-dir", type=Path, help="directory containing NIH PNG images")
    ap.add_argument("--bbox-csv", type=Path, help="optional BBox_List_2017.csv for localization metrics")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--unlock-test", action="store_true")
    ap.add_argument("--confirm-frozen", action="store_true")
    ap.add_argument("--max-cases", type=int)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "eval_demo")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    if args.demo_synthetic:
        recs, fill_cases = _synthetic_records()
        loc_cases: list[dict[str, Any]] = []
        note = "SYNTHETIC HARNESS VERIFICATION ONLY — NOT MEASURED CLINICAL OR NIH RESULTS"
    elif args.manifest and args.images_dir:
        if not args.manifest.is_file():
            print(f"Manifest not found: {args.manifest}. Run scripts/build_nih_manifest.py first.", file=sys.stderr)
            return 2
        allow_test = bool(args.unlock_test and args.confirm_frozen)
        if args.split == "test" and not allow_test:
            print("Refusing locked test split: pass both --unlock-test and --confirm-frozen after freezing config.", file=sys.stderr)
            return 3
        recs, fill_cases, loc_cases = _evaluate_real_split(
            args.manifest, args.images_dir, split=args.split, allow_test=allow_test,
            bbox_csv=args.bbox_csv, max_cases=args.max_cases,
        )
        note = f"NIH ChestX-ray14 split={args.split} (weak report-mined labels)"
    else:
        print("Pass --demo-synthetic, or --manifest and --images-dir.", file=sys.stderr)
        return 2

    report = {
        "note": note,
        "reader_metrics": compute_reader_metrics([r["y_true"] for r in recs], [r["score"] for r in recs]),
        "risk_coverage_comparison": compare_reader_vs_judge(recs, n_boot=200),
        "ablations": run_ablations(recs, n_boot=200),
        "fill_sensitivity": compute_fill_sensitivity(fill_cases),
        "localization": compute_localization_metrics(loc_cases),
        "latencies": summarize_latencies(recs),
        "subgroups_sex": compute_subgroup_metrics(recs, group_key="sex"),
        "subgroups_age_band": compute_subgroup_metrics(recs, group_key="age_band"),
    }
    (out / "eval_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_path = plot_risk_coverage(recs, out / "risk_coverage.png", title=note)
    print(f"Wrote {out / 'eval_summary.json'} and {plot_path} (n={len(recs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
