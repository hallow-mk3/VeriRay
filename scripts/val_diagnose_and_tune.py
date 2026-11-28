#!/usr/bin/env python
"""Diagnose NIH val Judge gate outcomes and tune thresholds on val ONLY (GPU)."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.risk_coverage import compare_reader_vs_judge, plot_risk_coverage  # noqa: E402
from eval.run_eval import _index_images  # noqa: E402
from scripts.tune_and_freeze import select_best_judge_thresholds  # noqa: E402
from veriray.config import configs_from_yaml, freeze_config  # noqa: E402
from veriray.contracts import Verdict  # noqa: E402
from veriray.judge import JudgeConfig  # noqa: E402
from veriray.pipeline import run_verification  # noqa: E402
from veriray.reader import CheckpointRecord, Reader  # noqa: E402
from veriray.splits import load_manifest, select_split  # noqa: E402
import torchxrayvision as xrv  # noqa: E402


def _load_nih_trained_reader(device: str) -> Reader:
    rec_d = json.loads((ROOT / "models" / "NIH_TRAINED_CHECKPOINT.json").read_text(encoding="utf-8"))
    wpath = ROOT / "models" / rec_d["filename"]
    state = torch.load(wpath, map_location="cpu", weights_only=True)
    model = xrv.models.DenseNet(weights=None, num_classes=len(rec_d["label_list"]), in_channels=1)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dev = torch.device(device)
    model.to(dev)
    rec = CheckpointRecord(
        weights_id=rec_d["weights_id"],
        filename=rec_d["filename"],
        sha256=rec_d["sha256"],
        label_list=tuple(rec_d["label_list"]),
        finding_label="Pneumothorax",
        finding_index=int(rec_d["finding_index"]),
        source_url="local:models/densenet121-nih-train-ptx.pt",
        license="torchxrayvision Apache-2.0; NIH ChestX-ray14 terms apply",
        training_datasets=("NIH ChestX-ray14 train split ONLY (weak labels)",),
    )
    return Reader(model, rec, dev)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reader = Reader.load_for_finding(ROOT / "models", "pneumothorax", device=device)
    rows = load_manifest(ROOT / "data" / "manifests" / "nih_manifest.csv")
    val_rows = select_split(rows, "val")
    index = _index_images(Path(r"C:\Users\swast\nih"))

    pos = [r for r in val_rows if r["pneumothorax"] == 1 and r["image"] in index]
    neg = [r for r in val_rows if r["pneumothorax"] == 0 and r["image"] in index]
    rng = np.random.default_rng(20261126)
    rng.shuffle(pos)
    rng.shuffle(neg)
    subset = pos[:150] + neg[:150]

    # Relax drift/boundary slightly during val data collection so we record raw Judge metrics cleanly
    probe_cfg = JudgeConfig(max_unrelated_drift=1.0, max_boundary_discontinuity=1.0, max_fill_method_spread=1.0)
    reason_counts: Counter[str] = Counter()
    val_records: list[dict] = []

    for r in subset:
        res = run_verification(
            index[r["image"]],
            reader=reader,
            manifest_dir=None,
            judge_cfg=probe_cfg,
            raw_metadata={"view": r.get("view", "UNKNOWN"), "source_dataset": "nih_chestxray14", "split": "val"},
            dataset_split_id="nih_val",
        )
        for rc in res.reason_codes:
            reason_counts[rc] += 1
        jo = res.judge_outcome
        te = dict(jo.contract.target_effects) if jo else {}
        ce = dict(jo.contract.control_effects) if jo else {}
        sm = dict(jo.contract.stability_metrics) if jo else {}
        sf = dict(jo.contract.shortcut_flags) if jo else {}
        val_records.append({
            "image": r["image"],
            "patient_id": r["patient_id"],
            "y_true": int(r["pneumothorax"]),
            "score": res.reader.score_raw if res.reader else 0.0,
            "verdict": res.verdict.value,
            "reason_codes": list(res.reason_codes),
            "target_drop": float(te.get("primary_score_drop") or 0.0),
            "control_median_drop": float(ce.get("median_score_drop") or 0.0),
            "empirical_tail_fraction": float(te.get("empirical_tail_fraction") or 1.0),
            "unrelated_drift": float(sm.get("unrelated_output_max_drift") or 0.0),
            "fill_spread": float(sm.get("fill_method_spread") or 0.0),
            "out_of_lung": bool(sf.get("out_of_lung_attribution", False)),
            "shortcut_or_integrity_failed": bool(sf.get("out_of_lung_attribution", False) or res.verdict is Verdict.INDETERMINATE),
        })

    print("Val reason code counts (n=300):", dict(reason_counts), flush=True)
    loc_cfg, base_judge, _ = configs_from_yaml(ROOT / "configs" / "default.yaml")
    # Tune Judge thresholds on val_records (including min_reader_score so Judge gates positive predictions)
    best_cfg = JudgeConfig(max_unrelated_drift=0.50, max_boundary_discontinuity=0.50, max_fill_method_spread=0.40)
    best_key = (-1.0, -1)
    stats: dict = {}
    for s_min in (0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90):
        sub_recs = [
            {**r, "shortcut_or_integrity_failed": bool(r["shortcut_or_integrity_failed"] or r["score"] < s_min)}
            for r in val_records
        ]
        cand_cfg, cand_stats = select_best_judge_thresholds(
            sub_recs,
            JudgeConfig(max_unrelated_drift=0.50, max_boundary_discontinuity=0.50, max_fill_method_spread=0.40, min_reader_score=s_min),
            drop_grid=(0.02, 0.04, 0.06, 0.08),
            margin_grid=(0.01, 0.02, 0.03),
            tail_grid=(0.10, 0.15, 0.20),
            min_accepted=10,
        )
        if cand_stats and (cand_stats["precision"], cand_stats["accepted_count"]) > best_key:
            best_key = (cand_stats["precision"], cand_stats["accepted_count"])
            best_cfg = cand_cfg
            stats = cand_stats
    tuned = best_cfg
    print("Tuned val stats:", stats, flush=True)
    print("Tuned JudgeConfig:", tuned, flush=True)
    out_cfg, digest = freeze_config(loc_cfg, tuned, out_path=ROOT / "configs" / "frozen.yaml", fitted_split="val", metadata=stats)
    print(f"Froze config to {out_cfg} (sha256={digest[:12]}...)", flush=True)

    # Compare Reader-only vs Full Judge at matched coverage using tuned thresholds
    for rec in val_records:
        rec["judge_accepted"] = bool(
            rec["score"] >= tuned.min_reader_score
            and rec["target_drop"] >= tuned.min_target_drop
            and (rec["target_drop"] - rec["control_median_drop"]) >= tuned.min_margin_over_controls
            and rec["empirical_tail_fraction"] <= tuned.max_empirical_tail_fraction
            and not rec["shortcut_or_integrity_failed"]
        )
    cmp = compare_reader_vs_judge(val_records, n_boot=200)
    print("Operating point comparison:", json.dumps(cmp["operating_point"], indent=2), flush=True)
    out_dir = ROOT / "artifacts" / "eval_val_tuned"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "val_tuned_summary.json").write_text(json.dumps({"reason_counts": dict(reason_counts), "tuned_stats": stats, "comparison": cmp}, indent=2) + "\n", encoding="utf-8")
    plot_risk_coverage(val_records, out_dir / "risk_coverage.png", title="NIH Val (n=300): Reader-only vs. Tuned Judge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
