#!/usr/bin/env python
"""Feasibility evaluation for extending VeriRay beyond pneumothorax (validation split ONLY).

Strict rules:
- Uses validation split ONLY for all model inference, localization, Judge, and eraser checks.
- Does NOT run any model or read any model output on the test split.
- Counting positive labels and bounding-box annotations in train/val/test is allowed.
- Loads configs/frozen.yaml read-only; changes no thresholds and touches no configs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import torchxrayvision as xrv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.metrics import compute_reader_metrics  # noqa: E402
from eval.run_eval import _index_images  # noqa: E402
from veriray.anatomy import LungSegmenter  # noqa: E402
from veriray.config import configs_from_yaml  # noqa: E402
from veriray.contracts import Verdict  # noqa: E402
from veriray.eraser import erase, render_labelled  # noqa: E402
from veriray.gate import run_input_gate  # noqa: E402
from veriray.judge import run_judge  # noqa: E402
from veriray.locator import locate  # noqa: E402
from veriray.reader import CheckpointRecord, Reader, preprocess  # noqa: E402
from veriray.splits import load_manifest, select_split  # noqa: E402

NIH_IMAGES_DIR = Path(r"C:\Users\swast\nih")
BBOX_CSV = NIH_IMAGES_DIR / "BBox_List_2017.csv"
MANIFEST_CSV = ROOT / "data" / "manifests" / "nih_manifest.csv"
OUT_DIR = ROOT / "artifacts" / "feasibility"

# 8 target findings: (key, nih_manifest_label, bbox_csv_label, xrv_nih_label, xrv_chex_label_or_None)
FINDINGS: tuple[tuple[str, str, str, str, str | None], ...] = (
    ("mass", "Mass", "Mass", "Mass", None),
    ("nodule", "Nodule", "Nodule", "Nodule", None),
    ("effusion", "Effusion", "Effusion", "Effusion", "Effusion"),
    ("cardiomegaly", "Cardiomegaly", "Cardiomegaly", "Cardiomegaly", "Cardiomegaly"),
    ("atelectasis", "Atelectasis", "Atelectasis", "Atelectasis", "Atelectasis"),
    ("infiltrate", "Infiltration", "Infiltrate", "Infiltration", None),
    ("pneumonia", "Pneumonia", "Pneumonia", "Pneumonia", "Pneumonia"),
    ("pneumothorax", "Pneumothorax", "Pneumothorax", "Pneumothorax", "Pneumothorax"),
)


def _count_labels_and_bboxes(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    bbox_imgs: dict[str, set[str]] = defaultdict(set)
    with BBOX_CSV.open(newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        next(r, None)
        for row in r:
            if len(row) >= 6:
                bbox_imgs[row[1].strip()].add(row[0].strip())

    counts = {k: Counter() for k, *_ in FINDINGS}
    bbox_counts = {k: Counter() for k, *_ in FINDINGS}
    for row in rows:
        sp = row["split"]
        labs = {x.strip() for x in row["finding_labels"].split("|") if x.strip()}
        img = row["image"]
        for k, m_lab, b_lab, *_ in FINDINGS:
            if m_lab in labs:
                counts[k][sp] += 1
            if img in bbox_imgs[b_lab]:
                bbox_counts[k][sp] += 1

    out: dict[str, dict[str, Any]] = {}
    for k, m_lab, b_lab, *_ in FINDINGS:
        out[k] = {
            "manifest_label": m_lab,
            "bbox_label": b_lab,
            "train_pos": int(counts[k]["train"]),
            "val_pos": int(counts[k]["val"]),
            "test_pos": int(counts[k]["test"]),
            "bbox_total": int(len(bbox_imgs[b_lab])),
            "bbox_train": int(bbox_counts[k]["train"]),
            "bbox_val": int(bbox_counts[k]["val"]),
            "bbox_test": int(bbox_counts[k]["test"]),
        }
    return out


def _patient_bootstrap_ci(
    patient_ids: list[str],
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    n_boot: int = 400,
    seed: int = 20261126,
) -> dict[str, Any]:
    """Patient-level bootstrap 95% CI for AUROC and AUPRC."""
    by_pat: dict[str, list[int]] = defaultdict(list)
    for i, pid in enumerate(patient_ids):
        by_pat[pid].append(i)
    pats = list(by_pat.keys())
    n_pats = len(pats)
    rng = np.random.default_rng(seed)

    aurocs: list[float] = []
    auprcs: list[float] = []
    for _ in range(n_boot):
        sampled = rng.integers(0, n_pats, size=n_pats)
        idx = [i for p_idx in sampled for i in by_pat[pats[p_idx]]]
        yb = y_true[idx]
        sb = scores[idx]
        if len(np.unique(yb)) < 2:
            continue
        rm = compute_reader_metrics(yb, sb)
        if rm["auroc"] is not None:
            aurocs.append(float(rm["auroc"]))
        if rm["auprc"] is not None:
            auprcs.append(float(rm["auprc"]))

    def _ci(vals: list[float]) -> list[float] | None:
        if not vals:
            return None
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    return {
        "n_boot": len(aurocs),
        "auroc_ci95": _ci(aurocs),
        "auprc_ci95": _ci(auprcs),
    }


def _make_reader(model: torch.nn.Module, pathologies: list[str], weights_id: str, finding_label: str, device: torch.device) -> Reader:
    idx = pathologies.index(finding_label)
    rec = CheckpointRecord(
        weights_id=weights_id,
        filename=f"local:{weights_id}",
        sha256="0" * 64,
        label_list=tuple(pathologies),
        finding_label=finding_label,
        finding_index=idx,
        source_url=f"local:{weights_id}",
        license="torchxrayvision Apache-2.0",
        training_datasets=(weights_id,),
    )
    return Reader(model, rec, device)


def _save_before_after_crop(
    out_path: Path,
    orig_224: np.ndarray,
    edited_224: np.ndarray,
    mask: np.ndarray,
    finding: str,
    image_name: str,
    score: float,
) -> None:
    """Save side-by-side before/after crop with mask contour + SYNTHETIC COUNTERFACTUAL banner."""
    def _to_bgr(arr: np.ndarray) -> np.ndarray:
        u8 = np.clip(arr, 0, 255).astype(np.uint8)
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)

    left = _to_bgr(orig_224)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(left, contours, -1, (0, 220, 255), 1)
    cv2.putText(left, f"ORIG {finding} s={score:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 255), 1, cv2.LINE_AA)

    labelled_after = render_labelled(edited_224)
    right = cv2.cvtColor(labelled_after, cv2.COLOR_RGB2BGR)
    cv2.drawContours(right, contours, -1, (0, 200, 0), 1)

    panel = np.hstack([left, right])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def _categorize_reason_codes(verdict: Verdict, reason_codes: tuple[str, ...]) -> str:
    rcs = set(reason_codes)
    if verdict is Verdict.MODEL_SUPPORTED:
        return "accepted_model_supported"
    if verdict is Verdict.SHORTCUT_RISK or "SHORTCUT_OUT_OF_LUNG_ATTRIBUTION" in rcs:
        return "localization_shortcut"
    if any(r.startswith("LOCALIZATION_") or r == "JUDGE_MASK_UNSTABLE" for r in rcs):
        return "localization_gate"
    if any(r.startswith("JUDGE_INSUFFICIENT_CONTROLS") or r.startswith("JUDGE_CONTROL_") for r in rcs):
        return "controls_invalid"
    if any(r in rcs for r in (
        "JUDGE_BOUNDARY_DISCONTINUITY_EXCEEDED",
        "JUDGE_UNRELATED_LABEL_DRIFT_EXCEEDED",
        "JUDGE_FILL_METHOD_INSTABILITY",
        "JUDGE_TARGET_EDIT_FAILED",
        "JUDGE_TARGET_EDIT_UNEXPECTED_ERROR",
        "ERASER_OUTSIDE_MASK_CHANGED",
    )):
        return "integrity_gate"
    if "READER_SCORE_BELOW_THRESHOLD" in rcs:
        return "reader_score_gate"
    if any(r in rcs for r in (
        "TARGET_DROP_BELOW_THRESHOLD",
        "MARGIN_OVER_CONTROLS_BELOW_THRESHOLD",
        "EMPIRICAL_RANK_NOT_DISTINCT_FROM_CONTROLS",
    )):
        return "effect_or_rank_not_supported"
    return "other_gate"


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    rows = load_manifest(MANIFEST_CSV)
    val_rows = select_split(rows, "val")
    index = _index_images(NIH_IMAGES_DIR)
    print(f"Loaded manifest: total={len(rows)}, val={len(val_rows)}, indexed_pngs={len(index)}", flush=True)

    # 1. Label & BBox counts across train / val / test (labels only, no model inference on test)
    split_counts = _count_labels_and_bboxes(rows)

    # 2. Load frozen config (read-only) and models
    loc_cfg, judge_cfg, _ = configs_from_yaml(ROOT / "configs" / "frozen.yaml")
    lung_seg = LungSegmenter.load(ROOT / "models", device=str(device))

    m_nih = xrv.models.DenseNet(weights="densenet121-res224-nih").to(device).eval()
    for p in m_nih.parameters():
        p.requires_grad_(False)
    nih_paths = list(m_nih.pathologies)

    m_chex = xrv.models.DenseNet(weights="densenet121-res224-chex").to(device).eval()
    for p in m_chex.parameters():
        p.requires_grad_(False)
    chex_paths = list(m_chex.pathologies)

    # 3. Full validation pass for Reader AUROC / AUPRC (batched for speed on all 17,222 val images)
    val_items = [r for r in val_rows if r["image"] in index]
    patient_ids = [r["patient_id"] for r in val_items]
    y_by_finding: dict[str, list[int]] = {k: [] for k, *_ in FINDINGS}
    for r in val_items:
        labs = {x.strip() for x in r["finding_labels"].split("|") if x.strip()}
        for k, m_lab, *_ in FINDINGS:
            y_by_finding[k].append(1 if m_lab in labs else 0)

    reader_cache_path = OUT_DIR / "reader_val_metrics.json"
    if reader_cache_path.is_file():
        print(f"Loaded cached Reader validation metrics from {reader_cache_path}", flush=True)
        reader_metrics = json.loads(reader_cache_path.read_text(encoding="utf-8"))
    else:
        print("Running batched Reader inference on full validation split...", flush=True)
        nih_scores: dict[str, list[float]] = {k: [] for k, *_ in FINDINGS}
        chex_scores: dict[str, list[float]] = {k: [] for k, _, _, _, cl in FINDINGS if cl is not None}

        batch_size = 64
        batch_tensors: list[torch.Tensor] = []
        for i, r in enumerate(val_items):
            img = cv2.imread(str(index[r["image"]]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((224, 224), dtype=np.uint8)
            t_img, _ = preprocess(img)
            batch_tensors.append(t_img[0])  # 1x224x224
            if len(batch_tensors) == batch_size or i == len(val_items) - 1:
                xb = torch.stack(batch_tensors, dim=0).to(device)
                with torch.no_grad():
                    feat_nih = torch.relu(m_nih.features(xb))
                    pooled_nih = torch.nn.functional.adaptive_avg_pool2d(feat_nih, (1, 1)).flatten(1)
                    probs_nih = torch.sigmoid(m_nih.classifier(pooled_nih)).cpu().numpy()

                    feat_chex = torch.relu(m_chex.features(xb))
                    pooled_chex = torch.nn.functional.adaptive_avg_pool2d(feat_chex, (1, 1)).flatten(1)
                    probs_chex = torch.sigmoid(m_chex.classifier(pooled_chex)).cpu().numpy()

                for k, _, _, n_lab, c_lab in FINDINGS:
                    n_idx = nih_paths.index(n_lab)
                    nih_scores[k].extend(probs_nih[:, n_idx].tolist())
                    if c_lab is not None:
                        c_idx = chex_paths.index(c_lab)
                        chex_scores[k].extend(probs_chex[:, c_idx].tolist())
                batch_tensors.clear()
                if (i + 1) % 3200 == 0 or i == len(val_items) - 1:
                    print(f"  val reader progress: {i + 1}/{len(val_items)} ({time.time() - t0:.1f}s)", flush=True)

        reader_metrics = {}
        for k, *_ in FINDINGS:
            y_arr = np.asarray(y_by_finding[k], dtype=int)
            s_nih = np.asarray(nih_scores[k], dtype=float)
            rm_nih = compute_reader_metrics(y_arr, s_nih)
            ci_nih = _patient_bootstrap_ci(patient_ids, y_arr, s_nih, n_boot=400, seed=20261126)
            entry: dict[str, Any] = {
                "nih_model": {
                    "n": rm_nih["n"],
                    "positives": rm_nih["positives"],
                    "auroc": rm_nih["auroc"],
                    "auroc_ci95": ci_nih["auroc_ci95"],
                    "auprc": rm_nih["auprc"],
                    "auprc_ci95": ci_nih["auprc_ci95"],
                }
            }
            if k in chex_scores:
                s_chex = np.asarray(chex_scores[k], dtype=float)
                rm_chex = compute_reader_metrics(y_arr, s_chex)
                ci_chex = _patient_bootstrap_ci(patient_ids, y_arr, s_chex, n_boot=400, seed=20261126)
                entry["chex_model"] = {
                    "auroc": rm_chex["auroc"],
                    "auroc_ci95": ci_chex["auroc_ci95"],
                    "auprc": rm_chex["auprc"],
                    "auprc_ci95": ci_chex["auprc_ci95"],
                }
            else:
                entry["chex_model"] = None
            reader_metrics[k] = entry
            chex_str = (
                f" | CHEX AUROC={entry['chex_model']['auroc']:.3f} AUPRC={entry['chex_model']['auprc']:.3f}"
                if entry["chex_model"]
                else ""
            )
            print(
                f"  [Reader] {k:13s} NIH AUROC={rm_nih['auroc']:.3f} {ci_nih['auroc_ci95']} "
                f"AUPRC={rm_nih['auprc']:.3f} {ci_nih['auprc_ci95']}{chex_str}",
                flush=True,
            )
        reader_cache_path.write_text(json.dumps(reader_metrics, indent=2) + "\n", encoding="utf-8")

    # 4. Judge reason-code histogram + Locator flags + 10 Eraser visual crops per finding (val sample)
    print("Running Judge & Eraser checks on validation samples (n=100 per finding: 50 pos + 50 neg)...", flush=True)
    rng = np.random.default_rng(20261126)
    judge_summary: dict[str, dict[str, Any]] = {}

    for k, m_lab, _, n_lab, _ in FINDINGS:
        reader_k = _make_reader(m_nih, nih_paths, "densenet121-res224-nih", n_lab, device)
        pos_rows = [r for r, y in zip(val_items, y_by_finding[k]) if y == 1]
        neg_rows = [r for r, y in zip(val_items, y_by_finding[k]) if y == 0]
        rng.shuffle(pos_rows)
        rng.shuffle(neg_rows)
        sample_rows = pos_rows[:50] + neg_rows[:50]

        verdict_counts: Counter[str] = Counter()
        reason_counts: Counter[str] = Counter()
        gate_bucket_counts: Counter[str] = Counter()
        crops_saved = 0
        crops_dir = OUT_DIR / "crops" / k

        for r in sample_rows:
            img_path = index[r["image"]]
            inp, gate_res = run_input_gate(
                img_path,
                raw_metadata={"view": r.get("view", "UNKNOWN"), "source_dataset": "nih_chestxray14", "split": "val"},
                model_id=reader_k.checkpoint_id,
                preprocessing_id="xrv-centercrop-cv2area224-norm1024-v1",
            )
            if gate_res.status.value != "ACCEPT" or inp is None:
                verdict_counts[Verdict.INDETERMINATE.value] += 1
                for rc in gate_res.reason_codes:
                    reason_counts[rc] += 1
                gate_bucket_counts["gate_reject"] += 1
                continue

            rc_out, tensor_224, display_224 = reader_k.score(inp.image_tensor)
            loc = locate(reader_k, tensor_224, loc_cfg)
            lung_mask = lung_seg.segment_lungs(display_224)

            if any(loc.localization_flags.get(f, False) for f in ("empty", "diffuse", "implausibly_large")):
                loc_reasons = tuple(
                    f"LOCALIZATION_{f.upper()}"
                    for f in ("empty", "diffuse", "implausibly_large")
                    if loc.localization_flags.get(f, False)
                )
                verdict_counts[Verdict.INDETERMINATE.value] += 1
                for lr in loc_reasons:
                    reason_counts[lr] += 1
                gate_bucket_counts["localization_gate"] += 1
                continue

            jo = run_judge(
                reader_k,
                display_224,
                rc_out,
                loc,
                cfg=judge_cfg,
                locator_cfg=loc_cfg,
                lung_mask=lung_mask,
                record_id=f"feas_{k}_{r['image']}",
            )
            verdict = jo.contract.verdict
            verdict_counts[verdict.value] += 1
            for rc in jo.contract.reason_codes:
                reason_counts[rc] += 1
            bucket = _categorize_reason_codes(verdict, jo.contract.reason_codes)
            gate_bucket_counts[bucket] += 1

            # Save up to 10 before/after crops from positive cases
            labs = {x.strip() for x in r["finding_labels"].split("|") if x.strip()}
            if m_lab in labs and crops_saved < 10 and loc.target_mask.any():
                edit_img = (
                    jo.primary_target_edit.edited_image
                    if jo.primary_target_edit is not None
                    else erase(display_224, loc.target_mask, "telea", seed=judge_cfg.seed).edited_image
                )
                crops_saved += 1
                _save_before_after_crop(
                    crops_dir / f"case_{crops_saved:02d}_{r['image']}",
                    display_224,
                    edit_img,
                    loc.target_mask,
                    k,
                    r["image"],
                    rc_out.score_raw,
                )

        judge_summary[k] = {
            "sample_n": len(sample_rows),
            "verdicts": dict(verdict_counts),
            "gate_buckets": dict(gate_bucket_counts),
            "reason_codes": dict(reason_counts),
            "crops_saved": crops_saved,
        }
        print(f"  [Judge] {k:13s} verdicts={dict(verdict_counts)} buckets={dict(gate_bucket_counts)} crops={crops_saved}", flush=True)

    report = {
        "split_used_for_models": "val",
        "test_split_model_outputs_read": False,
        "frozen_config_path": "configs/frozen.yaml",
        "split_counts_and_bboxes": split_counts,
        "reader_metrics_val": reader_metrics,
        "locator_val_bbox_note": "0 validation bounding-box cases across all 8 findings in BBox_List_2017.csv (all 984 boxes are in the locked test split).",
        "judge_val_summary": judge_summary,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    out_json = OUT_DIR / "feasibility_results.json"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {out_json} in {report['elapsed_sec']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
