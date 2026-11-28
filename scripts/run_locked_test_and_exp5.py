#!/usr/bin/env python
"""One-shot locked NIH test-set evaluation + Experiment 5 (MedGemma Claim Verifier).

Strictly uses frozen ``configs/frozen.yaml`` (never tunes on test) and reports
localization ONLY on ``bbox_role == 'reporting'`` expert boxes.
"""

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

from eval.ablations import compute_fill_sensitivity, run_ablations  # noqa: E402
from eval.metrics import (  # noqa: E402
    compute_localization_metrics,
    compute_reader_metrics,
    compute_subgroup_metrics,
    summarize_latencies,
)
from eval.risk_coverage import compare_reader_vs_judge, plot_risk_coverage  # noqa: E402
from eval.run_eval import _age_band, _index_images, _load_bboxes  # noqa: E402
from veriray.config import configs_from_yaml  # noqa: E402
from veriray.contracts import Verdict  # noqa: E402
from veriray.pipeline import run_verification  # noqa: E402
from veriray.reader import Reader  # noqa: E402
from veriray.splits import load_manifest, select_split  # noqa: E402
from veriray.vlm import MedGemmaRunner, VLMConfig, run_grounded_vlm_explainer  # noqa: E402


def main() -> int:
    frozen_path = ROOT / "configs" / "frozen.yaml"
    if not frozen_path.is_file():
        print("Refusing locked test run: configs/frozen.yaml does not exist.", file=sys.stderr)
        return 2

    loc_cfg, judge_cfg, _ = configs_from_yaml(frozen_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reader = Reader.load_for_finding(ROOT / "models", "pneumothorax", device=device)

    rows = load_manifest(ROOT / "data" / "manifests" / "nih_manifest.csv")
    test_rows = select_split(rows, "test", allow_test=True)
    index = _index_images(Path(r"C:\Users\swast\nih"))
    model_sz = float(getattr(reader, "img_size", 224))
    boxes = _load_bboxes(Path(r"C:\Users\swast\nih\BBox_List_2017.csv"), target_size=model_sz)

    # Balanced 400-case locked test sample + any reporting-half bbox images
    pos = [r for r in test_rows if r["pneumothorax"] == 1 and r["image"] in index]
    neg = [r for r in test_rows if r["pneumothorax"] == 0 and r["image"] in index]
    rng = np.random.default_rng(20261128)
    rng.shuffle(pos)
    rng.shuffle(neg)
    selected = pos[:200] + neg[:200]
    seen_ids = {r["image"] for r in selected}
    for r in test_rows:
        if r.get("bbox_role") == "reporting" and r["image"] in boxes and r["image"] in index and r["image"] not in seen_ids:
            selected.append(r)
            seen_ids.add(r["image"])

    print(f"Running locked test evaluation on {len(selected)} images (device={device})...", flush=True)
    records: list[dict] = []
    fill_cases: list[dict] = []
    loc_cases: list[dict] = []
    verdict_counts: Counter[str] = Counter()
    sample_evidence = []

    for r in selected:
        res = run_verification(
            index[r["image"]],
            reader=reader,
            manifest_dir=None,
            locator_cfg=loc_cfg,
            judge_cfg=judge_cfg,
            raw_metadata={"view": r.get("view", "UNKNOWN"), "source_dataset": "nih_chestxray14", "split": "test"},
            dataset_split_id="nih_test_locked",
        )
        verdict_counts[res.verdict.value] += 1
        score = res.reader.score_raw if res.reader is not None else 0.0
        jo = res.judge_outcome
        te = dict(jo.contract.target_effects) if jo else {}
        ce = dict(jo.contract.control_effects) if jo else {}
        records.append({
            "image": r["image"],
            "patient_id": r["patient_id"],
            "y_true": int(r["pneumothorax"]),
            "score": score,
            "judge_accepted": res.verdict is Verdict.MODEL_SUPPORTED,
            "verdict": res.verdict.value,
            "target_drop": float(te.get("primary_score_drop") or 0.0),
            "control_median_drop": float(ce.get("median_score_drop") or 0.0),
            "sex": r.get("sex") or "UNKNOWN",
            "age_band": _age_band(r.get("age", "")),
            "latencies_ms": res.latencies_ms,
        })
        if "per_method_score_drops" in te:
            fill_cases.append({"per_method_score_drops": te["per_method_score_drops"]})
        if res.locator is not None and r.get("bbox_role") == "reporting" and r["image"] in boxes:
            loc_cases.append({"heatmap": res.locator.heatmap, "mask": res.locator.target_mask, "gt_boxes_xyxy": boxes[r["image"]]})
        if jo is not None and len(sample_evidence) < 8 and res.verdict is not Verdict.INDETERMINATE:
            sample_evidence.append(jo.evidence_record)

    # Free DenseNet before loading 4-bit MedGemma for Experiment 5
    del reader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    exp5 = {"cases_tested": 0, "total_sentences_before_verifier": 0, "dropped_by_verifier": 0,
            "unsupported_rate_before_verifier": None, "unsupported_rate_after_verifier": 0.0}
    if sample_evidence and (ROOT / "models" / "medgemma" / "config.json").is_file():
        print(f"Running Experiment 5 (MedGemma Claim Verifier) on {len(sample_evidence)} test EvidenceRecords...", flush=True)
        vlm = MedGemmaRunner.load(VLMConfig(model_dir=ROOT / "models" / "medgemma", quantization="4bit"))
        for ev in sample_evidence:
            rep = run_grounded_vlm_explainer(ev, vlm.generate_text)
            exp5["cases_tested"] += 1
            exp5["total_sentences_before_verifier"] += rep.total_candidates
            exp5["dropped_by_verifier"] += rep.dropped_count
        vlm.unload()
        if exp5["total_sentences_before_verifier"] > 0:
            exp5["unsupported_rate_before_verifier"] = (
                exp5["dropped_by_verifier"] / float(exp5["total_sentences_before_verifier"])
            )

    out_dir = ROOT / "artifacts" / "eval_test_locked"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "note": "NIH ChestX-ray14 LOCKED TEST SPLIT (frozen configs/frozen.yaml; weak report-mined labels)",
        "verdict_counts": dict(verdict_counts),
        "reader_metrics": compute_reader_metrics([r["y_true"] for r in records], [r["score"] for r in records]),
        "risk_coverage_comparison": compare_reader_vs_judge(records, n_boot=300),
        "ablations": run_ablations(records, min_target_drop=judge_cfg.min_target_drop, min_margin=judge_cfg.min_margin_over_controls, n_boot=300),
        "fill_sensitivity": compute_fill_sensitivity(fill_cases, min_target_drop=judge_cfg.min_target_drop),
        "localization_reporting_half": compute_localization_metrics(loc_cases),
        "experiment_5_claim_verifier": exp5,
        "latencies": summarize_latencies(records),
        "subgroups_sex": compute_subgroup_metrics(records, group_key="sex"),
        "subgroups_age_band": compute_subgroup_metrics(records, group_key="age_band"),
    }
    (out_dir / "test_evidence_pack.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_risk_coverage(records, out_dir / "risk_coverage.png", title="NIH Locked Test: Reader-only vs. Full Judge")
    print(f"Wrote {out_dir / 'test_evidence_pack.json'} (n={len(records)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
