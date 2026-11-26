"""Tests for evaluation metrics, patient-level bootstrap risk-coverage, ablations, and localization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eval.ablations import compute_fill_sensitivity, run_ablations
from eval.metrics import (
    compute_localization_metrics,
    compute_reader_metrics,
    compute_subgroup_metrics,
    summarize_latencies,
)
from eval.risk_coverage import compare_reader_vs_judge, patient_bootstrap_ci, plot_risk_coverage


def test_reader_and_subgroup_metrics():
    y = [1, 1, 0, 0, 1, 0]
    s = [0.9, 0.7, 0.2, 0.1, 0.8, 0.4]
    m = compute_reader_metrics(y, s, threshold=0.5)
    assert m["auroc"] == pytest.approx(1.0)
    assert m["sensitivity"] == pytest.approx(1.0) and m["specificity"] == pytest.approx(1.0)
    assert compute_reader_metrics([1, 1], [0.8, 0.9])["auroc"] is None  # single class safe

    recs = [
        {"y_true": 1, "score": 0.9, "sex": "F", "verdict": "MODEL-SUPPORTED"},
        {"y_true": 0, "score": 0.2, "sex": "F", "verdict": "NOT SUPPORTED"},
        {"y_true": 1, "score": 0.8, "sex": "M", "verdict": "MODEL-SUPPORTED"},
        {"y_true": 0, "score": 0.3, "sex": "M", "verdict": "NOT SUPPORTED"},
    ]
    sg = compute_subgroup_metrics(recs, group_key="sex")
    assert set(sg["groups"]) == {"F", "M"}
    assert "does not claim zero bias" in sg["limitations"]


def test_localization_metrics_reports_denominators_and_pointing_game():
    hm = np.zeros((100, 100), float)
    hm[25, 35] = 1.0  # inside box (20, 20, 50, 50)
    mask = np.zeros((100, 100), bool)
    mask[20:45, 20:45] = True

    cases = [
        {"heatmap": hm, "mask": mask, "gt_boxes_xyxy": [(20, 20, 50, 50)]},
        {"heatmap": hm, "mask": mask, "gt_boxes_xyxy": []},  # unannotated => excluded from eligible denominator
    ]
    loc = compute_localization_metrics(cases, iou_success_threshold=0.5)
    assert loc["total_cases_seen"] == 2 and loc["eligible_annotated_cases"] == 1
    assert loc["pointing_game_accuracy"] == pytest.approx(1.0)
    assert loc["mean_iou"] == pytest.approx(625.0 / 900.0) and loc["localization_success_rate"] == pytest.approx(1.0)


def test_risk_coverage_matched_comparison_bootstrap_and_plot(tmp_path: Path):
    recs = []
    for p in range(20):
        y = int(p % 2 == 0)
        score = 0.85 if y else 0.70
        j_acc = bool(y == 1)  # Judge filters out the high-scoring negative errors
        for k in range(2):
            recs.append({
                "patient_id": f"P{p:03d}",
                "y_true": y,
                "score": score,
                "judge_accepted": j_acc,
                "target_drop": 0.20 if j_acc else 0.01,
                "control_median_drop": 0.01,
            })

    cmp = compare_reader_vs_judge(recs, target_coverages=(0.25, 0.50, 0.75), n_boot=100)
    op = cmp["operating_point"]
    assert op["full_judge"]["coverage"] == pytest.approx(0.5)
    assert op["full_judge"]["precision"] == pytest.approx(1.0)
    assert op["precision_delta_judge_minus_reader"] >= 0.0
    assert op["full_judge"]["precision_ci95"][0] <= op["full_judge"]["precision"] <= op["full_judge"]["precision_ci95"][1]

    # Bootstrap respects patient grouping
    lo, hi = patient_bootstrap_ci(recs, lambda s: sum(x["y_true"] for x in s) / len(s), n_boot=50)
    assert 0.0 <= lo <= hi <= 1.0

    out_png = plot_risk_coverage(recs, tmp_path / "rc.png")
    assert out_png.exists() and out_png.stat().st_size > 1000


def test_ablations_fill_sensitivity_and_latencies():
    recs = [
        {"patient_id": "P1", "y_true": 1, "score": 0.8, "judge_accepted": True, "target_drop": 0.2,
         "control_median_drop": 0.01, "random_control_median_drop": 0.02, "latencies_ms": {"gate": 5.0, "reader": 10.0}},
        {"patient_id": "P2", "y_true": 0, "score": 0.8, "judge_accepted": False, "target_drop": 0.05,
         "control_median_drop": 0.04, "random_control_median_drop": 0.01, "latencies_ms": {"gate": 6.0, "reader": 12.0}},
    ]
    ab = run_ablations(recs, n_boot=50)
    assert len(ab["configurations"]) == 6
    fs = compute_fill_sensitivity([
        {"per_method_score_drops": {"telea": 0.1, "gaussian_blur": 0.01, "mean_fill": 0.02}},
        {"per_method_score_drops": {"telea": 0.1, "gaussian_blur": 0.12, "mean_fill": 0.08}},
    ])
    assert fs["n"] == 2 and fs["flip_count"] == 1 and fs["verdict_flip_rate"] == pytest.approx(0.5)
    lat = summarize_latencies(recs)
    assert lat["end_to_end"]["p50_ms"] == pytest.approx(16.5)
