"""Tests for YAML config loading/freezing, validation-only threshold tuning, and capture pilot."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.capture_pilot import evaluate_capture_pilot
from scripts.tune_and_freeze import select_best_judge_thresholds
from veriray.config import configs_from_yaml, freeze_config


def test_default_yaml_loads_and_freeze_refuses_test_split(tmp_path: Path):
    loc, judge, raw = configs_from_yaml(Path("configs/default.yaml"))
    assert raw["finding"] == "pneumothorax"
    assert loc.threshold == 0.5 and judge.num_controls == 19

    with pytest.raises(ValueError, match="locked test split"):
        freeze_config(loc, judge, out_path=tmp_path / "frozen.yaml", fitted_split="test")

    out, digest = freeze_config(loc, judge, out_path=tmp_path / "frozen.yaml", fitted_split="val")
    assert out.is_file() and out.with_suffix(".sha256.json").is_file() and len(digest) == 64
    loc2, judge2, raw2 = configs_from_yaml(out)
    assert raw2["frozen"] is True and loc2 == loc and judge2 == judge


def test_validation_threshold_tuner():
    recs = [
        {"patient_id": f"P{i}", "y_true": 1, "target_drop": 0.12, "control_median_drop": 0.01, "empirical_tail_fraction": 0.05}
        for i in range(6)
    ] + [
        {"patient_id": f"N{i}", "y_true": 0, "target_drop": 0.035, "control_median_drop": 0.01, "empirical_tail_fraction": 0.15}
        for i in range(6)
    ]
    tuned, stats = select_best_judge_thresholds(recs, min_accepted=5)
    assert stats["precision"] == pytest.approx(1.0)
    assert stats["accepted_count"] == 6
    assert tuned.min_target_drop >= 0.05 or tuned.min_margin_over_controls >= 0.04 or tuned.max_empirical_tail_fraction <= 0.10


def test_capture_pilot_metrics_and_limitation():
    rep = evaluate_capture_pilot([
        {"source_image_id": "IMG1", "orig_gate_accepted": True, "phone_gate_accepted": False,
         "orig_verdict": "MODEL-SUPPORTED", "phone_verdict": "INDETERMINATE", "orig_score": 0.7, "phone_score": None},
        {"source_image_id": "IMG2", "orig_gate_accepted": True, "phone_gate_accepted": True,
         "orig_verdict": "NOT SUPPORTED", "phone_verdict": "NOT SUPPORTED", "orig_score": 0.2, "phone_score": 0.24},
    ])
    assert rep["n_pairs"] == 2 and rep["phone_gate_abstained"] == 1 and rep["verdict_flips"] == 1
    assert "NOT evidence of clinical performance" in rep["limitation"]
