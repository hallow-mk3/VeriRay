"""Tests for validation-only Platt calibrator and test-split leakage guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from veriray.calibration import PlattCalibrator, fit_platt_calibrator


def test_calibrator_refuses_test_split_and_roundtrips(tmp_path: Path):
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 1, 1, 1]
    with pytest.raises(ValueError, match="test split"):
        fit_platt_calibrator(scores, labels, split_name="test")

    cal = fit_platt_calibrator(scores, labels, split_name="val")
    assert cal.calibrate(0.9) > cal.calibrate(0.1)
    assert 0.0 <= cal.calibrate(0.5) <= 1.0

    p = cal.save(tmp_path / "cal.json")
    loaded = PlattCalibrator.load(p)
    assert loaded.calibrate(0.8) == pytest.approx(cal.calibrate(0.8))

    with pytest.raises(ValueError, match="test split"):
        PlattCalibrator(coef=1.0, intercept=0.0, fitted_split="test", num_samples=10)


def test_reader_loads_optional_platt_calibrator(fake_models_dir: Path):
    from tests.conftest import synthetic_xray
    from veriray.reader import Reader

    cal = fit_platt_calibrator([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], [0, 0, 0, 1, 1, 1], split_name="val")
    cal.save(fake_models_dir / "platt_calibrator.json")

    reader = Reader.load(fake_models_dir, device="cpu")
    rc, _, _ = reader.score(synthetic_xray(224, 224, seed=2))
    assert rc.score_calibrated_optional is not None
    assert 0.0 <= rc.score_calibrated_optional <= 1.0
    assert rc.threshold_id == cal.calibrator_id
