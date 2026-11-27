"""Tests for multi-finding Chest X-ray support (pneumothorax, effusion, cardiomegaly)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import synthetic_xray
from veriray.contracts import SUPPORTED_FINDINGS, Verdict
from veriray.pipeline import run_verification
from veriray.reader import CheckpointRecordError, Reader


def test_supported_findings_tuple():
    assert set(SUPPORTED_FINDINGS) == {"pneumothorax", "effusion", "cardiomegaly"}


def test_reader_load_for_finding_retargets_index(fake_models_dir: Path):
    r_eff = Reader.load_for_finding(fake_models_dir, "effusion", device="cpu")
    assert r_eff.record.finding_label == "Effusion"
    assert r_eff.finding_index == r_eff.record.label_list.index("Effusion")
    assert r_eff.calibrator is None  # Pneumothorax calibrator must not transfer

    r_card = Reader.load_for_finding(fake_models_dir, "cardiomegaly", device="cpu")
    assert r_card.record.finding_label == "Cardiomegaly"
    assert r_card.finding_index == r_card.record.label_list.index("Cardiomegaly")
    assert r_card.calibrator is None

    with pytest.raises(CheckpointRecordError):
        Reader.load_for_finding(fake_models_dir, "unknown_finding", device="cpu")


@pytest.mark.parametrize("finding_id", ["pneumothorax", "effusion", "cardiomegaly"])
def test_run_verification_multifinding(fake_models_dir: Path, tmp_path: Path, finding_id: str):
    r = Reader.load_for_finding(fake_models_dir, finding_id, device="cpu")
    res = run_verification(
        synthetic_xray(256, 256, seed=7),
        reader=r,
        models_dir=fake_models_dir,
        manifest_dir=tmp_path / "runs",
        finding_id=finding_id,
    )
    assert res.verdict in set(Verdict)
    if res.judge_outcome is not None:
        assert res.judge_outcome.evidence_record.finding_id == finding_id
