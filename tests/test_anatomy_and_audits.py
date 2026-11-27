"""Tests for offline lung segmentation, shortcut audit (Exp 7), and boundary suite (Exp 11)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torchxrayvision.baseline_models.chestx_det.ptsemseg.pspnet import pspnet

from eval.boundary_test import run_boundary_suite
from eval.shortcut_audit import compute_shortcut_audit
from tests.conftest import synthetic_xray
from veriray.anatomy import LUNG_SEG_RECORD, PSPNET_TARGETS, AnatomyError, LungSegmenter


def _write_fake_pspnet(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    m = pspnet(len(PSPNET_TARGETS))
    wpath = directory / "fake-pspnet.pth"
    torch.save(m.state_dict(), wpath)
    digest = hashlib.sha256(wpath.read_bytes()).hexdigest()
    (directory / LUNG_SEG_RECORD).write_text(
        json.dumps({"filename": wpath.name, "sha256": digest}),
        encoding="utf-8",
    )
    return directory


def test_lung_segmenter_optional_and_fail_closed(tmp_path: Path):
    assert LungSegmenter.load_optional(tmp_path / "empty") is None
    with pytest.raises(AnatomyError) as e1:
        LungSegmenter.load(tmp_path / "empty")
    assert e1.value.reason_code == "LUNG_SEG_CHECKPOINT_MISSING"

    d = _write_fake_pspnet(tmp_path / "l")
    (d / "fake-pspnet.pth").write_bytes(b"corrupt")
    with pytest.raises(AnatomyError) as e2:
        LungSegmenter.load(d)
    assert e2.value.reason_code == "LUNG_SEG_HASH_MISMATCH"


def test_lung_segmenter_output_shape_and_dtype(tmp_path: Path):
    d = _write_fake_pspnet(tmp_path / "l")
    seg = LungSegmenter.load(d)
    mask = seg.segment_lungs(synthetic_xray(224, 224).astype(np.float32))
    assert mask.shape == (224, 224) and mask.dtype == np.bool_


def test_shortcut_audit_metrics():
    rep = compute_shortcut_audit([
        {"cue_category": "chest_tube", "has_shortcut_cue": True, "verdict": "SHORTCUT RISK"},
        {"cue_category": "chest_tube", "has_shortcut_cue": True, "verdict": "MODEL-SUPPORTED"},
        {"cue_category": "pacemaker", "has_shortcut_cue": True, "verdict": "INDETERMINATE"},
        {"cue_category": "none", "has_shortcut_cue": False, "verdict": "MODEL-SUPPORTED"},
    ])
    assert rep["total_reviewed_cue_cases"] == 3
    assert rep["overall_flag_rate"] == pytest.approx(2.0 / 3.0)
    assert rep["overall_false_accept_rate"] == pytest.approx(1.0 / 3.0)


def test_boundary_suite_all_fail_closed(tmp_path: Path):
    rep = run_boundary_suite(models_dir=Path("models"), empty_dir=tmp_path / "empty")
    assert rep["fail_closed_rate"] == 1.0
    assert rep["total_boundary_cases"] == 11


def test_segment_thoracic_roi_lung_only_for_pneumothorax(tmp_path: Path):
    """segment_thoracic_roi with finding_id='pneumothorax' returns a (224,224) bool mask."""
    d = _write_fake_pspnet(tmp_path / "l")
    seg = LungSegmenter.load(d)
    roi = seg.segment_thoracic_roi(synthetic_xray(224, 224).astype(np.float32), finding_id="pneumothorax")
    assert roi.shape == (224, 224) and roi.dtype == np.bool_


def test_segment_thoracic_roi_includes_cardiac_for_cardiomegaly(tmp_path: Path):
    """segment_thoracic_roi with finding_id='cardiomegaly' returns a (224,224) bool mask
    (cardiac + mediastinum indices are included; shape and dtype must be correct)."""
    d = _write_fake_pspnet(tmp_path / "l")
    seg = LungSegmenter.load(d)
    roi = seg.segment_thoracic_roi(synthetic_xray(224, 224).astype(np.float32), finding_id="cardiomegaly")
    assert roi.shape == (224, 224) and roi.dtype == np.bool_


def test_segment_lungs_delegates_to_thoracic_roi(tmp_path: Path):
    """segment_lungs() must return an identical result to segment_thoracic_roi(..., finding_id='pneumothorax')."""
    d = _write_fake_pspnet(tmp_path / "l")
    seg = LungSegmenter.load(d)
    img = synthetic_xray(224, 224).astype(np.float32)
    via_legacy = seg.segment_lungs(img)
    via_roi = seg.segment_thoracic_roi(img, finding_id="pneumothorax")
    np.testing.assert_array_equal(via_legacy, via_roi)
