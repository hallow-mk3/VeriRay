"""Tests for 512x512 16-finding ChestHDReaderAdapter and FPN+Grad-CAM++ Locator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.conftest import synthetic_xray
from veriray.chest_hd import (
    CHEST_HD_FINDINGS,
    CHEST_HD_RECORD,
    ChestHDNet,
    ChestHDReaderAdapter,
)
from veriray.reader import CheckpointHashMismatchError, CheckpointMissingError, CheckpointRecordError


def _write_fake_chest_hd(directory: Path, img_size: int = 128) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    net = ChestHDNet(num_classes=len(CHEST_HD_FINDINGS), pretrained=False)
    wpath = directory / "convnext_tiny_fpn_chest16_512.pt"
    torch.save(net.state_dict(), wpath)
    sha = hashlib.sha256(wpath.read_bytes()).hexdigest()
    meta = {
        "weights_id": "fake-chest-hd",
        "filename": wpath.name,
        "sha256": sha,
        "img_size": img_size,
        "findings": list(CHEST_HD_FINDINGS),
        "platt_calibrators": {"pneumothorax": {"coef": 1.0, "intercept": 0.0}},
    }
    (directory / CHEST_HD_RECORD).write_text(json.dumps(meta), encoding="utf-8")
    return directory


def test_chest_hd_fail_closed_and_16_findings(tmp_path: Path):
    assert not ChestHDReaderAdapter.is_available(tmp_path / "empty")
    with pytest.raises(CheckpointMissingError):
        ChestHDReaderAdapter.load_for_finding(tmp_path / "empty", "pneumothorax")

    d = _write_fake_chest_hd(tmp_path / "hd")
    assert ChestHDReaderAdapter.is_available(d)

    for fid in ("pneumothorax", "effusion", "cardiomegaly", "tuberculosis", "pneumonia", "nodule_mass"):
        adapter = ChestHDReaderAdapter.load_for_finding(d, fid, device="cpu")
        rc, x, disp = adapter.score(synthetic_xray(256, 256).astype(np.float32))
        assert 0.0 <= rc.score_raw <= 1.0
        assert disp.shape == (224, 224)
        loc = adapter.locate_224(x)
        assert loc.heatmap.shape == (224, 224) and loc.target_mask.shape == (224, 224)

    with pytest.raises(CheckpointRecordError):
        ChestHDReaderAdapter.load_for_finding(d, "unknown_disease", device="cpu")

    # Tamper weights -> hash mismatch
    (d / "convnext_tiny_fpn_chest16_512.pt").write_bytes(b"tampered")
    with pytest.raises(CheckpointHashMismatchError):
        ChestHDReaderAdapter.load_for_finding(d, "pneumothorax", device="cpu")
