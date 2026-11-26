"""Tests for Gated-Convolution U-Net Eraser: exact outside-mask preservation and fail-closed loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.conftest import synthetic_xray
from veriray.contracts import SYNTHETIC_COUNTERFACTUAL_LABEL, check_outside_mask_exact_equality
from veriray.eraser import EraseError
from veriray.unet_eraser import UNET_RECORD, GatedUNet, UNetEraser


def _write_fake_unet(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    m = GatedUNet(base_ch=8)
    wpath = directory / "fake-unet.pt"
    torch.save(m.state_dict(), wpath)
    digest = hashlib.sha256(wpath.read_bytes()).hexdigest()
    (directory / UNET_RECORD).write_text(
        json.dumps({"filename": wpath.name, "sha256": digest, "base_ch": 8}),
        encoding="utf-8",
    )
    return directory


def test_unet_fails_closed_on_missing_or_corrupt_weights(tmp_path: Path):
    with pytest.raises(EraseError) as e1:
        UNetEraser.load(tmp_path / "none")
    assert e1.value.reason_code == "ERASER_UNET_CHECKPOINT_MISSING"

    d = _write_fake_unet(tmp_path / "u")
    (d / "fake-unet.pt").write_bytes(b"tampered")
    with pytest.raises(EraseError) as e2:
        UNetEraser.load(d)
    assert e2.value.reason_code == "ERASER_UNET_HASH_MISMATCH"


def test_unet_eraser_preserves_outside_mask_pixels_identically(tmp_path: Path):
    d = _write_fake_unet(tmp_path / "u")
    eraser = UNetEraser.load(d)
    img = synthetic_xray(224, 224, seed=5).astype(np.float32)
    mask = np.zeros((224, 224), bool)
    mask[70:140, 80:150] = True
    res = eraser.erase(img, mask, seed=1)
    exact, diff = check_outside_mask_exact_equality(img, res.edited_image, mask)
    assert exact and diff == 0.0
    assert res.fill_method == "gated_unet"
    assert res.synthetic_label == SYNTHETIC_COUNTERFACTUAL_LABEL


def test_unet_eraser_has_no_network_imports():
    import re

    src = (Path(__file__).resolve().parents[1] / "src" / "veriray" / "unet_eraser.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(?:import|from)\s+(?:urllib|requests|socket|http)", src, re.M)
