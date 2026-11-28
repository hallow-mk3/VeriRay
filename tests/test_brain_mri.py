"""Tests for offline Brain Tumor MRI Reader + Grad-CAM++ + fail-closed hash check."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.models as tvm

from tests.conftest import synthetic_xray
from veriray.brain_mri import (
    BRAIN_CLASSES,
    BRAIN_RECORD,
    BrainMRIError,
    BrainMRIReader,
    decode_brain_upload,
)


def test_brain_mri_fails_closed_on_missing_or_tampered_weights(tmp_path: Path):
    with pytest.raises(BrainMRIError) as e1:
        BrainMRIReader.load(tmp_path / "none")
    assert e1.value.reason_code == "BRAIN_CHECKPOINT_MISSING"

    d = tmp_path / "bm"
    d.mkdir()
    m = tvm.efficientnet_b0(weights=None)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, len(BRAIN_CLASSES))
    wpath = d / "efficientnet_b0-brain-mri-4cls.pt"
    torch.save(m.state_dict(), wpath)
    digest = hashlib.sha256(wpath.read_bytes()).hexdigest()
    (d / BRAIN_RECORD).write_text(json.dumps({"filename": wpath.name, "sha256": digest, "classes": list(BRAIN_CLASSES)}))

    reader = BrainMRIReader.load(d, device="cpu")
    out = reader.predict_and_cam(synthetic_xray(256, 256).astype(np.float32))
    assert out["predicted_class"] in BRAIN_CLASSES
    assert out["heatmap"].shape == (256, 256) and out["target_mask"].shape == (256, 256)

    wpath.write_bytes(b"corrupt")
    with pytest.raises(BrainMRIError) as e2:
        BrainMRIReader.load(d)
    assert e2.value.reason_code == "BRAIN_HASH_MISMATCH"


def test_brain_mri_fails_closed_on_bad_record_and_bad_input(tmp_path: Path):
    d = tmp_path / "bm_bad"
    d.mkdir()
    (d / BRAIN_RECORD).write_text("{not json")
    with pytest.raises(BrainMRIError) as e_rec:
        BrainMRIReader.load(d)
    assert e_rec.value.reason_code == "BRAIN_RECORD_INVALID"

    m = tvm.densenet121(weights=None)
    m.classifier = torch.nn.Linear(m.classifier.in_features, len(BRAIN_CLASSES))
    wpath = d / "densenet121-brain-mri-4cls.pt"
    torch.save(m.state_dict(), wpath)
    digest = hashlib.sha256(wpath.read_bytes()).hexdigest()
    (d / BRAIN_RECORD).write_text(json.dumps({"filename": wpath.name, "sha256": digest, "classes": list(BRAIN_CLASSES)}))

    reader = BrainMRIReader.load(d, device="cpu")
    with pytest.raises(BrainMRIError) as e_bad:
        reader.predict_and_cam(np.full((64, 64), np.nan, dtype=np.float32))
    assert e_bad.value.reason_code == "BRAIN_BAD_INPUT"


def test_brain_upload_gate_blocks_color_tiny_and_gzip_bomb_in_memory():
    import gzip
    import io
    import nibabel as nib
    from PIL import Image

    # 1. Multi-hue RGB photo rejected
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    rgb[..., 0] = 210
    rgb[..., 1] = 140
    rgb[..., 2] = 95
    buf_rgb = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf_rgb, format="PNG")
    with pytest.raises(BrainMRIError) as e_rgb:
        decode_brain_upload(buf_rgb.getvalue(), "selfie.png")
    assert e_rgb.value.reason_code == "NON_MRI_COLOR_IMAGE"

    # 2. Tiny 2x2 image rejected
    buf_tiny = io.BytesIO()
    Image.fromarray(np.array([[10, 240], [120, 60]], dtype=np.uint8), mode="L").save(buf_tiny, format="PNG")
    with pytest.raises(BrainMRIError) as e_tiny:
        decode_brain_upload(buf_tiny.getvalue(), "tiny.png")
    assert e_tiny.value.reason_code == "IMAGE_TOO_SMALL"

    # 3. Gzip bomb (> 128 MB decompressed from tiny compressed payload) aborted safely
    bomb_gz = gzip.compress(b"\x00" * (130 * 1024 * 1024), compresslevel=9)
    assert len(bomb_gz) < 1024 * 1024
    with pytest.raises(BrainMRIError) as e_bomb:
        decode_brain_upload(bomb_gz, "bomb.nii.gz")
    assert e_bomb.value.reason_code == "IMAGE_TOO_LARGE_OR_BOMB"

    # 4. Valid 3D .nii.gz decoded 100% in memory without touching tempfile
    rng = np.random.default_rng(42)
    vol = rng.normal(120.0, 30.0, size=(64, 64, 16)).astype(np.float32)
    nii_img = nib.Nifti1Image(vol, np.eye(4))
    bio = io.BytesIO()
    fh = nib.FileHolder(fileobj=bio)
    nii_img.to_file_map({"header": fh, "image": fh})
    gz_valid = gzip.compress(bio.getvalue(), compresslevel=1)
    sl = decode_brain_upload(gz_valid, "valid_scan.nii.gz")
    assert sl.shape == (64, 64) and sl.dtype == np.float32
