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
from veriray.brain_mri import BRAIN_CLASSES, BRAIN_RECORD, BrainMRIError, BrainMRIReader


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
