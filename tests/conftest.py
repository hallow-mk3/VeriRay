"""Shared fixtures: a randomly initialised DenseNet-121 saved as a *fake* checkpoint (tests only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torchxrayvision as xrv

LABELS = ["Atelectasis", "Consolidation", "", "Pneumothorax", "Edema", "", "", "Effusion", "Pneumonia", "",
          "Cardiomegaly", "", "", "", "Lung Lesion", "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum"]


def write_checkpoint_dir(directory: Path, weights_bytes_override: bytes | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    fname = "fake-densenet121.pt"
    path = directory / fname
    if weights_bytes_override is None:
        torch.manual_seed(0)
        model = xrv.models.DenseNet(weights=None, num_classes=len(LABELS), in_channels=1)
        torch.save(model, path)
    else:
        path.write_bytes(weights_bytes_override)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "CHECKPOINT.json").write_text(json.dumps({
        "weights_id": "fake-test-weights", "filename": fname, "source_url": "none", "license": "test only",
        "training_datasets": ["synthetic-random-init"], "label_list": LABELS, "finding_label": "Pneumothorax",
        "finding_index": 3, "input_resolution": 224, "sha256": digest,
    }), encoding="utf-8")
    return directory


@pytest.fixture(scope="session")
def fake_models_dir(tmp_path_factory) -> Path:
    return write_checkpoint_dir(tmp_path_factory.mktemp("models_ok"))


@pytest.fixture(scope="session")
def fake_reader(fake_models_dir):
    from veriray.reader import Reader

    return Reader.load(fake_models_dir)


def synthetic_xray(h: int = 320, w: int = 400, seed: int = 1) -> np.ndarray:
    """Smooth gradient + blobs + noise, uint8; passes the quality gate."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = 90 + 50 * np.sin(xx / 37.0) * np.cos(yy / 29.0) + 40 * np.exp(-(((xx - w / 2) / 90) ** 2 + ((yy - h / 2) / 110) ** 2))
    img += rng.normal(0, 6, size=(h, w))
    return np.clip(img, 0, 255).astype(np.uint8)
