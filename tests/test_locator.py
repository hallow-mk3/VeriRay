"""Locator tests: Grad-CAM++ shape/range, and quality flags on synthetic heatmaps."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import synthetic_xray
from veriray.contracts import LocatorContract
from veriray.locator import (
    LocatorConfig,
    gradcam_plus_plus,
    heatmap_to_mask,
    localization_flags,
    locate,
    normalize_heatmap,
)
from veriray.reader import preprocess

CFG = LocatorConfig()


def _blob(radius: float, center=(112, 112), size=224) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.hypot(yy - center[0], xx - center[1])
    return np.clip(1.0 - d / radius, 0, 1).astype(np.float32)


def _flags(hm):
    mask = heatmap_to_mask(hm, CFG)
    return mask, localization_flags(hm, mask, CFG)


def test_empty_heatmap_flags_empty():
    mask, f = _flags(np.zeros((224, 224), np.float32))
    assert not mask.any() and f["empty"]
    assert LocatorContract(np.zeros((224, 224), np.float32), mask, f).is_weak_or_unstable


def test_tiny_blob_flags_empty():
    hm = np.zeros((224, 224), np.float32)
    hm[100:104, 100:104] = 1.0  # 16 px << 0.5% of image and < min component size
    mask, f = _flags(hm)
    assert f["empty"]


def test_huge_mask_flags_implausibly_large():
    hm = np.ones((224, 224), np.float32)  # high almost everywhere\n    hm[0, 0] = 0.0
    mask, f = _flags(hm)
    assert f["implausibly_large"] and f["near_full_image"]
    assert LocatorContract(hm, mask, f).is_weak_or_unstable


def test_moderately_large_mask_flags_large_but_not_near_full():
    hm = np.zeros((224, 224), np.float32)
    hm[:, :150] = 1.0  # ~67% of image
    mask, f = _flags(hm)
    assert f["implausibly_large"] and not f["near_full_image"]


def test_diffuse_heatmap_flags_diffuse():
    rng = np.random.default_rng(0)
    hm = np.clip(0.55 + 0.45 * rng.random((224, 224)), 0, 1).astype(np.float32)
    _, f = _flags(normalize_heatmap(hm))
    assert f["diffuse"]


def test_compact_blob_is_not_flagged():
    hm = normalize_heatmap(_blob(radius=45))
    mask, f = _flags(hm)
    assert mask.any() and not (f["empty"] or f["diffuse"] or f["implausibly_large"])
    c = LocatorContract(hm, mask, f)
    assert not c.is_weak_or_unstable
    # Step 2 must not claim stability / anatomy were checked.
    assert f["stability_assessed"] is False and f["anatomy_assessed"] is False


def test_normalize_flat_map_is_zeros():
    assert not normalize_heatmap(np.full((8, 8), 3.0, np.float32)).any()


def test_gradcampp_on_synthetic_image(fake_reader):
    x, _ = preprocess(synthetic_xray().astype(np.float32))
    hm = gradcam_plus_plus(fake_reader, x)
    assert hm.shape == (224, 224) and hm.dtype == np.float32 and np.isfinite(hm).all()
    assert 0.0 <= hm.min() and hm.max() <= 1.0
    again = gradcam_plus_plus(fake_reader, x)
    assert np.array_equal(hm, again)  # deterministic
    c = locate(fake_reader, x)
    assert c.target_mask.shape == (224, 224) and c.target_mask.dtype == np.bool_


def test_mask_creation_uses_no_ground_truth():
    import inspect

    assert list(inspect.signature(heatmap_to_mask).parameters) == ["heatmap", "cfg"]
