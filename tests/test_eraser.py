"""Eraser tests: exact outside-mask pixel preservation (automated, many random masks), fail-closed, labelling."""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import synthetic_xray
from veriray import eraser as eraser_mod
from veriray.contracts import SYNTHETIC_COUNTERFACTUAL_LABEL, check_outside_mask_exact_equality
from veriray.eraser import METHODS, EraseError, erase, render_labelled


def _img(seed=0, size=224):
    return synthetic_xray(size, size, seed).astype(np.float32)


def _random_mask(rng, size=224):
    m = np.zeros((size, size), bool)
    for _ in range(rng.integers(1, 4)):
        cy, cx = rng.integers(0, size, 2)  # includes centres on/near borders
        ry, rx = rng.integers(3, 50, 2)
        yy, xx = np.ogrid[:size, :size]
        m |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
    return m if (m.any() and not m.all()) else _random_mask(rng, size)


@pytest.mark.parametrize("method", METHODS)
def test_outside_mask_pixels_exactly_identical_many_random_masks(method):
    rng = np.random.default_rng(123)
    for i in range(25):
        img, mask = _img(i), _random_mask(rng)
        res = erase(img, mask, method, seed=i)
        exact, diff = check_outside_mask_exact_equality(img, res.edited_image, mask)
        assert exact and diff == 0.0
        assert np.array_equal(res.edited_image[~mask], img[~mask])  # bit-for-bit
        assert res.integrity_report.outside_mask_exact_equality
        assert res.synthetic_label == SYNTHETIC_COUNTERFACTUAL_LABEL


@pytest.mark.parametrize("method", METHODS)
def test_edit_actually_changes_masked_region_and_is_in_range(method):
    img = _img()
    mask = np.zeros(img.shape, bool)
    mask[80:140, 90:160] = True
    res = erase(img, mask, method)
    assert not np.array_equal(res.edited_image[mask], img[mask])
    assert 0.0 <= res.edited_image.min() and res.edited_image.max() <= 255.0
    assert np.isfinite(res.edited_image).all() and res.edited_image.dtype == np.float32
    assert 0.0 < res.integrity_report.ssim_descriptive_proxy < 1.0
    assert res.integrity_report.boundary_discontinuity >= 0.0


def test_mean_fill_is_outside_mean_and_deterministic():
    img = _img()
    mask = np.zeros(img.shape, bool)
    mask[50:100, 50:100] = True
    a, b = erase(img, mask, "mean_fill"), erase(img, mask, "mean_fill")
    assert np.array_equal(a.edited_image, b.edited_image)
    assert np.allclose(a.edited_image[mask], img[~mask].mean())


def test_blur_does_not_leak_masked_content():
    img = np.full((224, 224), 100.0, np.float32)
    mask = np.zeros(img.shape, bool)
    mask[90:130, 90:130] = True
    img[mask] = 250.0  # bright content inside the mask must not influence the fill
    res = erase(img, mask, "gaussian_blur")
    assert np.allclose(res.edited_image[mask], 100.0, atol=1e-3)


@pytest.mark.parametrize("method", METHODS)
def test_fail_closed_on_bad_inputs(method):
    img = _img()
    ok = np.zeros(img.shape, bool)
    ok[10:20, 10:20] = True
    cases = {
        "ERASER_EMPTY_MASK": (img, np.zeros(img.shape, bool)),
        "ERASER_FULL_MASK": (img, np.ones(img.shape, bool)),
        "ERASER_BAD_MASK": (img, np.ones((5, 5), bool)),
        "ERASER_BAD_IMAGE": (np.full(img.shape, np.nan, np.float32), ok),
    }
    for code, (i, m) in cases.items():
        with pytest.raises(EraseError) as e:
            erase(i, m, method)
        assert e.value.reason_code == code
    with pytest.raises(EraseError) as e:
        erase(img * 2, ok, method)  # out of [0,255]
    assert e.value.reason_code == "ERASER_BAD_IMAGE"
    with pytest.raises(EraseError) as e:
        erase(np.zeros((3, 3, 3), np.float32), ok, method)
    assert e.value.reason_code == "ERASER_BAD_IMAGE"


def test_unknown_method_fails_closed():
    with pytest.raises(EraseError) as e:
        erase(_img(), np.pad(np.ones((4, 4), bool), 10)[:224, :224].copy() if False else _small_mask(), "diffusion")
    assert e.value.reason_code == "ERASER_UNKNOWN_METHOD"


def _small_mask():
    m = np.zeros((224, 224), bool)
    m[5:9, 5:9] = True
    return m


def test_safety_net_catches_a_leaky_composite(monkeypatch):
    """If a future change ever modified pixels outside the mask, the run must fail, not return."""
    def leaky(original, fill, mask):
        out = np.where(mask, fill, original).astype(np.float32)
        out[0, 0] = (out[0, 0] + 1.0) % 255.0 if not mask[0, 0] else out[0, 0]
        out[-1, -1] = (out[-1, -1] + 1.0) % 255.0 if not mask[-1, -1] else out[-1, -1]
        return out

    monkeypatch.setattr(eraser_mod, "_composite", leaky)
    with pytest.raises(EraseError) as e:
        erase(_img(), _small_mask(), "mean_fill")
    assert e.value.reason_code == "ERASER_OUTSIDE_MASK_CHANGED"


def test_unexpected_error_is_wrapped(monkeypatch):
    monkeypatch.setattr(eraser_mod, "_fill_mean", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(EraseError) as e:
        erase(_img(), _small_mask(), "mean_fill")
    assert e.value.reason_code == "ERASER_UNEXPECTED_ERROR"


def test_render_labelled_adds_banner_without_touching_source():
    img = _img()
    res = erase(img, _small_mask(), "telea")
    before = res.edited_image.copy()
    rgb = render_labelled(res.edited_image, scale=2)
    assert np.array_equal(res.edited_image, before)  # source array untouched
    assert rgb.shape == (448, 448, 3) and rgb.dtype == np.uint8
    banner = rgb[:20]
    assert (banner[..., 0] > 100).mean() > 0.5 and (banner[..., 1] < 60).mean() > 0.3  # red band present
    assert (rgb[40:, :, 0] != rgb[40:, :, 1]).sum() == 0  # below the banner it is still grayscale


def test_eraser_module_has_no_network():
    import inspect

    src = inspect.getsource(eraser_mod)
    assert "urllib" not in src and "requests" not in src and "socket" not in src
