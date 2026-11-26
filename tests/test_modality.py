"""Tests for veriray.modality — lightweight modality mismatch detector."""

from __future__ import annotations

import numpy as np
import pytest

from veriray.modality import ModalityGuess, guess_modality


# ---------------------------------------------------------------------------
# Synthetic image builders
# ---------------------------------------------------------------------------

def _make_mri_like(size: int = 256) -> np.ndarray:
    """Synthetic MRI-like image: dark background, bright circular brain mass."""
    img = np.zeros((size, size), dtype=np.float32)
    cy, cx = size // 2, size // 2
    radius = size // 3
    ys, xs = np.ogrid[:size, :size]
    mask = (ys - cy) ** 2 + (xs - cx) ** 2 < radius ** 2
    rng = np.random.default_rng(42)
    img[mask] = rng.normal(loc=140.0, scale=30.0, size=int(mask.sum())).clip(60, 220)
    return img


def _make_xray_like(size: int = 256, width: int = 320) -> np.ndarray:
    """Synthetic CXR-like image: bright background, dark bilateral lung fields."""
    rng = np.random.default_rng(7)
    img = rng.normal(loc=160.0, scale=25.0, size=(size, width)).astype(np.float32).clip(0, 255)
    # Dark bilateral lung fields
    h, w = img.shape
    lung_top, lung_bot = h // 4, 3 * h // 4
    img[lung_top:lung_bot, w // 8 : w // 3] = rng.normal(60, 15, (lung_bot - lung_top, w // 3 - w // 8)).clip(0, 100)
    img[lung_top:lung_bot, 2 * w // 3 : 7 * w // 8] = rng.normal(60, 15, (lung_bot - lung_top, 7 * w // 8 - 2 * w // 3)).clip(0, 100)
    return img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestModalityGuess:
    def test_mri_detected(self) -> None:
        img = _make_mri_like()
        result = guess_modality(img)
        assert isinstance(result, ModalityGuess)
        assert result.is_brain_mri_suspected is True, (
            f"Expected Brain MRI detected; got confidence={result.confidence:.2f}, reason={result.reason}"
        )
        assert result.confidence >= 0.55

    def test_xray_not_flagged(self) -> None:
        img = _make_xray_like()
        result = guess_modality(img)
        assert isinstance(result, ModalityGuess)
        assert result.is_brain_mri_suspected is False, (
            f"Expected Chest X-ray not flagged; got confidence={result.confidence:.2f}, reason={result.reason}"
        )

    def test_signals_keys_present(self) -> None:
        img = _make_mri_like()
        result = guess_modality(img)
        for key in ("aspect", "mean", "std", "dark_frac", "dark_hist_frac", "centre_rim_ratio"):
            assert key in result.signals, f"Missing signal key: {key}"

    def test_accepts_rgb_input(self) -> None:
        """guess_modality should handle H×W×3 input by averaging channels."""
        base = _make_mri_like(128)
        rgb = np.stack([base, base, base], axis=2)
        result = guess_modality(rgb)
        assert isinstance(result, ModalityGuess)

    def test_accepts_float01_input(self) -> None:
        """Should handle images normalised to [0, 1] range."""
        img = _make_mri_like() / 255.0
        result = guess_modality(img)
        assert isinstance(result, ModalityGuess)
        assert result.is_brain_mri_suspected is True

    def test_pure_black_does_not_crash(self) -> None:
        img = np.zeros((224, 224), dtype=np.float32)
        result = guess_modality(img)
        assert isinstance(result, ModalityGuess)

    def test_pure_white_does_not_crash(self) -> None:
        img = np.full((224, 224), 255.0, dtype=np.float32)
        result = guess_modality(img)
        assert isinstance(result, ModalityGuess)

    def test_confidence_range(self) -> None:
        for img in [_make_mri_like(), _make_xray_like()]:
            result = guess_modality(img)
            assert 0.0 <= result.confidence <= 1.0

    def test_reason_string_non_empty(self) -> None:
        img = _make_mri_like()
        result = guess_modality(img)
        assert len(result.reason) > 0
