"""Eraser baselines (Blueprint 26 Nov §5 module 4, v2 §7): synthetic counterfactual edits.

Methods: ``gaussian_blur`` (normalized-convolution blur inside the mask), ``mean_fill`` (mean of
outside-mask pixels), ``telea`` (OpenCV Telea inpainting). U-Net / diffusion are NOT built yet.

Guarantees:
- The composite is ``np.where(mask, fill, original)``, so every pixel outside the mask is exactly
  the original. This is re-checked after compositing and any violation raises :class:`EraseError`.
- Fail closed: empty/full/mismatched masks, non-finite or out-of-range images, unknown methods or
  any internal error raise :class:`EraseError` with a machine-readable ``reason_code``.
- The returned array is never labelled in-place (that would break pixel equality). Use
  :func:`render_labelled` for anything shown to a user: it adds the banner
  ``SYNTHETIC COUNTERFACTUAL — NOT A REAL SCAN``.
- Image-quality numbers are descriptive proxies only. No edit is a realistic or disease-free
  reconstruction of anatomy.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

from veriray.contracts import (
    SYNTHETIC_COUNTERFACTUAL_LABEL,
    EraserContract,
    IntegrityReport,
    check_outside_mask_exact_equality,
)

METHODS = ("gaussian_blur", "mean_fill", "telea")


class EraseError(Exception):
    """``reason_code`` is the machine-readable INDETERMINATE reason."""

    def __init__(self, reason_code: str, message: str = "") -> None:
        super().__init__(f"{reason_code}: {message}" if message else reason_code)
        self.reason_code = reason_code


def _validate(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        raise EraseError("ERASER_BAD_IMAGE", "image must be a non-empty 2D array")
    img = image.astype(np.float32)
    if not np.all(np.isfinite(img)) or img.min() < 0.0 or img.max() > 255.0:
        raise EraseError("ERASER_BAD_IMAGE", "image must be finite and in [0, 255]")
    if not isinstance(mask, np.ndarray) or mask.shape != img.shape:
        raise EraseError("ERASER_BAD_MASK", "mask must be an ndarray with the same shape as the image")
    m = mask.astype(bool)
    if not m.any():
        raise EraseError("ERASER_EMPTY_MASK")
    if m.all():
        raise EraseError("ERASER_FULL_MASK", "nothing outside the mask to borrow content from")
    return img, m


def _fill_gaussian_blur(img: np.ndarray, m: np.ndarray, sigma: float) -> np.ndarray:
    """Blur using only outside-mask pixels (normalized convolution), so masked content does not leak."""
    w = (~m).astype(np.float32)
    num = cv2.GaussianBlur(img * w, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(w, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    fallback = float(img[~m].mean())
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), fallback).astype(np.float32)


def _fill_mean(img: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.full(img.shape, float(img[~m].mean()), dtype=np.float32)


def _fill_telea(img: np.ndarray, m: np.ndarray, radius: float) -> np.ndarray:
    u8 = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    out = cv2.inpaint(u8, m.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
    return out.astype(np.float32)


def _composite(original: np.ndarray, fill: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, np.clip(fill, 0.0, 255.0), original).astype(np.float32)


def _boundary_discontinuity(edited: np.ndarray, mask: np.ndarray) -> float:
    """Mean |inner-ring pixel - adjacent outer-ring pixel| / 255 along the mask edge (descriptive)."""
    k = np.ones((3, 3), np.uint8)
    m8 = mask.astype(np.uint8)
    inner = mask & ~cv2.erode(m8, k).astype(bool)
    outer = cv2.dilate(m8, k).astype(bool) & ~mask
    if not inner.any() or not outer.any():
        return 0.0
    return float(abs(edited[inner].mean() - edited[outer].mean()) / 255.0)


def erase(
    image: np.ndarray,
    mask: np.ndarray,
    method: str,
    *,
    seed: int = 0,
    sigma: float = 8.0,
    telea_radius: float = 3.0,
    unrelated_output_max_drift: float = 0.0,
) -> EraserContract:
    """Create one synthetic counterfactual. ``image`` is 2D float/uint8 in [0, 255]."""
    try:
        if method not in METHODS:
            raise EraseError("ERASER_UNKNOWN_METHOD", f"{method!r} not in {METHODS}")
        original, m = _validate(image, mask)
        if method == "gaussian_blur":
            fill = _fill_gaussian_blur(original, m, sigma)
        elif method == "mean_fill":
            fill = _fill_mean(original, m)
        else:
            fill = _fill_telea(original, m, telea_radius)
        edited = _composite(original, fill, m)
        exact, max_diff = check_outside_mask_exact_equality(original, edited, m)
        if not exact:
            raise EraseError("ERASER_OUTSIDE_MASK_CHANGED", f"max abs diff outside mask = {max_diff}")
        if not np.all(np.isfinite(edited)):
            raise EraseError("ERASER_NON_FINITE_OUTPUT")
        report = IntegrityReport(
            outside_mask_exact_equality=True,
            outside_mask_max_abs_diff=0.0,
            boundary_discontinuity=_boundary_discontinuity(edited, m),
            unrelated_output_max_drift=float(unrelated_output_max_drift),
            ssim_descriptive_proxy=float(structural_similarity(original, edited, data_range=255.0)),
        )
        return EraserContract(
            edited_image=edited,
            mask=m,
            fill_method=method,
            seed=int(seed),
            integrity_report=report,
            config={
                "method": method,
                "sigma": sigma if method == "gaussian_blur" else None,
                "telea_radius": telea_radius if method == "telea" else None,
                "deterministic": True,
                "mask_area_fraction": float(m.mean()),
                "unrelated_drift_measured": False,  # Eraser alone cannot measure it; the Judge must
            },
        )
    except EraseError:
        raise
    except Exception as exc:  # noqa: BLE001 - fail closed on anything unexpected
        raise EraseError("ERASER_UNEXPECTED_ERROR", f"{type(exc).__name__}: {exc}") from exc


def render_labelled(edited_image: np.ndarray, scale: int = 1) -> np.ndarray:
    """RGB uint8 copy of an edited image with a visible banner. For DISPLAY/EXPORT only."""
    base = np.clip(np.rint(edited_image), 0, 255).astype(np.uint8)
    if scale > 1:
        base = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    rgb = Image.fromarray(np.stack([base] * 3, axis=-1))
    w, h = rgb.size
    size = max(10, h // 18)
    font: Callable | ImageFont.ImageFont
    try:
        font = ImageFont.load_default(size=size)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(rgb)
    text = SYNTHETIC_COUNTERFACTUAL_LABEL
    bbox = draw.textbbox((0, 0), text, font=font)
    band_h = (bbox[3] - bbox[1]) + 8
    draw.rectangle([0, 0, w, band_h], fill=(160, 0, 0))
    draw.text((max(4, (w - (bbox[2] - bbox[0])) // 2), 4 - bbox[1]), text, fill=(255, 255, 255), font=font)
    return np.asarray(rgb)
