"""Input Gate (Blueprint v2.0 §4, §8.1 step 1): validate file, dimensions, view and basic quality.

Returns ``(InputContract | None, InputGateContract)``. Fail closed: every problem is
reported as REJECT or RETAKE with machine-readable reason codes; this module never
raises on bad input. Thresholds in :class:`GateConfig` are PROVISIONAL, set a priori
and not tuned on any evaluation data (see docs/PROTOCOL.md).
"""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image

from veriray.contracts import (
    ALLOWED_METADATA_KEYS,
    InputContract,
    InputGateContract,
    InputGateStatus,
)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"})
SUPPORTED_VIEWS: frozenset[str] = frozenset({"PA", "AP", "UNKNOWN"})
MAX_INPUT_BYTES: int = 50 * 1024 * 1024
MAX_IMAGE_PIXELS: int = 8192 * 8192


@dataclass(frozen=True)
class GateConfig:
    min_height: int = 224
    min_width: int = 224
    max_height: int = 8192
    max_width: int = 8192
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    min_mean_intensity: float = 20.0  # below => too dark
    max_underexposed_fraction: float = 0.55
    max_mean_intensity: float = 205.0  # above => too bright
    max_overexposed_fraction: float = 0.28
    min_std_intensity: float = 8.0
    min_dynamic_range: float = 40.0
    max_clipped_fraction: float = 0.35
    min_laplacian_var: float = 1.0


def _reject(codes: list[str], flags: Mapping[str, bool] | None = None) -> tuple[None, InputGateContract]:
    return None, InputGateContract(InputGateStatus.REJECT, tuple(codes), dict(flags or {}))


def _retake(codes: list[str], flags: Mapping[str, bool]) -> tuple[None, InputGateContract]:
    return None, InputGateContract(InputGateStatus.RETAKE, tuple(codes), dict(flags))


def _burned_in_text_suspected(gray: np.ndarray) -> bool:
    """Heuristic: >=5 character-sized bright glyphs in the top/bottom margin bands."""
    h, w = gray.shape
    u8 = np.clip(np.round(gray), 0, 255).astype(np.uint8)
    band_h = max(16, int(0.14 * h))
    glyphs = 0
    for band in (u8[:band_h], u8[h - band_h :]):
        n, _, stats, _ = cv2.connectedComponentsWithStats((band >= 225).astype(np.uint8), connectivity=8)
        for i in range(1, n):
            bw, bh, area = (int(stats[i, k]) for k in (cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA))
            if 2 <= bw <= max(18, int(0.08 * w)) and 5 <= bh <= max(28, int(0.10 * h)):
                if 0.15 <= area / float(max(bw * bh, 1)) <= 0.85:
                    glyphs += 1
    return glyphs >= 5


def _rgb_to_gray_or_reason(rgb: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Accept grayscale-in-RGB or a uniformly tinted film/monitor photo; reject multi-hue scenes."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    spread = float(np.mean(rgb.max(axis=2) - rgb.min(axis=2)))
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    if spread < 2.0:
        return gray, "RGB_IDENTICAL_GRAYSCALE"
    chroma_std = max(float(np.std(r - g)), float(np.std(g - b)), float(np.std(r - b)))
    if min(float(np.std(c)) for c in (r, g, b)) > 2.0:
        corr = min(
            float(np.corrcoef(x.ravel(), y.ravel())[0, 1]) for x, y in ((r, g), (g, b), (r, b))
        )
    else:
        corr = 1.0
    skin = (r > 95) & (g > 40) & (b > 20) & (r > g) & (g > b) & ((r - g) > 18) & ((r - b) > 35)
    if float(np.mean(skin)) > 0.16 and spread > 22.0:
        return None, "NON_RADIOGRAPH_SELFIE_OR_PORTRAIT"
    if corr < 0.82 or chroma_std > 26.0 or spread > 68.0:
        return None, "NON_RADIOGRAPH_HIGH_COLOR_SATURATION"
    return gray, "RGB_TINTED_XRAY_PHOTO"


def _decode(source: Any, filename_hint: str | None) -> tuple[np.ndarray | None, str | None]:
    """Return (array, None) or (None, reason_code). Never raises."""
    try:
        ext = None
        if isinstance(source, (str, Path)):
            path = Path(source)
            ext = path.suffix.lower()
        elif filename_hint:
            ext = Path(filename_hint).suffix.lower()
        if ext and ext not in SUPPORTED_EXTENSIONS:
            return None, "UNSUPPORTED_FILE_FORMAT"

        if isinstance(source, (str, Path)):
            if not path.is_file():
                return None, "FILE_NOT_FOUND"
            if path.stat().st_size > MAX_INPUT_BYTES:
                return None, "INPUT_TOO_LARGE"
            if ext == ".npy":
                return np.load(path, allow_pickle=False), None
            with Image.open(path) as im:
                w, h = im.size
                if w <= 0 or h <= 0 or w > 8192 or h > 8192 or (w * h) > MAX_IMAGE_PIXELS:
                    return None, "IMAGE_TOO_LARGE_OR_BOMB"
                im.load()
                return np.array(im), None
        if isinstance(source, (bytes, bytearray)):
            if len(source) == 0:
                return None, "EMPTY_INPUT"
            if len(source) > MAX_INPUT_BYTES:
                return None, "INPUT_TOO_LARGE"
            if ext == ".npy":
                return np.load(io.BytesIO(bytes(source)), allow_pickle=False), None
            with Image.open(io.BytesIO(bytes(source))) as im:
                w, h = im.size
                if w <= 0 or h <= 0 or w > 8192 or h > 8192 or (w * h) > MAX_IMAGE_PIXELS:
                    return None, "IMAGE_TOO_LARGE_OR_BOMB"
                im.load()
                return np.array(im), None
                im.load()
                return np.array(im), None
        if isinstance(source, Image.Image):
            return np.array(source), None
        if isinstance(source, np.ndarray):
            return source.copy(), None
        return None, "UNSUPPORTED_INPUT_TYPE"
    except Exception:  # noqa: BLE001 - fail closed on ANY decoder error
        return None, "CORRUPT_OR_UNREADABLE_IMAGE"


def run_input_gate(
    source: Any,
    *,
    model_id: str,
    preprocessing_id: str,
    finding_id: str = "pneumothorax",
    raw_metadata: Mapping[str, Any] | None = None,
    filename_hint: str | None = None,
    config: GateConfig | None = None,
) -> tuple[InputContract | None, InputGateContract]:
    """Validate an input. Returns an InputContract only when the gate says ACCEPT."""
    try:
        return _run_input_gate(source, model_id, preprocessing_id, finding_id, raw_metadata, filename_hint,
                               config or GateConfig())
    except Exception:  # noqa: BLE001 - the gate must never raise
        return _reject(["GATE_INTERNAL_ERROR"])


def _run_input_gate(source, model_id, preprocessing_id, finding_id, raw_metadata, filename_hint, cfg):
    arr, err = _decode(source, filename_hint)
    if err is not None:
        return _reject([err])
    assert arr is not None

    if arr.dtype == object or not (np.issubdtype(arr.dtype, np.number) or arr.dtype == np.bool_):
        return _reject(["NON_NUMERIC_PIXELS"])
    if arr.size == 0:
        return _reject(["EMPTY_INPUT"])
    arr = arr.astype(np.float32) if arr.dtype != np.uint16 else arr
    if not np.all(np.isfinite(arr)):
        return _reject(["NON_FINITE_PIXELS"])

    # Metadata: allowlist only (§4.1, §10.4).
    meta = {k: v for k, v in dict(raw_metadata or {}).items() if k in ALLOWED_METADATA_KEYS}
    view = str(meta.get("view", "UNKNOWN")).upper().strip()
    if view not in SUPPORTED_VIEWS:
        return _reject(["UNSUPPORTED_VIEW"])

    storage_code = "GRAYSCALE"
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[2] == 3:
        rgb = arr.astype(np.float32)
        if arr.dtype == np.uint16:
            rgb = rgb / 257.0
        elif rgb.max() <= 1.05 and rgb.min() >= 0.0:
            rgb = rgb * 255.0
        gray, storage_code = _rgb_to_gray_or_reason(rgb)
        if gray is None:
            return _reject([storage_code])
        arr = gray
    elif arr.ndim != 2:
        return _reject(["UNSUPPORTED_ARRAY_SHAPE"])

    gray = arr.astype(np.float32)
    if arr.dtype == np.uint16:
        gray = gray / 257.0
    lo, hi = float(gray.min()), float(gray.max())
    if hi <= 1.05 and lo >= 0.0 and hi - lo > 0.01:
        gray = gray * 255.0
    elif hi > 255.0 or lo < 0.0:
        gray = (gray - lo) / max(hi - lo, 1e-6) * 255.0
    gray = np.clip(gray, 0.0, 255.0).astype(np.float32)
    h, w = gray.shape

    dim_codes = []
    if h < cfg.min_height or w < cfg.min_width:
        dim_codes.append("IMAGE_DIMENSIONS_TOO_SMALL")
    if h > cfg.max_height or w > cfg.max_width:
        dim_codes.append("IMAGE_DIMENSIONS_TOO_LARGE")
    if dim_codes:
        return _reject(dim_codes)

    # Bimodal black-on-white page scans are not radiographs.
    if float(np.mean(gray > 236.0)) > 0.50 and float(np.mean((gray >= 40.0) & (gray <= 215.0))) < 0.35:
        return _reject(["NON_RADIOGRAPH_DOCUMENT_SCAN"])

    mean, std = float(gray.mean()), float(gray.std())
    p1, p99 = np.percentile(gray, [1, 99])
    under = float(np.mean(gray <= 4.0))
    over = float(np.mean(gray >= 251.0))
    clipped = float(np.mean((gray <= 2.0) | (gray >= 253.0)))
    lap = float(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var())
    aspect = w / float(h)
    burned = _burned_in_text_suspected(gray)

    codes = []
    if aspect < cfg.min_aspect_ratio or aspect > cfg.max_aspect_ratio:
        codes.append("EXTREME_ASPECT_RATIO")
    if mean < cfg.min_mean_intensity or under > cfg.max_underexposed_fraction:
        codes.append("IMAGE_TOO_DARK")
    if mean > cfg.max_mean_intensity or over > cfg.max_overexposed_fraction:
        codes.append("IMAGE_TOO_BRIGHT")
    if std < cfg.min_std_intensity or float(p99 - p1) < cfg.min_dynamic_range:
        codes.append("LOW_CONTRAST_OR_BLANK_IMAGE")
    if clipped > cfg.max_clipped_fraction:
        codes.append("EXCESSIVE_CLIPPING")
    if lap < cfg.min_laplacian_var:
        codes.append("EXCESSIVE_BLUR")
    flags = {
        "burned_in_text_suspected": burned,
        "rgb_converted_to_luminance": storage_code in {"RGB_IDENTICAL_GRAYSCALE", "RGB_TINTED_XRAY_PHOTO"},
        "too_dark": "IMAGE_TOO_DARK" in codes,
        "too_bright": "IMAGE_TOO_BRIGHT" in codes,
        "low_contrast": "LOW_CONTRAST_OR_BLANK_IMAGE" in codes,
        "blurred": "EXCESSIVE_BLUR" in codes,
    }
    if codes:
        return _retake(codes, flags)

    contract = InputContract(
        image_tensor=gray,
        image_metadata_allowlist=meta,
        finding_id=finding_id,
        model_id=model_id,
        preprocessing_id=preprocessing_id,
    )
    accept = ["INPUT_QUALITY_OK"]
    if burned:
        accept.append("WARNING_BURNED_IN_TEXT_SUSPECTED")
    return contract, InputGateContract(InputGateStatus.ACCEPT, tuple(accept), flags)
