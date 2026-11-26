"""Capture Cleaner (Blueprint v2.0 §4, optional MVP): limited, logged geometric/level correction.

Original and transformed images are retained separately; nothing is silently replaced.
NOT imported by any other module (deferred; see docs/PROTOCOL.md).
"""

from __future__ import annotations

import cv2
import numpy as np

from veriray.contracts import CaptureCleanerContract


def apply_capture_cleaner(
    image: np.ndarray,
    *,
    enabled: bool = False,
    clahe_clip_limit: float = 1.5,
) -> CaptureCleanerContract:
    """Trim dark bezel margins (max 8% per side) and apply CLAHE; log every parameter."""
    original = np.asarray(image, dtype=np.float32).copy()
    if not enabled:
        return CaptureCleanerContract(original, None, {"applied": False, "reason": "disabled_by_config"})

    u8 = np.clip(original, 0, 255).astype(np.uint8)
    h, w = u8.shape
    rows, cols = u8.mean(axis=1), u8.mean(axis=0)
    max_y, max_x = int(0.08 * h), int(0.08 * w)
    top = 0
    while top < max_y and rows[top] < 12.0:
        top += 1
    bottom = h
    while (h - bottom) < max_y and rows[bottom - 1] < 12.0:
        bottom -= 1
    left = 0
    while left < max_x and cols[left] < 12.0:
        left += 1
    right = w
    while (w - right) < max_x and cols[right - 1] < 12.0:
        right -= 1

    cropped = u8[top:bottom, left:right]
    leveled = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8)).apply(cropped)
    cleaned = cv2.resize(leveled, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return CaptureCleanerContract(
        original,
        cleaned,
        {
            "applied": True,
            "crop_margins_top_bottom_left_right": [int(top), int(h - bottom), int(left), int(w - right)],
            "clahe_clip_limit": float(clahe_clip_limit),
            "note": "Controlled capture preprocessing; original retained separately.",
        },
    )
