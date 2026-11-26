"""Lightweight modality detection: distinguishes Brain MRI from Chest X-ray.

Runs entirely on image statistics (no model inference) so it is fast,
uses no VRAM, and cannot accidentally contaminate any evaluation split.

Returns a :class:`ModalityGuess` dataclass; the app shows a redirect
warning when ``is_brain_mri_suspected`` is True and the user has uploaded
to the Chest X-ray queue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModalityGuess:
    """Signal-analysis guess at image modality."""

    is_brain_mri_suspected: bool
    confidence: float  # 0.0–1.0, heuristic only
    signals: dict[str, float]
    reason: str


def _compute_signals(gray: np.ndarray) -> dict[str, float]:
    """Return a dict of raw statistics for the grayscale float32 image."""
    gray = gray.astype(np.float32)
    h, w = gray.shape
    aspect = w / float(h)

    mean = float(gray.mean())
    std = float(gray.std())

    # Black-background fraction (MRI has lots of near-zero pixels outside the head)
    dark_frac = float(np.mean(gray < 10.0))

    # Bright core fraction (MRI tissue is concentrated; CXR is more uniform)
    bright_frac = float(np.mean(gray > 150.0))

    # Local texture std via 8x8 patch sampling
    ph, pw = max(h // 8, 1), max(w // 8, 1)
    local_stds = []
    for r in range(0, h - ph + 1, ph):
        for c in range(0, w - pw + 1, pw):
            local_stds.append(float(gray[r : r + ph, c : c + pw].std()))
    local_std_mean = float(np.mean(local_stds)) if local_stds else 0.0

    # Circular symmetry: MRI brain is roughly centred, CXR has bilateral lung fields
    cy, cx = h // 2, w // 2
    r_max = min(cy, cx)
    if r_max > 0:
        ys, xs = np.ogrid[:h, :w]
        dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).astype(np.float32)
        centre_mass = float(np.mean(gray[dist < r_max * 0.4]))
        rim_mass = float(np.mean(gray[dist >= r_max * 0.75]))
        centre_rim_ratio = centre_mass / max(rim_mass, 1e-6)
    else:
        centre_rim_ratio = 1.0

    # Histogram bimodality: MRI has a sharp dark background + brighter tissue peak.
    hist, _ = np.histogram(gray, bins=64, range=(0.0, 255.0))
    hist_norm = hist / float(max(hist.sum(), 1))
    # Fraction of mass in the lowest 4 bins (deep dark background)
    dark_hist_frac = float(hist_norm[:4].sum())

    return {
        "aspect": aspect,
        "mean": mean,
        "std": std,
        "dark_frac": dark_frac,
        "bright_frac": bright_frac,
        "local_std_mean": local_std_mean,
        "centre_rim_ratio": centre_rim_ratio,
        "dark_hist_frac": dark_hist_frac,
    }


def guess_modality(image: np.ndarray) -> ModalityGuess:
    """Return a modality guess for *image* (H×W float32 or uint8 grayscale).

    Heuristic — not a classifier.  Used only to surface a UI redirect hint.
    Never used to gate or block a scan; the caller decides what to show.
    """
    gray = np.asarray(image, dtype=np.float32)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    # Normalise to 0–255 range
    lo, hi = float(gray.min()), float(gray.max())
    if hi - lo > 0.01:
        if hi <= 1.05 and lo >= 0.0:
            gray = gray * 255.0
        elif hi > 255.0 or lo < 0.0:
            gray = (gray - lo) / (hi - lo) * 255.0
    gray = np.clip(gray, 0.0, 255.0)

    s = _compute_signals(gray)
    votes: list[tuple[bool, float, str]] = []

    # --- Signal 1: large dark background fraction
    # Brain MRI on black background typically has >25 % near-zero pixels.
    # Chest X-ray has a relatively bright background.
    if s["dark_frac"] > 0.25:
        votes.append((True, min((s["dark_frac"] - 0.25) / 0.30, 1.0), "dark_background_fraction"))
    else:
        votes.append((False, min((0.25 - s["dark_frac"]) / 0.25, 1.0), "dark_background_fraction"))

    # --- Signal 2: dark histogram mass (deep background peak)
    if s["dark_hist_frac"] > 0.12:
        votes.append((True, min((s["dark_hist_frac"] - 0.12) / 0.20, 1.0), "dark_histogram_mass"))
    else:
        votes.append((False, min((0.12 - s["dark_hist_frac"]) / 0.12, 1.0), "dark_histogram_mass"))

    # --- Signal 3: centre–rim ratio (MRI brain centred, brighter core)
    if s["centre_rim_ratio"] > 1.6:
        votes.append((True, min((s["centre_rim_ratio"] - 1.6) / 2.0, 1.0), "centre_rim_ratio"))
    else:
        votes.append((False, min((1.6 - s["centre_rim_ratio"]) / 1.6, 1.0), "centre_rim_ratio"))

    # --- Signal 4: near-square aspect (brain MRI is roughly square)
    if 0.75 <= s["aspect"] <= 1.35:
        votes.append((True, 0.5, "near_square_aspect"))
    else:
        votes.append((False, 0.5, "not_square_aspect"))

    # --- Signal 5: local texture std (MRI has fine-grained texture in the brain mass)
    if s["local_std_mean"] > 25.0:
        votes.append((True, min((s["local_std_mean"] - 25.0) / 20.0, 1.0), "local_texture_std"))
    else:
        votes.append((False, min((25.0 - s["local_std_mean"]) / 25.0, 1.0), "local_texture_std"))

    mri_weight = sum(w for is_mri, w, _ in votes if is_mri)
    xray_weight = sum(w for is_mri, w, _ in votes if not is_mri)
    total = mri_weight + xray_weight or 1.0
    mri_score = mri_weight / total
    is_mri = mri_score >= 0.55  # slight bias toward not raising a false alarm

    firing = [name for is_mri_v, _, name in votes if is_mri_v == is_mri]
    reason = (
        f"Brain MRI suspected (score={mri_score:.2f}): signals={firing}"
        if is_mri
        else f"Chest X-ray likely (mri_score={mri_score:.2f})"
    )

    return ModalityGuess(
        is_brain_mri_suspected=is_mri,
        confidence=mri_score if is_mri else 1.0 - mri_score,
        signals=s,
        reason=reason,
    )
