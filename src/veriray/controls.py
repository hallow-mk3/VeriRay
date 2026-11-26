"""Matched control mask generator (Blueprint 26 Nov §7 step 6; v2 §8.1 step 5, §8.2).

Generates up to K reproducible control masks matching the target mask's area and shape
(and anatomical zone when a lung/region mask is supplied), enforcing a strict overlap
limit (< 0.1) against the target mask and diversity among controls.

Controls can still contain relevant image information; anatomical/geometric matching
does not make them guaranteed negatives, and empirical ranks are never p-values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ControlConfig:
    k: int = 19
    min_controls: int = 8
    max_target_overlap: float = 0.10
    area_tolerance: float = 0.25
    max_pairwise_control_iou: float = 0.80
    seed: int = 20261126


@dataclass(frozen=True)
class ControlGenerationResult:
    masks: tuple[np.ndarray, ...]
    overlaps_with_target: tuple[float, ...]
    area_ratios: tuple[float, ...]
    zone_matched_count: int
    valid: bool
    reason_codes: tuple[str, ...]


def _overlap_metrics(cand: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    inter = int(np.logical_and(cand, target).sum())
    if inter == 0:
        return 0.0, 0.0
    union = int(np.logical_or(cand, target).sum())
    min_area = max(1, min(int(cand.sum()), int(target.sum())))
    return inter / float(max(union, 1)), inter / float(min_area)


def _crop_shape(target: np.ndarray) -> np.ndarray:
    ys, xs = np.where(target)
    return target[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _place_shape(shape_patch: np.ndarray, cy: int, cx: int, h: int, w: int) -> np.ndarray:
    ph, pw = shape_patch.shape
    y0 = cy - ph // 2
    x0 = cx - pw // 2
    y1 = y0 + ph
    x1 = x0 + pw
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        return np.zeros((h, w), dtype=bool)
    out = np.zeros((h, w), dtype=bool)
    out[y0:y1, x0:x1] = shape_patch
    return out


def _ellipse_mask(cy: int, cx: int, ry: float, rx: float, h: int, w: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) / max(ry, 1.0)) ** 2 + ((xx - cx) / max(rx, 1.0)) ** 2 <= 1.0


def generate_matched_controls(
    target_mask: np.ndarray,
    cfg: ControlConfig = ControlConfig(),
    *,
    anatomical_mask: np.ndarray | None = None,
) -> ControlGenerationResult:
    """Return up to ``cfg.k`` reproducible matched control masks."""
    target = np.asarray(target_mask, dtype=bool)
    if target.ndim != 2 or not target.any():
        return ControlGenerationResult((), (), (), 0, False, ("CONTROLS_EMPTY_TARGET",))

    h, w = target.shape
    target_area = int(target.sum())
    if target_area >= h * w:
        return ControlGenerationResult((), (), (), 0, False, ("CONTROLS_FULL_TARGET",))

    if anatomical_mask is not None:
        anat = np.asarray(anatomical_mask, dtype=bool)
        if anat.shape != target.shape:
            return ControlGenerationResult((), (), (), 0, False, ("CONTROLS_BAD_ANATOMICAL_MASK",))
        target_in_anat = float(target[anat].sum()) / float(target_area)
    else:
        anat = None
        target_in_anat = 0.0

    patch = _crop_shape(target)
    variants = [patch, np.fliplr(patch), np.flipud(patch)]
    if patch.shape[0] <= w and patch.shape[1] <= h:
        variants.append(patch.T)

    # Also prepare a matched-area/aspect ellipse fallback in case the bounding box clips near borders.
    ph, pw = patch.shape
    scale = np.sqrt(target_area / max(1.0, np.pi * (ph / 2.0) * (pw / 2.0)))
    ell_ry = max(2.0, (ph / 2.0) * scale)
    ell_rx = max(2.0, (pw / 2.0) * scale)

    ys, xs = np.where(target)
    t_cy, t_cx = float(ys.mean()), float(xs.mean())
    target_band = int(min(2, (t_cy / h) * 3))  # upper=0, mid=1, lower=2

    step_y = max(4, h // 28)
    step_x = max(4, w // 28)
    grid_pts = [(cy, cx) for cy in range(step_y // 2, h, step_y) for cx in range(step_x // 2, w, step_x)]
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(grid_pts))

    candidates: list[tuple[float, np.ndarray, float, float, bool]] = []
    for idx in order:
        cy, cx = grid_pts[int(idx)]
        v_idx = int(rng.integers(0, len(variants)))
        cand = _place_shape(variants[v_idx], cy, cx, h, w)
        if not cand.any():
            cand = _ellipse_mask(cy, cx, ell_ry, ell_rx, h, w)
        c_area = int(cand.sum())
        if c_area == 0 or c_area == h * w:
            continue
        area_ratio = c_area / float(target_area)
        if abs(area_ratio - 1.0) > cfg.area_tolerance:
            continue
        iou, overlap_frac = _overlap_metrics(cand, target)
        if iou >= cfg.max_target_overlap or overlap_frac >= cfg.max_target_overlap:
            continue

        # Score candidate for anatomical/zone matching (lower penalty = better match).
        cand_band = int(min(2, (cy / h) * 3))
        zone_penalty = 0.0 if cand_band == target_band else 0.25
        if anat is not None and anat.any():
            cand_in_anat = float(cand[anat].sum()) / float(c_area)
            anat_penalty = abs(cand_in_anat - target_in_anat)
            zone_matched = anat_penalty <= 0.30 and cand_band == target_band
        else:
            anat_penalty = 0.0
            zone_matched = cand_band == target_band

        # Slight deterministic jitter breaks ties reproducibly.
        score = anat_penalty + zone_penalty + 0.05 * float(rng.random())
        candidates.append((score, cand, max(iou, overlap_frac), area_ratio, zone_matched))

    candidates.sort(key=lambda item: item[0])

    selected_masks: list[np.ndarray] = []
    selected_overlaps: list[float] = []
    selected_ratios: list[float] = []
    zone_matched_count = 0

    for _, cand, ov, ratio, z_ok in candidates:
        if len(selected_masks) >= cfg.k:
            break
        if any(_overlap_metrics(cand, prev)[0] > cfg.max_pairwise_control_iou for prev in selected_masks):
            continue
        selected_masks.append(cand)
        selected_overlaps.append(float(ov))
        selected_ratios.append(float(ratio))
        if z_ok:
            zone_matched_count += 1

    if len(selected_masks) < cfg.min_controls:
        return ControlGenerationResult(
            masks=tuple(selected_masks),
            overlaps_with_target=tuple(selected_overlaps),
            area_ratios=tuple(selected_ratios),
            zone_matched_count=zone_matched_count,
            valid=False,
            reason_codes=("INSUFFICIENT_MATCHED_CONTROLS",),
        )

    return ControlGenerationResult(
        masks=tuple(selected_masks),
        overlaps_with_target=tuple(selected_overlaps),
        area_ratios=tuple(selected_ratios),
        zone_matched_count=zone_matched_count,
        valid=True,
        reason_codes=("CONTROLS_OK",),
    )
