"""Locator (Blueprint v2.0 §6): Grad-CAM++ candidate region, NOT evidence of pathology.

Parameters are written down in docs/PROTOCOL.md and are PROVISIONAL (set a priori; no
ground truth is used to produce the mask). Weak masks are flagged so the pipeline can
return INDETERMINATE. Stability / anatomical-plausibility checks are NOT done yet and
are reported as not-assessed (never as passed).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from veriray.contracts import LocatorContract
from veriray.reader import Reader

LOCATOR_ID = "gradcampp-norm5-relu-v1"


class LocatorError(Exception):
    reason_code = "LOCATOR_FAILED"


@dataclass(frozen=True)
class LocatorConfig:
    threshold: float = 0.5  # on min-max normalized heatmap
    opening_kernel: int = 5  # elliptical kernel, pixels at 224x224
    closing_kernel: int = 5
    min_component_frac: float = 0.005  # drop connected components smaller than this fraction of the image
    min_area_frac: float = 0.005  # mask area below this => empty
    max_area_frac: float = 0.50  # above => implausibly_large
    near_full_frac: float = 0.90  # at/above => near_full_image (also implausibly_large)
    diffuse_low_level: float = 0.25  # diffuse if > diffuse_frac of pixels have heatmap >= this level ...
    diffuse_frac: float = 0.75
    max_components: int = 3  # ... or if the mask is fragmented into more components than this
    epsilon: float = 1e-8


def gradcam_plus_plus(reader: Reader, x: torch.Tensor, out_size: int = 224) -> np.ndarray:
    """Grad-CAM++ heatmap in [0, 1] (out_size x out_size) for the Pneumothorax *logit*.

    Layer: Reader.feature_map = ReLU(features.norm5(.)) (last conv feature layer, 1024x7x7).
    Uses the standard closed-form Grad-CAM++ weights (Chattopadhay et al. 2018):
        alpha = g^2 / (2 g^2 + sum_ab(A) * g^3);  w_k = sum_ab(alpha * ReLU(g));  cam = ReLU(sum_k w_k A_k)
    where g = d logit / d A. Output is bilinearly upsampled and min-max normalized.
    """
    try:
        with torch.no_grad():
            a = reader.feature_map(x)
        a = a.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            logit = reader.head_logits(a)[0, reader.finding_index]
            (g,) = torch.autograd.grad(logit, a)
        g2, g3 = g.pow(2), g.pow(3)
        denom = 2.0 * g2 + a.sum(dim=(2, 3), keepdim=True) * g3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = g2 / denom
        weights = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * a).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(out_size, out_size), mode="bilinear", align_corners=False)[0, 0]
        cam_np = cam.detach().cpu().numpy().astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        raise LocatorError(f"{type(exc).__name__}: {exc}") from exc
    if not np.all(np.isfinite(cam_np)):
        raise LocatorError("non-finite Grad-CAM++ values")
    return normalize_heatmap(cam_np)


def normalize_heatmap(cam: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize to [0, 1]; a flat (constant) map becomes all zeros."""
    cam = np.asarray(cam, dtype=np.float32)
    lo, hi = float(cam.min()), float(cam.max())
    if hi - lo < eps:
        return np.zeros_like(cam, dtype=np.float32)
    return np.clip((cam - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def heatmap_to_mask(heatmap: np.ndarray, cfg: LocatorConfig = LocatorConfig()) -> np.ndarray:
    """Threshold -> opening -> closing -> drop tiny components. Pure function of the heatmap."""
    mask = (heatmap >= cfg.threshold) & (heatmap > cfg.epsilon)
    m = mask.astype(np.uint8)
    if cfg.opening_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.opening_kernel, cfg.opening_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if cfg.closing_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.closing_kernel, cfg.closing_kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    min_px = cfg.min_component_frac * m.size
    out = np.zeros_like(m, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            out |= labels == i
    return out


def localization_flags(
    heatmap: np.ndarray, mask: np.ndarray, cfg: LocatorConfig = LocatorConfig()
) -> Mapping[str, bool]:
    """Quality flags. True on any of empty/diffuse/implausibly_large/... means INDETERMINATE-eligible."""
    area_frac = float(mask.mean())
    n_comp = int(cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1)
    empty = (not bool(mask.any())) or area_frac < cfg.min_area_frac
    near_full = area_frac >= cfg.near_full_frac
    implausibly_large = area_frac > cfg.max_area_frac or near_full
    diffuse = float((heatmap >= cfg.diffuse_low_level).mean()) > cfg.diffuse_frac or n_comp > cfg.max_components
    return {
        "empty": bool(empty),
        "diffuse": bool(diffuse),
        "implausibly_large": bool(implausibly_large),
        "near_full_image": bool(near_full),
        # Not assessed in Step 2 (need perturbation tests / anatomy model). NOT evidence of pass.
        "anatomically_implausible": False,
        "unstable": False,
        "anatomy_assessed": False,
        "stability_assessed": False,
    }


def locate(reader: Reader, x: torch.Tensor, cfg: LocatorConfig = LocatorConfig()) -> LocatorContract:
    heatmap = gradcam_plus_plus(reader, x, out_size=int(x.shape[-1]))
    mask = heatmap_to_mask(heatmap, cfg)
    return LocatorContract(heatmap=heatmap, target_mask=mask, localization_flags=localization_flags(heatmap, mask, cfg))
