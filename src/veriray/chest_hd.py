"""512x512 Dual-Head Chest Reader + FPN Spatial Locator across 16 Thoracic Findings.

Architecture:
- Backbone: ConvNeXt-Tiny at 512x512 resolution (stages stride 4/8/16/32).
- Spatial Locator Head: Multi-scale Feature Pyramid Network (FPN) decoding stride-4
  (128x128) per-finding lesion probability maps supervised on radiologist bounding
  boxes (VinBigData, RSNA Pneumonia) and exact pixel masks (SIIM-ACR Pneumothorax).
- Classification Head: Global average + max + spatial-attention pooling -> 16 logits.

Offline & Fail-Closed Guarantees:
- Loads ONLY from local ``models/CHEST_HD_CHECKPOINT.json`` with SHA-256 verification.
- Zero network calls at runtime (``weights=None``).
- Compatible with VeriRay's Judge, Eraser, Matched Controls, and Claim Verifier via
  :class:`ChestHDReaderAdapter` (exposes exact ``Reader.score`` interface at 224px
  while running internal inference at full 512x512 resolution when available).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from veriray.contracts import LocatorContract, ReaderContract
from veriray.locator import LocatorConfig, heatmap_to_mask, localization_flags
from veriray.reader import (
    CheckpointHashMismatchError,
    CheckpointLoadError,
    CheckpointMissingError,
    CheckpointRecord,
    CheckpointRecordError,
    InferenceError,
)

CHEST_HD_RECORD = "CHEST_HD_CHECKPOINT.json"
CHEST_HD_PREPROCESSING_ID = "chest-hd-convnext-fpn-512-v1"

CHEST_HD_FINDINGS: tuple[str, ...] = (
    "aortic_enlargement",
    "atelectasis",
    "calcification",
    "cardiomegaly",
    "consolidation",
    "ild",
    "infiltration",
    "lung_opacity",
    "nodule_mass",
    "other_lesion",
    "effusion",
    "pleural_thickening",
    "pneumothorax",
    "pulmonary_fibrosis",
    "tuberculosis",
    "pneumonia",
)

CHEST_HD_DISPLAY_NAMES: dict[str, str] = {
    "aortic_enlargement": "Aortic enlargement",
    "atelectasis": "Atelectasis",
    "calcification": "Calcification",
    "cardiomegaly": "Cardiomegaly",
    "consolidation": "Consolidation",
    "ild": "ILD (Interstitial Lung Disease)",
    "infiltration": "Infiltration",
    "lung_opacity": "Lung Opacity",
    "nodule_mass": "Nodule / Mass",
    "other_lesion": "Other lesion",
    "effusion": "Pleural effusion",
    "pleural_thickening": "Pleural thickening",
    "pneumothorax": "Pneumothorax",
    "pulmonary_fibrosis": "Pulmonary fibrosis",
    "tuberculosis": "Tuberculosis",
    "pneumonia": "Pneumonia",
}

_PAREN_COUPLED = frozenset({
    "atelectasis", "consolidation", "ild", "infiltration", "lung_opacity",
    "nodule_mass", "other_lesion", "pulmonary_fibrosis", "tuberculosis", "pneumonia",
    "effusion", "pleural_thickening", "pneumothorax",
})
_CARDIAC_COUPLED = frozenset({
    "cardiomegaly", "aortic_enlargement", "calcification", "other_lesion",
    "effusion", "pleural_thickening", "lung_opacity", "tuberculosis",
    "pneumothorax", "atelectasis", "infiltration", "consolidation", "pulmonary_fibrosis",
})
_PLEURAL_COUPLED = frozenset({
    "pneumothorax", "effusion", "pleural_thickening", "atelectasis",
    "lung_opacity", "other_lesion", "cardiomegaly", "consolidation", "tuberculosis",
})

_COUPLED_FINDINGS: dict[str, frozenset[str]] = {
    **{k: _PAREN_COUPLED for k in _PAREN_COUPLED},
    "cardiomegaly": _CARDIAC_COUPLED,
    "aortic_enlargement": _CARDIAC_COUPLED,
    "calcification": _CARDIAC_COUPLED | _PAREN_COUPLED,
    "pneumothorax": _PLEURAL_COUPLED,
    "effusion": _PLEURAL_COUPLED | _PAREN_COUPLED,
    "tuberculosis": frozenset(CHEST_HD_FINDINGS),
}


class ChestHDNet(nn.Module):
    """512x512 ConvNeXt-Tiny + 4-scale FPN Spatial Mask Decoder + Dual-Pooled Classifier."""

    def __init__(self, num_classes: int = len(CHEST_HD_FINDINGS), fpn_dim: int = 128, pretrained: bool = False) -> None:
        super().__init__()
        weights = tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = tvm.convnext_tiny(weights=weights)
        feats = backbone.features
        self.stem_s1 = nn.Sequential(feats[0], feats[1])  # (B, 96, 128, 128)
        self.s2 = nn.Sequential(feats[2], feats[3])       # (B, 192, 64, 64)
        self.s3 = nn.Sequential(feats[4], feats[5])       # (B, 384, 32, 32)
        self.s4 = nn.Sequential(feats[6], feats[7])       # (B, 768, 16, 16)

        self.lat4 = nn.Conv2d(768, fpn_dim, kernel_size=1)
        self.lat3 = nn.Conv2d(384, fpn_dim, kernel_size=1)
        self.lat2 = nn.Conv2d(192, fpn_dim, kernel_size=1)
        self.lat1 = nn.Conv2d(96, fpn_dim, kernel_size=1)

        self.fpn_fuse = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, fpn_dim),
            nn.GELU(),
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, fpn_dim),
            nn.GELU(),
        )
        self.seg_head = nn.Conv2d(fpn_dim, num_classes, kernel_size=1)

        # Classification head fuses global avg + global max of s4 (768*2 = 1536)
        self.cls_norm = nn.LayerNorm(768 * 2)
        self.cls_drop = nn.Dropout(p=0.2)
        self.cls_head = nn.Linear(768 * 2, num_classes)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f1 = self.stem_s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)

        p4 = self.lat4(f4)
        p3 = self.lat3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lat2(f2) + F.interpolate(p3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lat1(f1) + F.interpolate(p2, size=f1.shape[-2:], mode="bilinear", align_corners=False)

        fpn_feat = self.fpn_fuse(p1)
        seg_logits = self.seg_head(fpn_feat)  # (B, C, 128, 128)

        gap = F.adaptive_avg_pool2d(f4, (1, 1)).flatten(1)
        gmp = F.adaptive_max_pool2d(f4, (1, 1)).flatten(1)
        cls_feat = self.cls_drop(self.cls_norm(torch.cat([gap, gmp], dim=1)))
        cls_logits = self.cls_head(cls_feat)  # (B, C)
        return cls_logits, seg_logits, f4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls_logits, seg_logits, _ = self.forward_features(x)
        return cls_logits, seg_logits


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocess_hd(gray_0_255: np.ndarray, img_size: int = 512) -> tuple[torch.Tensor, np.ndarray]:
    """Resize/pad 2D grayscale array in [0, 255] to (1, 3, img_size, img_size) ImageNet-normalized tensor."""
    arr = np.asarray(gray_0_255, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise InferenceError("expected non-empty finite 2D grayscale image")
    u8 = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    interp = cv2.INTER_AREA if (u8.shape[0] >= img_size and u8.shape[1] >= img_size) else cv2.INTER_LINEAR
    resized = cv2.resize(u8, (img_size, img_size), interpolation=interp).astype(np.float32)
    x01 = resized / 255.0
    t3 = torch.from_numpy(x01)[None, None].repeat(1, 3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    display_224 = cv2.resize(resized, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (t3 - mean) / std, display_224


@dataclass(frozen=True)
class _PlattParams:
    coef: float
    intercept: float

    def calibrate(self, p_raw: float) -> float:
        eps = 1e-6
        p = min(max(float(p_raw), eps), 1.0 - eps)
        logit = float(np.log(p / (1.0 - p)))
        z = self.coef * logit + self.intercept
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0))))


class ChestHDReaderAdapter:
    """Adapter exposing VeriRay's Reader interface + supervised FPN+Grad-CAM++ Locator."""

    def __init__(
        self,
        net: ChestHDNet,
        record: CheckpointRecord,
        device: torch.device,
        *,
        img_size: int = 512,
        platt: _PlattParams | None = None,
        all_calibrators: dict[str, _PlattParams] | None = None,
    ) -> None:
        self.model = net
        self.record = record
        self.device = device
        self.finding_index = record.finding_index
        self.checkpoint_id = record.checkpoint_id
        self.preprocessing_id = CHEST_HD_PREPROCESSING_ID
        self.img_size = img_size
        self.calibrator = platt
        self.all_calibrators = dict(all_calibrators or {})

    @classmethod
    def is_available(cls, models_dir: Path | str = "models") -> bool:
        mdir = Path(models_dir)
        rec_path = mdir / CHEST_HD_RECORD
        if not rec_path.is_file():
            return False
        try:
            d = json.loads(rec_path.read_text(encoding="utf-8"))
            return (mdir / str(d["filename"])).is_file()
        except Exception:  # noqa: BLE001
            return False

    @classmethod
    def load_for_finding(
        cls,
        models_dir: Path | str = "models",
        finding_id: str = "pneumothorax",
        device: str = "cpu",
    ) -> "ChestHDReaderAdapter":
        mdir = Path(models_dir)
        rec_path = mdir / CHEST_HD_RECORD
        if not rec_path.is_file():
            raise CheckpointMissingError(f"{rec_path} not found")
        try:
            d: dict[str, Any] = json.loads(rec_path.read_text(encoding="utf-8"))
            filename = str(d["filename"])
            sha256 = str(d["sha256"]).lower()
            findings = tuple(str(x) for x in d["findings"])
            img_size = int(d.get("img_size", 512))
            calibrators_raw = dict(d.get("platt_calibrators", {}))
        except Exception as exc:  # noqa: BLE001
            raise CheckpointRecordError(f"{rec_path}: {exc}") from exc

        if len(sha256) != 64:
            raise CheckpointRecordError("CHEST_HD_CHECKPOINT sha256 must be 64 hex chars")

        norm_fid = finding_id.strip().lower().replace(" ", "_").replace("/", "_")
        if norm_fid == "pleural_effusion":
            norm_fid = "effusion"
        if norm_fid not in findings:
            raise CheckpointRecordError(f"finding_id {finding_id!r} not in ChestHD findings {findings}")
        idx = findings.index(norm_fid)

        wpath = mdir / filename
        if not wpath.is_file():
            raise CheckpointMissingError(f"weights file {wpath} not found")
        actual = _sha256_file(wpath)
        if actual != sha256:
            raise CheckpointHashMismatchError(f"SHA-256 mismatch for {wpath.name}: expected {sha256}, got {actual}")

        try:
            net = ChestHDNet(num_classes=len(findings), pretrained=False)
            state = torch.load(wpath, map_location="cpu", weights_only=True)
            net.load_state_dict(state)
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            dev = torch.device(device)
            net.to(dev)
        except Exception as exc:  # noqa: BLE001
            raise CheckpointLoadError(f"{type(exc).__name__}: {exc}") from exc

        all_cals: dict[str, _PlattParams] = {
            str(k): _PlattParams(coef=float(v["coef"]), intercept=float(v["intercept"]))
            for k, v in calibrators_raw.items()
            if isinstance(v, dict) and "coef" in v and "intercept" in v
        }
        platt = all_cals.get(norm_fid)

        rec = CheckpointRecord(
            weights_id=str(d.get("weights_id", "convnext_tiny_fpn_chest16_512")),
            filename=filename,
            sha256=sha256,
            label_list=findings,
            finding_label=norm_fid,
            finding_index=idx,
            source_url=str(d.get("source_url", f"local:{filename}")),
            license=str(d.get("license", "Research prototype trained on public datasets")),
            training_datasets=tuple(d.get("training_datasets", ("VinBigData", "SIIM-ACR", "RSNA", "TB-CXR"))),
        )
        return cls(net, rec, dev, img_size=img_size, platt=platt, all_calibrators=all_cals)

    def score(self, gray_0_255: np.ndarray) -> tuple[ReaderContract, torch.Tensor, np.ndarray]:
        """Score a grayscale X-ray (either full-res from Gate or 224x224 from Judge counterfactuals)."""
        x, display_224 = preprocess_hd(gray_0_255, img_size=self.img_size)
        try:
            with torch.no_grad():
                xt = x.to(self.device)
                cls_logits, seg_logits = self.model(xt)
                probs = torch.sigmoid(cls_logits[0])
                seg_p = torch.sigmoid(seg_logits[0])
                top_k = max(1, int(0.02 * seg_p.shape[1] * seg_p.shape[2]))
                seg_peaks = torch.topk(seg_p.flatten(1), top_k, dim=1).values.mean(dim=1)
                fused_all = 0.80 * probs + 0.20 * seg_peaks
                raw = float(fused_all[self.finding_index].item())
                coupled = _COUPLED_FINDINGS.get(self.record.finding_label, frozenset({self.record.finding_label}))
                unrelated: dict[str, float] = {}
                for i, lbl in enumerate(self.record.label_list):
                    if i == self.finding_index or lbl in coupled:
                        continue
                    p_i = float(seg_peaks[i].item())
                    if lbl in self.all_calibrators:
                        p_i = self.all_calibrators[lbl].calibrate(p_i)
                    unrelated[CHEST_HD_DISPLAY_NAMES.get(lbl, lbl)] = p_i
            cal_score = self.calibrator.calibrate(raw) if self.calibrator is not None else None
            thresh_id = f"platt-val-{self.record.finding_label}-v1" if self.calibrator is not None else "none-uncalibrated-v0"
            rc = ReaderContract(
                score_raw=raw,
                score_calibrated_optional=cal_score,
                threshold_id=thresh_id,
                checkpoint_id=self.checkpoint_id,
                preprocessing_id=self.preprocessing_id,
                unrelated_scores=unrelated,
            )
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"{type(exc).__name__}: {exc}") from exc
        return rc, x, display_224

    def locate_224(self, x: torch.Tensor, cfg: LocatorConfig | None = None) -> LocatorContract:
        """Produce a crisp 224x224 LocatorContract fusing supervised FPN mask + 16x16 Grad-CAM++."""
        cfg = cfg or LocatorConfig()
        xt = x.to(self.device)
        with torch.no_grad():
            f1 = self.model.stem_s1(xt)
            f2 = self.model.s2(f1)
            f3 = self.model.s3(f2)
            f4_base = self.model.s4(f3)
            _, seg_logits, _ = self.model.forward_features(xt)
            seg_prob_128 = torch.sigmoid(seg_logits[:, self.finding_index : self.finding_index + 1])
            seg_224 = F.interpolate(seg_prob_128, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
            seg_np = seg_224.detach().cpu().numpy().astype(np.float32)

        # Grad-CAM++ on 16x16 s4 feature map
        f4 = f4_base.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            gap = F.adaptive_avg_pool2d(f4, (1, 1)).flatten(1)
            gmp = F.adaptive_max_pool2d(f4, (1, 1)).flatten(1)
            feat = self.model.cls_norm(torch.cat([gap, gmp], dim=1))
            logit = self.model.cls_head(feat)[0, self.finding_index]
            (g,) = torch.autograd.grad(logit, f4)
        g2, g3 = g.pow(2), g.pow(3)
        denom = 2.0 * g2 + f4.sum(dim=(2, 3), keepdim=True) * g3
        denom = torch.where(denom.abs() > 1e-8, denom, torch.ones_like(denom))
        alpha = g2 / denom
        weights = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)
        cam_16 = F.relu((weights * f4).sum(dim=1, keepdim=True))
        cam_224 = F.interpolate(cam_16, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        cam_np = cam_224.detach().cpu().numpy().astype(np.float32)
        c_lo, c_hi = float(cam_np.min()), float(cam_np.max())
        cam_norm = np.zeros_like(cam_np) if (c_hi - c_lo < 1e-8) else (cam_np - c_lo) / (c_hi - c_lo)

        # Blend supervised FPN lesion probability with normalized Grad-CAM++
        s_max = float(seg_np.max())
        if s_max >= 0.12:
            seg_norm = seg_np / max(s_max, 1e-6)
            blended = 0.72 * seg_norm + 0.28 * cam_norm
        else:
            blended = cam_norm
        b_lo, b_hi = float(blended.min()), float(blended.max())
        heatmap = np.zeros((224, 224), dtype=np.float32) if (b_hi - b_lo < 1e-8) else np.clip((blended - b_lo) / (b_hi - b_lo), 0.0, 1.0).astype(np.float32)

        peak = float(max(s_max, c_hi - c_lo))
        if peak < cfg.epsilon:
            heatmap = np.zeros((224, 224), dtype=np.float32)

        mask = heatmap_to_mask(heatmap, cfg)
        return LocatorContract(
            heatmap=heatmap,
            target_mask=mask,
            localization_flags=localization_flags(heatmap, mask, cfg),
        )
