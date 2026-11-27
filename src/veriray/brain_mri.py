"""Offline Brain MRI Reader + FPN Lesion Segmenter + Grad-CAM++ Locator.

Supports:
- 4-class Brain Tumor benchmark (glioma, meningioma, notumor, pituitary)
- 5-class Brain Tumor + Stroke HD model (glioma, meningioma, notumor, pituitary, ischemic_stroke)
  with Dual-Head ConvNeXt-Tiny + Multi-Scale FPN pixel-level lesion segmentation trained on
  Brain MRI 4-class + BraTS 2020 (3D glioma voxel masks) + ISLES 2022 (3D stroke voxel masks).
- Both 2D slice uploads (PNG/JPG/BMP) and 3D volumetric NIfTI (.nii / .nii.gz) slice extraction.

Loads ONLY from local ``models/`` after SHA-256 verification. Zero network calls.
Scores and spatial maps describe model behaviour, never a clinical diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

BRAIN_CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
BRAIN_HD_CLASSES = ("glioma", "meningioma", "notumor", "pituitary", "ischemic_stroke")
BRAIN_RECORD = "BRAIN_MRI_CHECKPOINT.json"


class BrainMRIError(Exception):
    def __init__(self, reason_code: str, message: str = "") -> None:
        super().__init__(f"{reason_code}: {message}" if message else reason_code)
        self.reason_code = reason_code


class BrainHDNet(nn.Module):
    """256x256 ConvNeXt-Tiny + 4-scale FPN Lesion Segmentation Head + Dual-Pooled Classifier."""

    def __init__(self, num_classes: int = 5, fpn_dim: int = 128, pretrained: bool = False) -> None:
        super().__init__()
        weights = tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = tvm.convnext_tiny(weights=weights)
        feats = backbone.features
        self.stem_s1 = nn.Sequential(feats[0], feats[1])  # (B, 96, 64, 64)
        self.s2 = nn.Sequential(feats[2], feats[3])       # (B, 192, 32, 32)
        self.s3 = nn.Sequential(feats[4], feats[5])       # (B, 384, 16, 16)
        self.s4 = nn.Sequential(feats[6], feats[7])       # (B, 768, 8, 8)

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
        self.seg_head = nn.Conv2d(fpn_dim, num_classes, kernel_size=1)  # (B, C, 64, 64)

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
        seg_logits = self.seg_head(fpn_feat)

        gap = F.adaptive_avg_pool2d(f4, (1, 1)).flatten(1)
        gmp = F.adaptive_max_pool2d(f4, (1, 1)).flatten(1)
        cls_feat = self.cls_drop(self.cls_norm(torch.cat([gap, gmp], dim=1)))
        cls_logits = self.cls_head(cls_feat)
        return cls_logits, seg_logits, f4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls_logits, seg_logits, _ = self.forward_features(x)
        return cls_logits, seg_logits


def decode_brain_upload(payload: bytes, filename: str = "slice.png") -> np.ndarray:
    """Decode either a 2D image (PNG/JPG/BMP) or a 3D NIfTI (.nii / .nii.gz) central slice to [0, 255] float32."""
    if not payload:
        raise BrainMRIError("BRAIN_BAD_INPUT", "empty upload")
    lower = filename.lower()
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        try:
            import nibabel as nib

            raw = gzip.decompress(payload) if lower.endswith(".nii.gz") else payload
            with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tf:
                tf.write(raw)
                tmp_path = Path(tf.name)
            try:
                vol = np.asarray(nib.load(str(tmp_path)).get_fdata(), dtype=np.float32)
            finally:
                tmp_path.unlink(missing_ok=True)
            if vol.ndim == 4:
                vol = vol[..., 0]
            if vol.ndim != 3 or vol.size == 0:
                raise BrainMRIError("BRAIN_BAD_INPUT", f"unexpected NIfTI shape {vol.shape}")
            # Choose axial slice in middle 50% with highest foreground variance
            z_lo, z_hi = max(0, int(0.25 * vol.shape[2])), max(1, int(0.75 * vol.shape[2]))
            best_z = z_lo + int(np.argmax([float(np.std(vol[:, :, z])) for z in range(z_lo, z_hi)]))
            sl = np.rot90(vol[:, :, best_z])
            sl = np.nan_to_num(sl, nan=0.0, posinf=0.0, neginf=0.0)
            p1, p99 = float(np.percentile(sl, 1)), float(np.percentile(sl, 99.5))
            if p99 - p1 < 1e-6:
                raise BrainMRIError("BRAIN_BAD_INPUT", "blank NIfTI volume")
            return np.clip((sl - p1) / (p99 - p1) * 255.0, 0.0, 255.0).astype(np.float32)
        except BrainMRIError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise BrainMRIError("BRAIN_BAD_INPUT", f"unreadable NIfTI: {exc}") from exc

    try:
        return np.asarray(Image.open(io.BytesIO(payload)).convert("L"), dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        raise BrainMRIError("CORRUPT_OR_UNREADABLE_IMAGE", str(exc)) from exc


@dataclass
class BrainMRIReader:
    model: nn.Module
    checkpoint_id: str
    classes: tuple[str, ...]
    img_size: int
    device: torch.device

    @classmethod
    def load(cls, models_dir: Path | str = "models", device: str | None = None) -> "BrainMRIReader":
        mdir = Path(models_dir)
        rec_path = mdir / BRAIN_RECORD
        if not rec_path.is_file():
            raise BrainMRIError("BRAIN_CHECKPOINT_MISSING", f"{rec_path} not found")
        try:
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
            wpath = mdir / str(rec["filename"])
            expected = str(rec["sha256"]).lower()
            classes = tuple(rec["classes"])
            img_size = int(rec.get("img_size", 256))
        except Exception as exc:  # noqa: BLE001
            raise BrainMRIError("BRAIN_RECORD_INVALID", str(exc)) from exc
        if len(expected) != 64 or not classes:
            raise BrainMRIError("BRAIN_RECORD_INVALID", "invalid sha256 or empty classes")
        if not wpath.is_file():
            raise BrainMRIError("BRAIN_CHECKPOINT_MISSING", f"{wpath} not found")
        actual = hashlib.sha256(wpath.read_bytes()).hexdigest()
        if actual != expected:
            raise BrainMRIError("BRAIN_HASH_MISMATCH", f"expected {expected}, got {actual}")

        if "convnext_tiny_fpn" in wpath.name:
            model: nn.Module = BrainHDNet(num_classes=len(classes), pretrained=False)
        elif "efficientnet_b0" in wpath.name:
            model = tvm.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
        elif "convnext_tiny" in wpath.name:
            model = tvm.convnext_tiny(weights=None)
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(classes))
        elif "densenet121" in wpath.name:
            model = tvm.densenet121(weights=None)
            model.classifier = nn.Linear(model.classifier.in_features, len(classes))
        else:
            raise BrainMRIError("BRAIN_UNKNOWN_ARCH", wpath.name)

        try:
            model.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            model.to(dev)
        except Exception as exc:  # noqa: BLE001
            raise BrainMRIError("BRAIN_LOAD_FAILED", f"{type(exc).__name__}: {exc}") from exc
        return cls(
            model=model,
            checkpoint_id=f"{wpath.stem}@sha256:{expected[:12]}",
            classes=classes,
            img_size=img_size,
            device=dev,
        )

    def _preprocess(self, gray_0_255: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        arr = np.asarray(gray_0_255, dtype=np.float32)
        if arr.ndim != 2 or arr.size == 0 or not np.all(np.isfinite(arr)):
            raise BrainMRIError("BRAIN_BAD_INPUT", "expected non-empty finite 2D grayscale array")
        u8 = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
        resized = cv2.resize(u8, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA).astype(np.float32)
        t3 = torch.from_numpy(resized / 255.0)[None, None].repeat(1, 3, 1, 1).to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (t3 - mean) / std, resized

    def predict_and_cam(self, gray_0_255: np.ndarray) -> dict:
        """Return predicted class, all class model scores, display image, and FPN/Grad-CAM++ spatial map."""
        x, display = self._preprocess(gray_0_255)
        if isinstance(self.model, BrainHDNet):
            with torch.no_grad():
                logits1, seg1, f4_base = self.model.forward_features(x)
                logits2, seg2, _ = self.model.forward_features(torch.flip(x, dims=[3]))
                logits = 0.5 * (logits1 + logits2)
                seg_logits = 0.5 * (seg1 + torch.flip(seg2, dims=[3]))
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_cls = self.classes[pred_idx]

            seg_prob = torch.sigmoid(seg_logits[:, pred_idx : pred_idx + 1])
            seg_up = F.interpolate(seg_prob, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)[0, 0]
            seg_np = seg_up.detach().cpu().numpy().astype(np.float32)

            f4 = f4_base.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                gap = F.adaptive_avg_pool2d(f4, (1, 1)).flatten(1)
                gmp = F.adaptive_max_pool2d(f4, (1, 1)).flatten(1)
                cls_logit = self.model.cls_head(self.model.cls_norm(torch.cat([gap, gmp], dim=1)))[0, pred_idx]
                (g,) = torch.autograd.grad(cls_logit, f4)
            g2, g3 = g.pow(2), g.pow(3)
            denom = 2.0 * g2 + f4.sum(dim=(2, 3), keepdim=True) * g3
            denom = torch.where(denom.abs() > 1e-8, denom, torch.ones_like(denom))
            alpha = g2 / denom
            weights = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * f4).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)[0, 0]
            cam_np = cam.detach().cpu().numpy().astype(np.float32)
            c_lo, c_hi = float(cam_np.min()), float(cam_np.max())
            cam_norm = np.zeros_like(cam_np) if (c_hi - c_lo < 1e-8) else (cam_np - c_lo) / (c_hi - c_lo)

            s_max = float(seg_np.max())
            if pred_cls != "notumor" and s_max >= 0.15:
                blended = 0.75 * (seg_np / max(s_max, 1e-6)) + 0.25 * cam_norm
            else:
                blended = cam_norm
            b_lo, b_hi = float(blended.min()), float(blended.max())
            heatmap = np.zeros_like(blended) if (b_hi - b_lo < 1e-8) else np.clip((blended - b_lo) / (b_hi - b_lo), 0.0, 1.0)
            mask = heatmap >= 0.45
            return {
                "predicted_class": pred_cls,
                "scores": {c: float(probs[i]) for i, c in enumerate(self.classes)},
                "display_image": display,
                "heatmap": heatmap.astype(np.float32),
                "target_mask": mask,
                "supervised_lesion_peak": round(s_max, 4),
                "checkpoint_id": self.checkpoint_id,
            }

        with torch.no_grad():
            logits = 0.5 * (self.model(x) + self.model(torch.flip(x, dims=[3])))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))

        # Grad-CAM++ on last conv feature block
        feats = self.model.features(x).detach().clone().requires_grad_(True)
        with torch.enable_grad():
            pooled = F.adaptive_avg_pool2d(feats, (1, 1))
            if isinstance(self.model.classifier, nn.Sequential) and isinstance(self.model.classifier[0], nn.LayerNorm):
                cls_logit = self.model.classifier(pooled)[0, pred_idx]
            else:
                cls_logit = self.model.classifier(pooled.flatten(1))[0, pred_idx]
            (g,) = torch.autograd.grad(cls_logit, feats)
        g2, g3 = g.pow(2), g.pow(3)
        denom = 2.0 * g2 + feats.sum(dim=(2, 3), keepdim=True) * g3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = g2 / denom
        weights = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feats).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)[0, 0]
        cam_np = cam.detach().cpu().numpy().astype(np.float32)
        lo, hi = float(cam_np.min()), float(cam_np.max())
        heatmap = np.zeros_like(cam_np) if (hi - lo < 1e-8) else np.clip((cam_np - lo) / (hi - lo), 0.0, 1.0)
        mask = heatmap >= 0.5
        return {
            "predicted_class": self.classes[pred_idx],
            "scores": {c: float(probs[i]) for i, c in enumerate(self.classes)},
            "display_image": display,
            "heatmap": heatmap,
            "target_mask": mask,
            "checkpoint_id": self.checkpoint_id,
        }
