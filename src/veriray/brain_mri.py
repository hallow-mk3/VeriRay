"""Offline Brain Tumor MRI Reader + Grad-CAM++ Locator (2nd Modality: glioma/meningioma/notumor/pituitary).

Loads ONLY from local ``models/`` after SHA-256 verification. Zero network calls.
Uses the exact same VeriRay principle: scores and heatmaps describe model behaviour,
never a clinical diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

BRAIN_CLASSES = ("glioma", "meningioma", "notumor", "pituitary")
BRAIN_RECORD = "BRAIN_MRI_CHECKPOINT.json"


class BrainMRIError(Exception):
    def __init__(self, reason_code: str, message: str = "") -> None:
        super().__init__(f"{reason_code}: {message}" if message else reason_code)
        self.reason_code = reason_code


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
        except Exception as exc:  # noqa: BLE001
            raise BrainMRIError("BRAIN_RECORD_INVALID", str(exc)) from exc
        if len(expected) != 64 or not classes:
            raise BrainMRIError("BRAIN_RECORD_INVALID", "invalid sha256 or empty classes")
        if not wpath.is_file():
            raise BrainMRIError("BRAIN_CHECKPOINT_MISSING", f"{wpath} not found")
        actual = hashlib.sha256(wpath.read_bytes()).hexdigest()
        if actual != expected:
            raise BrainMRIError("BRAIN_HASH_MISMATCH", f"expected {expected}, got {actual}")

        if "efficientnet_b0" in wpath.name:
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
            img_size=256,
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
        """Return predicted class, all 4 model scores, 256px display image, and Grad-CAM++ heatmap."""
        x, display = self._preprocess(gray_0_255)
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
