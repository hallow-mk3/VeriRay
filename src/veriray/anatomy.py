"""Offline anatomical lung segmentation (Blueprint 26 Nov §6, §7 step 10).

Wraps TorchXRayVision's ChestX-Det PSPNet architecture, loading weights ONLY from
``models/`` after verifying SHA-256 against ``models/LUNG_SEG_CHECKPOINT.json``.
Never downloads at runtime. If weights are absent, ``load_optional()`` returns ``None``
so the Judge honestly records ``lung_mask_available = False``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchxrayvision.baseline_models.chestx_det.ptsemseg.pspnet import pspnet

LUNG_SEG_RECORD = "LUNG_SEG_CHECKPOINT.json"
PSPNET_TARGETS = (
    "Left Clavicle", "Right Clavicle", "Left Scapula", "Right Scapula",
    "Left Lung", "Right Lung",
    "Left Hilus Pulmonis", "Right Hilus Pulmonis",
    "Heart", "Aorta", "Facies Diaphragmatica", "Mediastinum", "Weasand", "Spine",
)
LEFT_LUNG_IDX = 4
RIGHT_LUNG_IDX = 5
LEFT_HILUS_IDX = 6
RIGHT_HILUS_IDX = 7
HEART_IDX = 8
AORTA_IDX = 9
DIAPHRAGM_IDX = 10
MEDIASTINUM_IDX = 11
WEASAND_IDX = 12
SPINE_IDX = 13

_THORACIC_ROI_MAP: dict[str, list[int]] = {
    "cardiomegaly": [LEFT_LUNG_IDX, RIGHT_LUNG_IDX, HEART_IDX, MEDIASTINUM_IDX],
    "enlarged cardiomediastinum": [LEFT_LUNG_IDX, RIGHT_LUNG_IDX, HEART_IDX, AORTA_IDX, MEDIASTINUM_IDX],
    "aortic_enlargement": [LEFT_LUNG_IDX, RIGHT_LUNG_IDX, HEART_IDX, AORTA_IDX, MEDIASTINUM_IDX],
    "effusion": [LEFT_LUNG_IDX, RIGHT_LUNG_IDX, DIAPHRAGM_IDX],
    "pleural_thickening": [LEFT_LUNG_IDX, RIGHT_LUNG_IDX, DIAPHRAGM_IDX],
    "calcification": list(range(len(PSPNET_TARGETS))),
    "fracture": list(range(len(PSPNET_TARGETS))),
    "other_lesion": list(range(len(PSPNET_TARGETS))),
}


class AnatomyError(Exception):
    def __init__(self, reason_code: str, message: str = "") -> None:
        super().__init__(f"{reason_code}: {message}" if message else reason_code)
        self.reason_code = reason_code


def _strip_module_prefix(state_dict: dict) -> OrderedDict:
    out: OrderedDict = OrderedDict()
    for k, v in state_dict.items():
        out[k[7:] if k.startswith("module.") else k] = v
    return out


@dataclass
class LungSegmenter:
    model: torch.nn.Module
    checkpoint_id: str
    device: torch.device

    @classmethod
    def load(cls, models_dir: Path | str = "models", device: str = "cpu") -> "LungSegmenter":
        mdir = Path(models_dir)
        rec_path = mdir / LUNG_SEG_RECORD
        if not rec_path.is_file():
            raise AnatomyError("LUNG_SEG_CHECKPOINT_MISSING", f"{rec_path} not found")
        try:
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
            wpath = mdir / str(rec["filename"])
            expected_sha = str(rec["sha256"]).lower()
        except Exception as exc:  # noqa: BLE001
            raise AnatomyError("LUNG_SEG_RECORD_INVALID", str(exc)) from exc
        if not wpath.is_file():
            raise AnatomyError("LUNG_SEG_CHECKPOINT_MISSING", f"{wpath} not found")
        actual = hashlib.sha256(wpath.read_bytes()).hexdigest()
        if actual != expected_sha:
            raise AnatomyError("LUNG_SEG_HASH_MISMATCH", f"expected {expected_sha}, got {actual}")
        try:
            ckpt = torch.load(wpath, map_location="cpu", weights_only=True)
            model = pspnet(len(PSPNET_TARGETS))
            model.load_state_dict(_strip_module_prefix(ckpt))
            model.eval()
            dev = torch.device(device)
            model.to(dev)
        except Exception as exc:  # noqa: BLE001
            raise AnatomyError("LUNG_SEG_LOAD_FAILED", f"{type(exc).__name__}: {exc}") from exc
        return cls(model=model, checkpoint_id=f"chestx-det-pspnet@sha256:{expected_sha[:12]}", device=dev)

    @classmethod
    def load_optional(cls, models_dir: Path | str = "models", device: str = "cpu") -> "LungSegmenter | None":
        """Load if configured in models/; return None if weights have not been fetched yet."""
        rec_path = Path(models_dir) / LUNG_SEG_RECORD
        if not rec_path.is_file():
            return None
        return cls.load(models_dir, device=device)

    def segment_lungs(self, image_224_0_255: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return (224, 224) bool mask of Left Lung | Right Lung."""
        return self.segment_thoracic_roi(image_224_0_255, finding_id="pneumothorax", threshold=threshold)

    def segment_thoracic_roi(
        self,
        image_224_0_255: np.ndarray,
        *,
        finding_id: str = "pneumothorax",
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Return (H, W) bool ROI mask matching input image dimensions for the target finding.

        Parenchymal/pleural findings use Left Lung | Right Lung.
        Cardiomegaly includes Heart | Mediastinum alongside the lungs.
        """
        img = np.asarray(image_224_0_255, dtype=np.float32)
        if img.ndim != 2 or img.shape[0] < 64 or img.shape[1] < 64:
            raise AnatomyError("LUNG_SEG_BAD_SHAPE", f"expected 2D image >= (64, 64), got {img.shape}")
        out_h, out_w = int(img.shape[0]), int(img.shape[1])
        # Match ChestX-Det PSPNet normalization: [0, 1] -> ImageNet mean/std, 512x512 RGB
        x01 = np.clip(img / 255.0, 0.0, 1.0)
        t = torch.from_numpy(x01)[None, None].repeat(1, 3, 1, 1).to(self.device)
        if (out_h, out_w) != (512, 512):
            t = F.interpolate(t, size=(512, 512), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        t = (t - mean) / std
        indices = _THORACIC_ROI_MAP.get(
            finding_id.lower().strip(),
            [LEFT_LUNG_IDX, RIGHT_LUNG_IDX],
        )
        with torch.inference_mode():
            logits = self.model(t)
            roi_probs = torch.sigmoid(logits[:, indices]).max(dim=1, keepdim=True).values
            roi_out = F.interpolate(roi_probs, size=(out_h, out_w), mode="bilinear", align_corners=False)[0, 0]
        return (roi_out.detach().cpu().numpy() >= threshold).astype(bool)
