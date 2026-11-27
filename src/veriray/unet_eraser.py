"""Gated-Convolution U-Net Eraser (Blueprint 26 Nov §6, §7).

Optional learned inpainting Eraser trained locally on NIH ``No Finding`` images.
Note (§10): ``No Finding`` is not a certified healthy cohort, and generated patches are
synthetic counterfactuals — NEVER true anatomy or disease-free reconstructions.

Runtime rules:
- Loads weights ONLY from ``models/`` after verifying SHA-256 against ``UNET_CHECKPOINT.json``.
- Fails closed with ``EraseError`` if weights/hash are missing or mismatched.
- Composites ``np.where(mask, pred, original)`` and verifies exact outside-mask pixel equality.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F

from veriray.contracts import EraserContract, IntegrityReport, check_outside_mask_exact_equality
from veriray.eraser import EraseError, _boundary_discontinuity, _validate

UNET_RECORD = "UNET_CHECKPOINT.json"


class GatedConv2d(nn.Module):
    """2D gated convolution: y = phi(W_f * x) * sigmoid(W_g * x)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.feat = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.gate = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(self.feat(x)) * torch.sigmoid(self.gate(x))


class GatedUNet(nn.Module):
    """Compact Gated-Convolution U-Net for 224x224 single-channel X-ray counterfactuals."""

    def __init__(self, base_ch: int = 32) -> None:
        super().__init__()
        # Input: 2 channels [masked_image_norm, mask]
        self.enc1 = nn.Sequential(GatedConv2d(2, base_ch), GatedConv2d(base_ch, base_ch))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), GatedConv2d(base_ch, base_ch * 2), GatedConv2d(base_ch * 2, base_ch * 2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), GatedConv2d(base_ch * 2, base_ch * 4), GatedConv2d(base_ch * 4, base_ch * 4))
        self.dec2 = nn.Sequential(GatedConv2d(base_ch * 6, base_ch * 2), GatedConv2d(base_ch * 2, base_ch * 2))
        self.dec1 = nn.Sequential(GatedConv2d(base_ch * 3, base_ch), GatedConv2d(base_ch, base_ch))
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, masked_img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([masked_img, mask], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        u2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


@dataclass
class UNetEraser:
    model: GatedUNet
    checkpoint_id: str
    device: torch.device

    @classmethod
    def load_optional(cls, models_dir: Path | str = "models", device: str = "cpu") -> "UNetEraser | None":
        """Load if configured in models/; return None if the checkpoint record is absent."""
        rec_path = Path(models_dir) / UNET_RECORD
        if not rec_path.is_file():
            return None
        return cls.load(models_dir, device=device)

    @classmethod
    def load(cls, models_dir: Path | str = "models", device: str = "cpu") -> "UNetEraser":
        mdir = Path(models_dir)
        rec_path = mdir / UNET_RECORD
        if not rec_path.is_file():
            raise EraseError("ERASER_UNET_CHECKPOINT_MISSING", f"{rec_path} not found")
        try:
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
            wpath = mdir / str(rec["filename"])
            expected_sha = str(rec["sha256"]).lower()
        except Exception as exc:  # noqa: BLE001
            raise EraseError("ERASER_UNET_RECORD_INVALID", str(exc)) from exc
        if not wpath.is_file():
            raise EraseError("ERASER_UNET_CHECKPOINT_MISSING", f"{wpath} not found")
        actual = hashlib.sha256(wpath.read_bytes()).hexdigest()
        if actual != expected_sha:
            raise EraseError("ERASER_UNET_HASH_MISMATCH", f"expected {expected_sha}, got {actual}")
        try:
            state = torch.load(wpath, map_location="cpu", weights_only=True)
            model = GatedUNet(base_ch=int(rec.get("base_ch", 32)))
            model.load_state_dict(state)
            model.eval()
            dev = torch.device(device)
            model.to(dev)
        except Exception as exc:  # noqa: BLE001
            raise EraseError("ERASER_UNET_LOAD_FAILED", f"{type(exc).__name__}: {exc}") from exc
        return cls(model=model, checkpoint_id=f"gated-unet@sha256:{expected_sha[:12]}", device=dev)

    def erase(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        seed: int = 0,
        unrelated_output_max_drift: float = 0.0,
    ) -> EraserContract:
        original, m = _validate(image, mask)
        # Normalize to [0, 1] and zero out the masked interior before feeding the U-Net
        img_01 = original / 255.0
        masked_01 = np.where(m, 0.0, img_01).astype(np.float32)
        t_img = torch.from_numpy(masked_01)[None, None].to(self.device)
        t_mask = torch.from_numpy(m.astype(np.float32))[None, None].to(self.device)
        with torch.inference_mode():
            pred_01 = self.model(t_img, t_mask)[0, 0].detach().cpu().numpy()
        fill = np.clip(pred_01 * 255.0, 0.0, 255.0).astype(np.float32)
        edited = np.where(m, fill, original).astype(np.float32)
        exact, max_diff = check_outside_mask_exact_equality(original, edited, m)
        if not exact:
            raise EraseError("ERASER_OUTSIDE_MASK_CHANGED", f"max abs diff = {max_diff}")
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
            fill_method="gated_unet",
            seed=int(seed),
            integrity_report=report,
            config={"method": "gated_unet", "checkpoint_id": self.checkpoint_id, "deterministic": True},
        )
