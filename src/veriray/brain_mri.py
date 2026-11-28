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
MAX_BRAIN_UPLOAD_BYTES: int = 50 * 1024 * 1024
MAX_NIFTI_DECOMPRESSED_BYTES: int = 128 * 1024 * 1024
MAX_BRAIN_IMAGE_PIXELS: int = 4096 * 4096
MAX_NIFTI_VOXELS: int = 512 * 512 * 320
MIN_BRAIN_DIM: int = 64


def canonicalize_brain_slice(
    gray: np.ndarray,
    mask: np.ndarray | None = None,
    out_size: int = 256,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Source-removing canonical brain MRI preprocessing (Change 3).

    1. Segments largest head/brain foreground component and fills internal holes.
    2. Strips outer scalp/skull rim when unstripped 2D head background is detected.
    3. Radially normalizes each row/angle onto a common inscribed brain ellipse
       (rx=0.43*W, ry=0.45*H) so skull-stripped NIfTIs and 2D scans share identical
       outer geometry with zero cortical/margin notches.
    4. Applies common PSF smoothing (sigma=0.85) and exact within-ellipse Gaussian
       rank-quantile intensity standardization (mean=128, std=38 in [15, 245]; outside=0).
    """
    arr = np.asarray(gray, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise BrainMRIError("BRAIN_BAD_INPUT", "expected non-empty finite 2D grayscale array")

    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        z_out = np.zeros((out_size, out_size), dtype=np.float32)
        m_out = np.zeros((out_size, out_size), dtype=np.uint8) if mask is not None else None
        return z_out, m_out

    norm255 = (arr - lo) / (hi - lo) * 255.0
    blur = cv2.GaussianBlur(norm255, (5, 5), 1.0)
    zero_bg_frac = float((norm255 <= 2.0).mean())
    # Skull-stripped NIfTIs already have exact-zero background (>= 25% of frame);
    # 2D JPG scans have non-zero JPEG background + bright scalp/skull rim.
    is_skull_stripped = zero_bg_frac >= 0.28 and float(
        np.concatenate([norm255[:8, :].ravel(), norm255[-8:, :].ravel()]).mean()
    ) < 3.0
    thr = 3.0 if is_skull_stripped else max(14.0, float(np.percentile(blur, 18)))
    fg = (blur > thr).astype(np.uint8)

    # Largest connected component + contour hole-fill
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if num_l > 1:
        best_lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = (labels == best_lbl).astype(np.uint8)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(fg)
    if contours:
        cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    else:
        filled = fg

    # Peel outer scalp/skull rim on unstripped 2D scans
    if not is_skull_stripped and filled.sum() > 400:
        h0, w0 = filled.shape
        k_peel = max(3, int(round(0.065 * min(h0, w0))) | 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_peel, k_peel))
        eroded = cv2.erode(filled, ker, iterations=1)
        if eroded.sum() > 200:
            filled = eroded

    ys, xs = np.where(filled > 0)
    if ys.size < 64:
        ys, xs = np.where(norm255 > lo)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    crop_img = norm255[y1:y2, x1:x2]
    crop_fg = filled[y1:y2, x1:x2]
    res_img = cv2.resize(crop_img, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    res_fg = (cv2.resize(crop_fg, (out_size, out_size), interpolation=cv2.INTER_NEAREST) > 0)
    res_msk = None
    if mask is not None:
        crop_m = np.asarray(mask, dtype=np.uint8)[y1:y2, x1:x2]
        res_msk = (cv2.resize(crop_m, (out_size, out_size), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)

    # Row-wise horizontal span alignment onto canonical inner-parenchyma ellipse
    # (cx=0.5*S, cy=0.5*S, rx=0.41*S, ry=0.41*S) + concentric radial-shell standardization
    # so sequence/rim profiles (T1 bright dura vs DWI dark CSF rim) cannot leak source identity!
    cx = (out_size - 1) * 0.5
    cy = (out_size - 1) * 0.5
    rx = 0.41 * out_size
    ry = 0.41 * out_size
    warped = np.zeros((out_size, out_size), dtype=np.float32)
    warped_m = np.zeros((out_size, out_size), dtype=np.uint8) if res_msk is not None else None
    canon_mask = np.zeros((out_size, out_size), dtype=bool)
    rad_map = np.ones((out_size, out_size), dtype=np.float32)

    y_top = max(0, int(np.ceil(cy - ry)))
    y_bot = min(out_size - 1, int(np.floor(cy + ry)))
    fg_rows = np.where(res_fg.any(axis=1))[0]
    fy1 = int(fg_rows[0]) if fg_rows.size > 0 else 0
    fy2 = int(fg_rows[-1]) if fg_rows.size > 0 else (out_size - 1)

    for y_out in range(y_top, y_bot + 1):
        dy = (y_out - cy) / ry
        if abs(dy) >= 1.0:
            continue
        half_w = rx * float(np.sqrt(max(0.0, 1.0 - dy * dy)))
        xl = int(np.ceil(cx - half_w))
        xr = int(np.floor(cx + half_w))
        if xr <= xl:
            continue
        t_y = (y_out - y_top) / max(float(y_bot - y_top), 1.0)
        src_y = int(np.clip(round(fy1 + t_y * (fy2 - fy1)), 0, out_size - 1))
        row_fg = np.where(res_fg[src_y])[0]
        if row_fg.size >= 4:
            sxl, sxr = float(row_fg[0]), float(row_fg[-1])
        else:
            sxl, sxr = float(xl), float(xr)
        xs_out = np.arange(xl, xr + 1, dtype=np.float32)
        u = (xs_out - xl) / max(float(xr - xl), 1.0)
        src_xs = np.clip(sxl + u * (sxr - sxl), 0.0, out_size - 1.0)
        x0 = np.floor(src_xs).astype(np.int32)
        x1_i = np.minimum(x0 + 1, out_size - 1)
        w_x = src_xs - x0
        warped[y_out, xl : xr + 1] = (1.0 - w_x) * res_img[src_y, x0] + w_x * res_img[src_y, x1_i]
        canon_mask[y_out, xl : xr + 1] = True
        dx_arr = (xs_out - cx) / rx
        rad_map[y_out, xl : xr + 1] = np.sqrt(dx_arr * dx_arr + dy * dy)
        if warped_m is not None and res_msk is not None:
            warped_m[y_out, xl : xr + 1] = res_msk[src_y, np.rint(src_xs).astype(np.int32)]

    # Common PSF smoothing inside mask (sigma=1.1 equalizes DWI vs 2D resolution)
    w_flt = canon_mask.astype(np.float32)
    num = cv2.GaussianBlur(warped * w_flt, (0, 0), 1.1)
    den = cv2.GaussianBlur(w_flt, (0, 0), 1.1)
    smoothed = np.where(canon_mask & (den > 1e-5), num / np.maximum(den, 1e-5), 0.0)

    # Concentric radial-shell rank-quantile standardization (12 equal-area radial shells):
    # Removes radial sequence/skull-rim signatures while preserving focal lesions!
    out_img = np.zeros((out_size, out_size), dtype=np.float32)
    if int(canon_mask.sum()) > 64:
        r_sq = (rad_map[canon_mask]) ** 2  # uniform in [0, 1] over ellipse area
        v_all = smoothed[canon_mask]
        stand_all = np.empty_like(v_all)
        n_shells = 12
        edges = np.linspace(0.0, 1.0001, n_shells + 1)
        for s_i in range(n_shells):
            in_sh = (r_sq >= edges[s_i]) & (r_sq < edges[s_i + 1])
            vs = v_all[in_sh]
            if vs.size < 4:
                stand_all[in_sh] = 128.0
                continue
            # Add tiny deterministic spatial tie-breaker so flat quantized JPEG patches
            # don't produce quantized plateaus that leak bit-depth!
            idx_sh = np.where(in_sh)[0].astype(np.float32)
            jitter = 1e-4 * np.sin(idx_sh * 12.9898)
            order = np.argsort(vs + jitter, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float32)
            ranks[order] = (np.arange(vs.size, dtype=np.float32) + 0.5) / float(vs.size)
            z = np.sqrt(2.0) * torch.erfinv(torch.from_numpy(np.clip(2.0 * ranks - 1.0, -0.999, 0.999))).numpy()
            stand_all[in_sh] = np.clip(128.0 + 38.0 * z, 15.0, 245.0)
        out_img[canon_mask] = stand_all
    else:
        out_img = smoothed.astype(np.float32)

    return out_img, warped_m


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


def _decompress_gzip_bounded(payload: bytes, max_bytes: int = MAX_NIFTI_DECOMPRESSED_BYTES) -> bytes:
    """Stream-decompress gzip payload in memory with a strict byte ceiling to block gzip bombs."""
    chunks: list[bytes] = []
    total = 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as gz:
            while True:
                piece = gz.read(1024 * 1024)
                if not piece:
                    break
                total += len(piece)
                if total > max_bytes:
                    raise BrainMRIError(
                        "IMAGE_TOO_LARGE_OR_BOMB",
                        f"decompressed NIfTI exceeds {max_bytes // (1024 * 1024)} MB safety limit",
                    )
                chunks.append(piece)
    except BrainMRIError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise BrainMRIError("BRAIN_BAD_INPUT", f"corrupt gzip stream: {exc}") from exc
    return b"".join(chunks)


def _load_nifti_in_memory(raw_nii: bytes):
    """Parse an uncompressed .nii byte buffer 100% in RAM without writing temporary files to disk."""
    import nibabel as nib

    fh = nib.FileHolder(fileobj=io.BytesIO(raw_nii))
    for cls_nii in (nib.Nifti1Image, nib.Nifti2Image):
        try:
            return cls_nii.from_file_map({"header": fh, "image": fh})
        except Exception:  # noqa: BLE001
            fh.fileobj.seek(0)
    raise BrainMRIError("BRAIN_BAD_INPUT", "buffer is not a valid NIfTI-1 or NIfTI-2 volume")


def decode_brain_upload(payload: bytes, filename: str = "slice.png") -> np.ndarray:
    """Gate and decode a 2D MRI slice or 3D NIfTI (.nii/.nii.gz) 100% in memory to [0, 255] float32."""
    if not payload:
        raise BrainMRIError("BRAIN_BAD_INPUT", "empty upload")
    if len(payload) > MAX_BRAIN_UPLOAD_BYTES:
        raise BrainMRIError("INPUT_TOO_LARGE", f"upload exceeds {MAX_BRAIN_UPLOAD_BYTES} bytes")
    lower = filename.lower()
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        try:
            raw = _decompress_gzip_bounded(payload) if lower.endswith(".nii.gz") else payload
            if len(raw) > MAX_NIFTI_DECOMPRESSED_BYTES:
                raise BrainMRIError("IMAGE_TOO_LARGE_OR_BOMB", "NIfTI exceeds decompressed byte limit")
            img_nii = _load_nifti_in_memory(raw)
            hdr_shape = tuple(int(d) for d in img_nii.header.get_data_shape())
            if len(hdr_shape) not in (3, 4) or any(d <= 0 for d in hdr_shape):
                raise BrainMRIError("BRAIN_BAD_INPUT", f"unexpected NIfTI header shape {hdr_shape}")
            spatial_voxels = hdr_shape[0] * hdr_shape[1] * hdr_shape[2]
            if hdr_shape[0] < MIN_BRAIN_DIM or hdr_shape[1] < MIN_BRAIN_DIM:
                raise BrainMRIError("IMAGE_TOO_SMALL", f"NIfTI slice {hdr_shape[:2]} below {MIN_BRAIN_DIM}px")
            if spatial_voxels > MAX_NIFTI_VOXELS or (len(hdr_shape) == 4 and hdr_shape[3] > 16):
                raise BrainMRIError("IMAGE_TOO_LARGE_OR_BOMB", f"NIfTI dimensions {hdr_shape} exceed voxel limit")
            vol = np.asarray(img_nii.dataobj, dtype=np.float32)
            if vol.ndim == 4:
                vol = vol[..., 0]
            if vol.ndim != 3 or vol.size == 0:
                raise BrainMRIError("BRAIN_BAD_INPUT", f"unexpected NIfTI shape {vol.shape}")
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
        with Image.open(io.BytesIO(payload)) as im:
            w, h = im.size
            if w <= 0 or h <= 0 or w > 4096 or h > 4096 or (w * h) > MAX_BRAIN_IMAGE_PIXELS:
                raise BrainMRIError("IMAGE_TOO_LARGE_OR_BOMB", f"image size {w}x{h} exceeds safe bounds")
            if w < MIN_BRAIN_DIM or h < MIN_BRAIN_DIM:
                raise BrainMRIError("IMAGE_TOO_SMALL", f"image size {w}x{h} below minimum {MIN_BRAIN_DIM}x{MIN_BRAIN_DIM}")
            im.load()
            if im.mode not in ("L", "I", "I;16", "F"):
                rgb = np.asarray(im.convert("RGB"), dtype=np.float32)
                r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
                spread = float(np.mean(rgb.max(axis=2) - rgb.min(axis=2)))
                chroma_std = max(float(np.std(r - g)), float(np.std(g - b)), float(np.std(r - b)))
                if spread > 12.0 or chroma_std > 10.0:
                    raise BrainMRIError("NON_MRI_COLOR_IMAGE", "multi-hue color photograph rejected by Brain MRI gate")
                gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
            else:
                gray = np.asarray(im.convert("L"), dtype=np.float32)
    except BrainMRIError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise BrainMRIError("CORRUPT_OR_UNREADABLE_IMAGE", str(exc)) from exc

    if not np.all(np.isfinite(gray)):
        raise BrainMRIError("BRAIN_BAD_INPUT", "non-finite pixel values")
    dyn = float(gray.max() - gray.min())
    if dyn < 15.0 or float(gray.std()) < 3.0:
        raise BrainMRIError("LOW_CONTRAST_OR_BLANK", "image lacks minimum anatomical contrast")
    return gray


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
        if arr.shape[0] < MIN_BRAIN_DIM or arr.shape[1] < MIN_BRAIN_DIM:
            raise BrainMRIError("IMAGE_TOO_SMALL", f"expected >= {MIN_BRAIN_DIM}x{MIN_BRAIN_DIM}, got {arr.shape}")
        if isinstance(self.model, BrainHDNet):
            resized, _ = canonicalize_brain_slice(arr, mask=None, out_size=self.img_size)
        else:
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
