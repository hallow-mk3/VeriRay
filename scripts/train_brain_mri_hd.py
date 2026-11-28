"""Step 2B: Train 5-Class Brain MRI Reader + FPN Lesion Segmenter on Leakage-Aware Splits.

Implements all approved Change 3 requirements:
  1. Loads ONLY 'train' and 'val' from SHA-256-verified brain_mri_leakage_aware_manifest.csv
     (locked 'test' split and Kaggle 'Testing/' folder are NEVER read).
  2. Applies source-removing canonicalize_brain_slice() to EVERY image and mask across all
     sources (Figshare .mat with exact tumorMask, Br35H/SARTAJ 2D, BraTS T1ce/FLAIR glioma +
     normal slices, and ISLES DWI/FLAIR stroke + normal slices).
  3. Reports WITHIN-SOURCE metrics on dev_val as the headline:
       - Within-ISLES: ischemic_stroke vs notumor (AUROC, AUPRC, FP rate on normal ISLES, Dice)
       - Within-BraTS: glioma vs notumor (AUROC, AUPRC, Dice)
       - Within-2D: 4-class accuracy (glioma, meningioma, notumor, pituitary) + Figshare Dice
"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import cv2
import h5py
import nibabel as nib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veriray.brain_mri import (  # noqa: E402
    BRAIN_HD_CLASSES,
    BRAIN_RECORD,
    BrainHDNet,
    canonicalize_brain_slice,
)

IMG_SIZE = 256
SEG_SIZE = 64
NUM_CLASSES = len(BRAIN_HD_CLASSES)  # 5
SEED = 20261127


@dataclass
class BrainSample:
    sample_id: str
    source: str
    sequence: str
    cls_idx: int
    img_u8: np.ndarray        # canonicalized (256, 256) uint8
    seg_known: bool
    mask_u8: np.ndarray | None = None  # canonicalized (64, 64) uint8 in {0, 1}


def load_manifest_train_val() -> tuple[list[BrainSample], list[BrainSample]]:
    man_dir = ROOT / "data" / "manifests"
    csv_p = man_dir / "brain_mri_leakage_aware_manifest.csv"
    sha_p = man_dir / "brain_mri_manifest.sha256.json"
    expected = json.loads(sha_p.read_text(encoding="utf-8"))["sha256"].lower()
    actual = hashlib.sha256(csv_p.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"brain_mri_leakage_aware_manifest.csv SHA-256 mismatch: {actual} != {expected}")

    rows_to_load: list[dict] = []
    with csv_p.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] in ("train", "val"):
                rows_to_load.append(row)
            # Strictly NEVER load 'test'!

    # Group NIfTI slices by volume path so each 3D file is opened only once
    nifti_vol_cache: dict[str, np.ndarray] = {}

    def _get_vol(vpath: str) -> np.ndarray:
        if vpath not in nifti_vol_cache:
            if len(nifti_vol_cache) > 12:
                nifti_vol_cache.clear()
            arr = np.asarray(nib.load(vpath).dataobj)
            if arr.ndim == 4:
                arr = arr[..., 0]
            nifti_vol_cache[vpath] = arr
        return nifti_vol_cache[vpath]

    tr_out: list[BrainSample] = []
    va_out: list[BrainSample] = []

    # Sort by locator_spec so slices from the same NIfTI volume are contiguous (cache hit!)
    rows_to_load.sort(key=lambda r: r["locator_spec"])

    for r in rows_to_load:
        spec = r["locator_spec"]
        kind, rest = spec.split("|", 1)
        c_idx = BRAIN_HD_CLASSES.index(r["class_name"])
        raw_img: np.ndarray | None = None
        raw_msk: np.ndarray | None = None
        seg_known = r["has_mask"] == "true"

        if kind == "figshare_mat":
            with h5py.File(rest, "r") as f:
                cj = f["cjdata"]
                raw_img = np.asarray(cj["image"], dtype=np.float32).T
                if "tumorMask" in cj:
                    raw_msk = (np.asarray(cj["tumorMask"]).T > 0).astype(np.uint8)
                else:
                    seg_known = False
        elif kind == "image_file":
            with Image.open(rest) as im:
                raw_img = np.asarray(im.convert("L"), dtype=np.float32)
            if c_idx == 2:  # notumor
                raw_msk = np.zeros(raw_img.shape, dtype=np.uint8)
        elif kind == "nifti_slice":
            vol_p, msk_p, z_str = rest.split("|")
            z = int(z_str)
            vol = _get_vol(vol_p)
            raw_img = np.rot90(vol[:, :, z]).astype(np.float32)
            if msk_p == "NONE":
                raw_msk = np.zeros(raw_img.shape, dtype=np.uint8)
            else:
                mvol = _get_vol(msk_p)
                raw_msk = np.rot90((mvol[:, :, z] > 0).astype(np.uint8))

        if raw_img is None:
            continue
        can_img, can_msk = canonicalize_brain_slice(raw_img, mask=raw_msk, out_size=IMG_SIZE)
        if float(can_img.max()) <= 0.0:
            continue
        msk_64 = None
        if seg_known and can_msk is not None:
            msk_64 = (cv2.resize(can_msk, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)

        bs = BrainSample(
            sample_id=r["sample_id"],
            source=r["source"],
            sequence=r["sequence"],
            cls_idx=c_idx,
            img_u8=np.clip(np.rint(can_img), 0, 255).astype(np.uint8),
            seg_known=seg_known,
            mask_u8=msk_64,
        )
        (tr_out if r["split"] == "train" else va_out).append(bs)

    nifti_vol_cache.clear()
    return tr_out, va_out


class BrainCanonicalDataset(Dataset):
    def __init__(self, items: list[BrainSample], augment: bool = False) -> None:
        self.items = items
        self.augment = augment
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        s = self.items[idx]
        img = s.img_u8.astype(np.float32) / 255.0
        y_seg = np.zeros((NUM_CLASSES, SEG_SIZE, SEG_SIZE), dtype=np.float32)
        m_seg = np.zeros(NUM_CLASSES, dtype=np.float32)
        if s.seg_known and s.mask_u8 is not None:
            m_seg[:] = 1.0
            if s.cls_idx != 2:
                y_seg[s.cls_idx] = s.mask_u8.astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                y_seg = np.ascontiguousarray(y_seg[:, :, ::-1])
            if random.random() < 0.5:
                ang = random.uniform(-12.0, 12.0)
                mat_i = cv2.getRotationMatrix2D((IMG_SIZE / 2, IMG_SIZE / 2), ang, 1.0)
                img = cv2.warpAffine(img, mat_i, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR, borderValue=0.0)
                if m_seg.sum() > 0:
                    mat_s = cv2.getRotationMatrix2D((SEG_SIZE / 2, SEG_SIZE / 2), ang, 1.0)
                    for c in range(NUM_CLASSES):
                        if y_seg[c].max() > 0:
                            y_seg[c] = (cv2.warpAffine(y_seg[c], mat_s, (SEG_SIZE, SEG_SIZE), flags=cv2.INTER_NEAREST, borderValue=0.0) > 0.5).astype(np.float32)

        x3 = (np.repeat(img[None, :, :], 3, axis=0) - self.mean) / self.std
        return torch.from_numpy(x3), int(s.cls_idx), torch.from_numpy(y_seg), torch.from_numpy(m_seg)


@torch.no_grad()
def evaluate_within_source(net: nn.Module, val_items: list[BrainSample], val_loader: DataLoader, device: torch.device) -> dict[str, Any]:
    net.eval()
    probs_list, seg_list = [], []
    ys_list, ms_list, yseg_list = [], [], []
    for x, y, y_seg, m_seg in val_loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            l1, s1 = net(x)
            l2, s2 = net(torch.flip(x, dims=[3]))
            p_cls = torch.softmax(0.5 * (l1 + l2), dim=1).float().cpu().numpy()
            p_seg = torch.sigmoid(0.5 * (s1 + torch.flip(s2, dims=[3]))).float().cpu().numpy()
        probs_list.append(p_cls)
        seg_list.append(p_seg)
        ys_list.append(y.numpy())
        ms_list.append(m_seg.numpy())
        yseg_list.append(y_seg.numpy())

    probs = np.concatenate(probs_list, axis=0)
    segs = np.concatenate(seg_list, axis=0)
    ys = np.concatenate(ys_list, axis=0)
    ms = np.concatenate(ms_list, axis=0)
    ysegs = np.concatenate(yseg_list, axis=0)
    preds = probs.argmax(axis=1)
    srcs = np.array([s.source for s in val_items])

    # 1. Headline: Within-ISLES (ischemic_stroke [4] vs notumor [2])
    m_isles = srcs == "isles2022"
    y_isles_bin = (ys[m_isles] == 4).astype(int)
    # Score = P(stroke) / (P(stroke) + P(notumor)) and raw P(stroke)
    p_isles_raw = probs[m_isles, 4]
    p_isles_rel = probs[m_isles, 4] / np.maximum(probs[m_isles, 4] + probs[m_isles, 2], 1e-8)
    pred_isles = preds[m_isles]
    norm_isles = m_isles & (ys == 2)
    pos_isles = m_isles & (ys == 4)

    isles_dices = []
    for idx in np.where(pos_isles)[0]:
        if ms[idx, 4] > 0.5 and ysegs[idx, 4].sum() > 0:
            pb = (segs[idx, 4] >= 0.45).astype(np.float32)
            isles_dices.append(float(2.0 * (pb * ysegs[idx, 4]).sum() / max(pb.sum() + ysegs[idx, 4].sum(), 1.0)))

    within_isles = {
        "n_stroke_slices": int(pos_isles.sum()),
        "n_normal_slices": int(norm_isles.sum()),
        "auroc_p_stroke": round(float(roc_auc_score(y_isles_bin, p_isles_raw)), 4),
        "auroc_stroke_vs_notumor": round(float(roc_auc_score(y_isles_bin, p_isles_rel)), 4),
        "auprc_p_stroke": round(float(average_precision_score(y_isles_bin, p_isles_raw)), 4),
        "stroke_recall": round(float((pred_isles[y_isles_bin == 1] == 4).mean()), 4),
        "normal_specificity_not_stroke": round(float((pred_isles[y_isles_bin == 0] != 4).mean()), 4),
        "false_positive_stroke_rate_on_normal_isles": round(float((pred_isles[y_isles_bin == 0] == 4).mean()), 4),
        "stroke_mask_dice": round(float(np.mean(isles_dices)), 4) if isles_dices else 0.0,
    }

    # 2. Within-BraTS (glioma [0] vs notumor [2])
    m_brats = srcs == "brats2020"
    y_brats_bin = (ys[m_brats] == 0).astype(int)
    p_brats_raw = probs[m_brats, 0]
    pred_brats = preds[m_brats]
    brats_dices = []
    for idx in np.where(m_brats & (ys == 0))[0]:
        if ms[idx, 0] > 0.5 and ysegs[idx, 0].sum() > 0:
            pb = (segs[idx, 0] >= 0.45).astype(np.float32)
            brats_dices.append(float(2.0 * (pb * ysegs[idx, 0]).sum() / max(pb.sum() + ysegs[idx, 0].sum(), 1.0)))
    within_brats = {
        "n_glioma_slices": int((m_brats & (ys == 0)).sum()),
        "n_normal_slices": int((m_brats & (ys == 2)).sum()),
        "auroc_p_glioma": round(float(roc_auc_score(y_brats_bin, p_brats_raw)), 4),
        "auprc_p_glioma": round(float(average_precision_score(y_brats_bin, p_brats_raw)), 4),
        "glioma_recall": round(float((pred_brats[y_brats_bin == 1] == 0).mean()), 4),
        "false_positive_glioma_rate_on_normal_brats": round(float((pred_brats[y_brats_bin == 0] == 0).mean()), 4),
        "glioma_mask_dice": round(float(np.mean(brats_dices)), 4) if brats_dices else 0.0,
    }

    # 3. Within-2D (Figshare + Br35H/SARTAJ 4-class)
    m_2d = srcs == "brain_mri_2d"
    acc_2d = float(accuracy_score(ys[m_2d], preds[m_2d]))
    dices_2d: dict[str, list[float]] = defaultdict(list)
    for idx in np.where(m_2d)[0]:
        c = int(ys[idx])
        if c != 2 and ms[idx, c] > 0.5 and ysegs[idx, c].sum() > 0:
            pb = (segs[idx, c] >= 0.45).astype(np.float32)
            dices_2d[BRAIN_HD_CLASSES[c]].append(
                float(2.0 * (pb * ysegs[idx, c]).sum() / max(pb.sum() + ysegs[idx, c].sum(), 1.0))
            )
    within_2d = {
        "n_slices": int(m_2d.sum()),
        "accuracy_4cls": round(acc_2d, 4),
        "tumor_mask_dice_by_class": {k: round(float(np.mean(v)), 4) for k, v in dices_2d.items()},
    }

    all_dices = isles_dices + brats_dices + [d for v in dices_2d.values() for d in v]
    pooled_acc = float(accuracy_score(ys, preds))
    rep = classification_report(ys, preds, labels=list(range(NUM_CLASSES)), target_names=list(BRAIN_HD_CLASSES), output_dict=True, zero_division=0)
    return {
        "headline_within_isles_stroke_vs_normal": within_isles,
        "within_brats_glioma_vs_normal": within_brats,
        "within_2d_4class": within_2d,
        "pooled_5cls_dev_val_accuracy": round(pooled_acc, 4),
        "overall_lesion_segmentation_dice": round(float(np.mean(all_dices)), 4) if all_dices else 0.0,
        "classification_report_dev_val": rep,
    }


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BrainHD-Deconfounded] Device: {device}", flush=True)

    t0_load = time.time()
    tr_items, va_items = load_manifest_train_val()
    print(f"[BrainHD-Deconfounded] Loaded canonicalized train={len(tr_items)} | dev_val={len(va_items)} in {time.time()-t0_load:.1f}s", flush=True)

    cls_counts = np.bincount([s.cls_idx for s in tr_items], minlength=NUM_CLASSES).astype(np.float64)
    cls_weights = np.clip(cls_counts.mean() / np.maximum(cls_counts, 1.0), 0.5, 2.5)
    print(f"[BrainHD-Deconfounded] Train counts: {dict(zip(BRAIN_HD_CLASSES, cls_counts.astype(int).tolist()))}", flush=True)

    tr_loader = DataLoader(BrainCanonicalDataset(tr_items, augment=True), batch_size=24, shuffle=True, num_workers=0, pin_memory=True)
    va_loader = DataLoader(BrainCanonicalDataset(va_items, augment=False), batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    net = BrainHDNet(num_classes=NUM_CLASSES, pretrained=True).to(device)
    epochs = 10
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    ce_fn = nn.CrossEntropyLoss(weight=torch.tensor(cls_weights, dtype=torch.float32, device=device), label_smoothing=0.03)

    models_dir = ROOT / "models"
    weights_name = "convnext_tiny_fpn-brain-mri-5cls.pt"
    weights_path = models_dir / weights_name

    best_score = -1.0
    best_eval: dict[str, Any] = {}

    for ep in range(1, epochs + 1):
        t0 = time.time()
        net.train()
        sum_loss, steps = 0.0, 0
        for x, y, y_seg, m_seg in tr_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_seg = y_seg.to(device, non_blocking=True)
            m_seg = m_seg.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                cls_logits, seg_logits = net(x)
                l_cls = ce_fn(cls_logits, y)
                bce_s = F.binary_cross_entropy_with_logits(seg_logits, y_seg, reduction="none").mean(dim=(2, 3))
                l_seg = (bce_s * m_seg).sum() / m_seg.sum().clamp_min(1.0)
                seg_p = torch.sigmoid(seg_logits)
                pos_m = m_seg * (y_seg.amax(dim=(2, 3)) > 0.5).float()
                if pos_m.sum() > 0:
                    inter = (seg_p * y_seg).sum(dim=(2, 3))
                    den = seg_p.sum(dim=(2, 3)) + y_seg.sum(dim=(2, 3)) + 1.0
                    l_dice = ((1.0 - (2.0 * inter + 1.0) / den) * pos_m).sum() / pos_m.sum().clamp_min(1.0)
                else:
                    l_dice = cls_logits.new_tensor(0.0)
                loss = l_cls + 0.65 * l_seg + 0.65 * l_dice
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            sum_loss += float(loss.item())
            steps += 1

        sched.step()
        ev = evaluate_within_source(net, va_items, va_loader, device)
        wi = ev["headline_within_isles_stroke_vs_normal"]
        wb = ev["within_brats_glioma_vs_normal"]
        w2 = ev["within_2d_4class"]
        # Selection criterion prioritizes within-source discrimination (ISLES AUROC + BraTS AUROC + 2D acc) / 3
        comp = (wi["auroc_p_stroke"] + wb["auroc_p_glioma"] + w2["accuracy_4cls"]) / 3.0
        print(
            f"[Epoch {ep}/{epochs}] loss={sum_loss/max(steps,1):.4f} | "
            f"ISLES_AUROC={wi['auroc_p_stroke']:.4f} (FP_norm={wi['false_positive_stroke_rate_on_normal_isles']:.3f}, dice={wi['stroke_mask_dice']:.3f}) | "
            f"BraTS_AUROC={wb['auroc_p_glioma']:.4f} | 2D_acc={w2['accuracy_4cls']:.4f} | pooled={ev['pooled_5cls_dev_val_accuracy']:.4f} ({time.time()-t0:.1f}s)",
            flush=True,
        )
        if comp > best_score:
            best_score = comp
            best_eval = ev
            torch.save(net.state_dict(), weights_path)

    sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    ckpt = {
        "weights_id": "convnext_tiny_fpn-brain-mri-5cls-deconfounded",
        "filename": weights_name,
        "sha256": sha,
        "img_size": IMG_SIZE,
        "classes": list(BRAIN_HD_CLASSES),
        "train_samples": len(tr_items),
        "val_samples": len(va_items),
        "evaluated_split": "dev_val",
        "note": "Evaluated on leakage-aware dev_val split ONLY (locked test untouched; Tr-aug synthetic copies purged). Headline metric is within-source.",
        "within_2d_4cls_dev_val_accuracy": best_eval["within_2d_4class"]["accuracy_4cls"],
        "lesion_segmentation_dice_dev_val": best_eval["overall_lesion_segmentation_dice"],
        **best_eval,
    }
    (models_dir / BRAIN_RECORD).write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
    (ROOT / "artifacts" / "brain_mri_hd_metrics.json").write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
    print(f"[BrainHD-Deconfounded] Saved {weights_name} (sha256={sha[:16]}...) composite={best_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
