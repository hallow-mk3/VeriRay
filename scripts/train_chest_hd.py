"""Train 512x512 Dual-Head ConvNeXt-Tiny + FPN Spatial Locator across 16 Chest Findings.

Datasets combined with masked multi-task loss (zero cross-dataset false negatives):
  1. VinBigData 512x512 (15,000 scans, 67,914 radiologist box annotations across 14 classes + No finding)
  2. SIIM-ACR Pneumothorax (10,675 scans + exact pixel-level pneumothorax masks)
  3. RSNA Pneumonia (26,684 scans + radiologist pneumonia/lung-opacity bounding boxes)
  4. TB Chest Radiography Database (4,200 scans: Normal + Tuberculosis)

Strict split discipline:
  - Deterministic SHA-256 split by unique scan/patient ID (85% train / 15% val).
  - Zero overlap asserted before training starts.
  - Per-finding Platt calibration fitted on val split ONLY.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veriray.chest_hd import (  # noqa: E402
    CHEST_HD_DISPLAY_NAMES,
    CHEST_HD_FINDINGS,
    CHEST_HD_RECORD,
    ChestHDNet,
)

DATASETS_DIR = Path(r"C:\Users\swast\datasets")
IMG_SIZE = 512
SEG_SIZE = 128
NUM_CLASSES = len(CHEST_HD_FINDINGS)  # 16
SEED = 20261127


@dataclass
class SampleMeta:
    uid: str
    source: str
    img_path: Path
    # Channels with known binary labels & values
    known_cls_indices: list[int]
    pos_cls_indices: list[int]
    # Spatial supervision: either boxes (cls_idx, x1_norm, y1_norm, x2_norm, y2_norm) or mask_path
    known_seg_indices: list[int]
    boxes_norm: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    pneumo_mask_path: Path | None = None


def _is_train(uid: str) -> bool:
    h = int(hashlib.sha256(uid.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 100) < 85


def load_all_metadata() -> tuple[list[SampleMeta], list[SampleMeta]]:
    samples: list[SampleMeta] = []

    # 1. VinBigData (classes 0..13)
    vbd_dir = DATASETS_DIR / "vindr_cxr" / "vinbigdata"
    vbd_csv = vbd_dir / "train.csv"
    if vbd_csv.is_file():
        df = pd.read_csv(vbd_csv)
        all_14 = list(range(14))
        for img_id, grp in df.groupby("image_id"):
            img_path = vbd_dir / "train" / f"{img_id}.png"
            if not img_path.is_file():
                continue
            pos_set: set[int] = set()
            boxes: list[tuple[int, float, float, float, float]] = []
            for row in grp.itertuples(index=False):
                cid = int(row.class_id)
                if cid < 14:
                    pos_set.add(cid)
                    w = float(row.width) if pd.notna(row.width) and float(row.width) > 0 else 1.0
                    h = float(row.height) if pd.notna(row.height) and float(row.height) > 0 else 1.0
                    if pd.notna(row.x_min) and pd.notna(row.x_max):
                        x1 = float(np.clip(float(row.x_min) / w, 0.0, 1.0))
                        y1 = float(np.clip(float(row.y_min) / h, 0.0, 1.0))
                        x2 = float(np.clip(float(row.x_max) / w, 0.0, 1.0))
                        y2 = float(np.clip(float(row.y_max) / h, 0.0, 1.0))
                        if x2 > x1 and y2 > y1:
                            boxes.append((cid, x1, y1, x2, y2))
            samples.append(
                SampleMeta(
                    uid=f"vbd:{img_id}",
                    source="vinbigdata",
                    img_path=img_path,
                    known_cls_indices=all_14,
                    pos_cls_indices=sorted(pos_set),
                    known_seg_indices=all_14,
                    boxes_norm=boxes,
                )
            )

    # 2. Tuberculosis (class 14: tuberculosis)
    tb_root = DATASETS_DIR / "tb_cxr" / "TB_Chest_Radiography_Database"
    if tb_root.is_dir():
        for folder, is_pos in [("Normal", False), ("Tuberculosis", True)]:
            for p in sorted((tb_root / folder).glob("*.png")):
                samples.append(
                    SampleMeta(
                        uid=f"tb:{folder}:{p.stem}",
                        source="tb_cxr",
                        img_path=p,
                        known_cls_indices=[14],
                        pos_cls_indices=[14] if is_pos else [],
                        known_seg_indices=[] if is_pos else [14],  # Normal scans known zero TB mask
                    )
                )

    # 3. SIIM-ACR Pneumothorax (class 12: pneumothorax + exact pixel masks)
    siim_root = DATASETS_DIR / "siim_pneumothorax" / "siim-acr-pneumothorax"
    siim_csv = siim_root / "stage_1_train_images.csv"
    if siim_csv.is_file():
        sdf = pd.read_csv(siim_csv)
        for row in sdf.itertuples(index=False):
            fname = str(row.new_filename)
            img_p = siim_root / "png_images" / fname
            msk_p = siim_root / "png_masks" / fname
            if not img_p.is_file():
                continue
            is_pos = int(row.has_pneumo) == 1
            samples.append(
                SampleMeta(
                    uid=f"siim:{row.ImageId}",
                    source="siim_pneumo",
                    img_path=img_p,
                    known_cls_indices=[12],
                    pos_cls_indices=[12] if is_pos else [],
                    known_seg_indices=[12] if msk_p.is_file() else [],
                    pneumo_mask_path=msk_p if (is_pos and msk_p.is_file()) else None,
                )
            )

    # 4. RSNA Pneumonia (class 15: pneumonia + class 7: lung_opacity + boxes)
    rsna_root = DATASETS_DIR / "rsna_pneumonia" / "input"
    rsna_csv = rsna_root / "stage_2_train_labels.csv"
    if rsna_csv.is_file():
        rdf = pd.read_csv(rsna_csv)
        for pid, grp in rdf.groupby("patientId"):
            img_p = rsna_root / "images" / f"{pid}.jpg"
            if not img_p.is_file():
                img_p = rsna_root / "images" / f"{pid}.png"
                if not img_p.is_file():
                    continue
            is_pos = any(int(r.Target) == 1 for r in grp.itertuples(index=False))
            boxes_r: list[tuple[int, float, float, float, float]] = []
            if is_pos:
                for r in grp.itertuples(index=False):
                    if int(r.Target) == 1 and pd.notna(r.x) and pd.notna(r.width):
                        x1 = float(np.clip(float(r.x) / 1024.0, 0.0, 1.0))
                        y1 = float(np.clip(float(r.y) / 1024.0, 0.0, 1.0))
                        x2 = float(np.clip((float(r.x) + float(r.width)) / 1024.0, 0.0, 1.0))
                        y2 = float(np.clip((float(r.y) + float(r.height)) / 1024.0, 0.0, 1.0))
                        if x2 > x1 and y2 > y1:
                            boxes_r.append((15, x1, y1, x2, y2))
                            boxes_r.append((7, x1, y1, x2, y2))
            samples.append(
                SampleMeta(
                    uid=f"rsna:{pid}",
                    source="rsna_pneumonia",
                    img_path=img_p,
                    known_cls_indices=[7, 15],
                    pos_cls_indices=[7, 15] if is_pos else [],
                    known_seg_indices=[7, 15],
                    boxes_norm=boxes_r,
                )
            )

    train_s = [s for s in samples if _is_train(s.uid)]
    val_s = [s for s in samples if not _is_train(s.uid)]
    train_ids = {s.uid for s in train_s}
    val_ids = {s.uid for s in val_s}
    assert train_ids.isdisjoint(val_ids), "CRITICAL: train/val UID overlap detected!"
    return train_s, val_s


def build_epoch_subset(train_all: list[SampleMeta], epoch: int) -> list[SampleMeta]:
    """Keep 100% of VinBigData + TB + all SIIM/RSNA positives + rotating negative subsample for balanced epochs."""
    rng = random.Random(SEED + epoch)
    out: list[SampleMeta] = []
    siim_neg: list[SampleMeta] = []
    rsna_neg: list[SampleMeta] = []
    for s in train_all:
        if s.source in ("vinbigdata", "tb_cxr") or s.pos_cls_indices:
            out.append(s)
        elif s.source == "siim_pneumo":
            siim_neg.append(s)
        elif s.source == "rsna_pneumonia":
            rsna_neg.append(s)
    rng.shuffle(siim_neg)
    rng.shuffle(rsna_neg)
    out.extend(siim_neg[:3500])
    out.extend(rsna_neg[:4000])
    rng.shuffle(out)
    return out


class ChestMultiDataset(Dataset):
    def __init__(self, items: list[SampleMeta], augment: bool = False) -> None:
        self.items = items
        self.augment = augment
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        s = self.items[idx]
        with Image.open(s.img_path) as im:
            gray = np.asarray(im.convert("L"), dtype=np.uint8)
        if gray.shape != (IMG_SIZE, IMG_SIZE):
            gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        y_cls = np.zeros(NUM_CLASSES, dtype=np.float32)
        m_cls = np.zeros(NUM_CLASSES, dtype=np.float32)
        m_cls[s.known_cls_indices] = 1.0
        if s.pos_cls_indices:
            y_cls[s.pos_cls_indices] = 1.0

        y_seg = np.zeros((NUM_CLASSES, SEG_SIZE, SEG_SIZE), dtype=np.float32)
        m_seg = np.zeros(NUM_CLASSES, dtype=np.float32)
        if s.known_seg_indices:
            m_seg[s.known_seg_indices] = 1.0

        for cid, x1, y1, x2, y2 in s.boxes_norm:
            ix1 = max(0, min(SEG_SIZE - 1, int(round(x1 * SEG_SIZE))))
            iy1 = max(0, min(SEG_SIZE - 1, int(round(y1 * SEG_SIZE))))
            ix2 = max(ix1 + 1, min(SEG_SIZE, int(round(x2 * SEG_SIZE))))
            iy2 = max(iy1 + 1, min(SEG_SIZE, int(round(y2 * SEG_SIZE))))
            # Soft elliptical interior + box union so centers get strongest supervision
            y_seg[cid, iy1:iy2, ix1:ix2] = np.maximum(y_seg[cid, iy1:iy2, ix1:ix2], 0.85)
            cx, cy = (ix1 + ix2) // 2, (iy1 + iy2) // 2
            rx, ry = max(1, (ix2 - ix1) // 2), max(1, (iy2 - iy1) // 2)
            ell = np.zeros((SEG_SIZE, SEG_SIZE), dtype=np.float32)
            cv2.ellipse(ell, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
            y_seg[cid] = np.maximum(y_seg[cid], ell)

        if s.pneumo_mask_path is not None:
            with Image.open(s.pneumo_mask_path) as mim:
                pm = np.asarray(mim.convert("L"), dtype=np.uint8)
            pm_small = cv2.resize(pm, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_AREA)
            y_seg[12] = np.maximum(y_seg[12], (pm_small > 64).astype(np.float32))

        img_f = gray.astype(np.float32) / 255.0
        if self.augment:
            if random.random() < 0.5:
                img_f = np.ascontiguousarray(img_f[:, ::-1])
                y_seg = np.ascontiguousarray(y_seg[:, :, ::-1])
            if random.random() < 0.5:
                alpha = random.uniform(0.88, 1.12)
                beta = random.uniform(-0.06, 0.06)
                img_f = np.clip(img_f * alpha + beta, 0.0, 1.0)

        x3 = (np.repeat(img_f[None, :, :], 3, axis=0) - self.mean) / self.std
        return (
            torch.from_numpy(x3),
            torch.from_numpy(y_cls),
            torch.from_numpy(m_cls),
            torch.from_numpy(y_seg),
            torch.from_numpy(m_seg),
        )


def compute_pos_weights(train_items: list[SampleMeta]) -> torch.Tensor:
    pos = np.zeros(NUM_CLASSES, dtype=np.float64)
    tot = np.zeros(NUM_CLASSES, dtype=np.float64)
    for s in train_items:
        for c in s.known_cls_indices:
            tot[c] += 1.0
        for c in s.pos_cls_indices:
            pos[c] += 1.0
    neg = np.maximum(tot - pos, 1.0)
    pw = np.clip(np.sqrt(neg / np.maximum(pos, 1.0)), 1.0, 8.0)
    return torch.tensor(pw, dtype=torch.float32)


def masked_loss(
    cls_logits: torch.Tensor,
    seg_logits: torch.Tensor,
    y_cls: torch.Tensor,
    m_cls: torch.Tensor,
    y_seg: torch.Tensor,
    m_seg: torch.Tensor,
    pos_weight: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    bce_cls = F.binary_cross_entropy_with_logits(cls_logits, y_cls, pos_weight=pos_weight, reduction="none")
    loss_cls = (bce_cls * m_cls).sum() / m_cls.sum().clamp_min(1.0)

    # Spatial BCE + positive-channel Soft Dice
    bce_seg = F.binary_cross_entropy_with_logits(seg_logits, y_seg, reduction="none").mean(dim=(2, 3))
    loss_seg_bce = (bce_seg * m_seg).sum() / m_seg.sum().clamp_min(1.0)

    seg_prob = torch.sigmoid(seg_logits)
    pos_seg_mask = m_seg * (y_seg.amax(dim=(2, 3)) > 0.5).float()
    if pos_seg_mask.sum() > 0:
        inter = (seg_prob * y_seg).sum(dim=(2, 3))
        denom = seg_prob.sum(dim=(2, 3)) + y_seg.sum(dim=(2, 3)) + 1.0
        dice_per_ch = 1.0 - (2.0 * inter + 1.0) / denom
        loss_dice = (dice_per_ch * pos_seg_mask).sum() / pos_seg_mask.sum().clamp_min(1.0)
    else:
        loss_dice = cls_logits.new_tensor(0.0)

    total = loss_cls + 0.65 * loss_seg_bce + 0.55 * loss_dice
    return total, float(loss_cls.item()), float((loss_seg_bce + loss_dice).item())


@torch.no_grad()
def evaluate_val(
    net: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    net.eval()
    all_probs: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_m: list[np.ndarray] = []
    dice_sums = np.zeros(NUM_CLASSES, dtype=np.float64)
    dice_cnts = np.zeros(NUM_CLASSES, dtype=np.float64)

    for x, y_cls, m_cls, y_seg, m_seg in val_loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            cls_logits, seg_logits = net(x)
            cls_p = torch.sigmoid(cls_logits).float()
            seg_p = torch.sigmoid(seg_logits).float()
            # Match ChestHDReaderAdapter inference fusion: 0.80 cls + 0.20 top-2% spatial
            top_k = max(1, int(0.02 * SEG_SIZE * SEG_SIZE))
            seg_peaks = torch.topk(seg_p.flatten(2), top_k, dim=2).values.mean(dim=2)
            fused_p = 0.80 * cls_p + 0.20 * seg_peaks

        all_probs.append(fused_p.cpu().numpy())
        all_y.append(y_cls.numpy())
        all_m.append(m_cls.numpy())

        sp_np = (seg_p.cpu().numpy() >= 0.40).astype(np.float32)
        ys_np = (y_seg.numpy() >= 0.50).astype(np.float32)
        ms_np = m_seg.numpy()
        for b in range(sp_np.shape[0]):
            for c in range(NUM_CLASSES):
                if ms_np[b, c] > 0.5 and ys_np[b, c].sum() > 0:
                    inter = float((sp_np[b, c] * ys_np[b, c]).sum())
                    denom = float(sp_np[b, c].sum() + ys_np[b, c].sum())
                    dice_sums[c] += (2.0 * inter) / max(denom, 1.0)
                    dice_cnts[c] += 1.0

    probs = np.concatenate(all_probs, axis=0)
    ys = np.concatenate(all_y, axis=0)
    ms = np.concatenate(all_m, axis=0)

    metrics: dict[str, Any] = {}
    calibrators: dict[str, dict[str, float]] = {}
    aurocs: list[float] = []
    for c, fid in enumerate(CHEST_HD_FINDINGS):
        valid = ms[:, c] > 0.5
        yc = ys[valid, c]
        pc = probs[valid, c]
        n_pos = int(yc.sum())
        n_tot = int(valid.sum())
        if 0 < n_pos < n_tot:
            auroc = float(roc_auc_score(yc, pc))
            auprc = float(average_precision_score(yc, pc))
            aurocs.append(auroc)
            # Fit 1D Platt scaling on val split ONLY
            eps = 1e-6
            p_clip = np.clip(pc, eps, 1.0 - eps)
            logits_1d = np.log(p_clip / (1.0 - p_clip)).reshape(-1, 1)
            lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
            lr.fit(logits_1d, yc.astype(int))
            calibrators[fid] = {
                "coef": round(float(lr.coef_[0, 0]), 6),
                "intercept": round(float(lr.intercept_[0]), 6),
            }
        else:
            auroc, auprc = 0.0, 0.0
        mean_dice = float(dice_sums[c] / dice_cnts[c]) if dice_cnts[c] > 0 else None
        metrics[fid] = {
            "display_name": CHEST_HD_DISPLAY_NAMES[fid],
            "val_total": n_tot,
            "val_positives": n_pos,
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            "pos_mask_dice": round(mean_dice, 4) if mean_dice is not None else None,
            "pos_mask_cases": int(dice_cnts[c]),
        }
    metrics["_macro_auroc"] = round(float(np.mean(aurocs)), 4) if aurocs else 0.0
    return metrics, calibrators


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ChestHD] Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})", flush=True)

    train_all, val_all = load_all_metadata()
    by_src_tr = {src: sum(1 for s in train_all if s.source == src) for src in ("vinbigdata", "tb_cxr", "siim_pneumo", "rsna_pneumonia")}
    by_src_va = {src: sum(1 for s in val_all if s.source == src) for src in ("vinbigdata", "tb_cxr", "siim_pneumo", "rsna_pneumonia")}
    print(f"[ChestHD] Loaded train={len(train_all)} {by_src_tr} | val={len(val_all)} {by_src_va}", flush=True)

    pos_weight = compute_pos_weights(train_all).to(device)
    print(f"[ChestHD] pos_weight: {[round(float(x), 2) for x in pos_weight.cpu()]}", flush=True)

    val_ds = ChestMultiDataset(val_all, augment=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    net = ChestHDNet(num_classes=NUM_CLASSES, pretrained=True).to(device)
    epochs = 5
    batch_size = 12
    grad_accum = 2
    optimizer = torch.optim.AdamW(net.parameters(), lr=2.5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1.5e-5)

    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    weights_name = "convnext_tiny_fpn_chest16_512.pt"
    weights_path = models_dir / weights_name

    best_macro = -1.0
    best_metrics: dict[str, Any] = {}
    best_cals: dict[str, dict[str, float]] = {}

    for ep in range(1, epochs + 1):
        t0 = time.time()
        ep_items = build_epoch_subset(train_all, ep)
        tr_ds = ChestMultiDataset(ep_items, augment=True)
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

        net.train()
        optimizer.zero_grad(set_to_none=True)
        sum_cls, sum_seg, steps = 0.0, 0.0, 0
        for step, (x, y_cls, m_cls, y_seg, m_seg) in enumerate(tr_loader, 1):
            x = x.to(device, non_blocking=True)
            y_cls = y_cls.to(device, non_blocking=True)
            m_cls = m_cls.to(device, non_blocking=True)
            y_seg = y_seg.to(device, non_blocking=True)
            m_seg = m_seg.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                cls_logits, seg_logits = net(x)
                loss, l_cls, l_seg = masked_loss(cls_logits, seg_logits, y_cls, m_cls, y_seg, m_seg, pos_weight)
                loss = loss / grad_accum
            loss.backward()
            if step % grad_accum == 0 or step == len(tr_loader):
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            sum_cls += l_cls
            sum_seg += l_seg
            steps += 1
            if step == 10 or step % 200 == 0:
                print(
                    f"  [Ep {ep}/{epochs} step {step}/{len(tr_loader)}] cls={sum_cls/steps:.4f} seg={sum_seg/steps:.4f} ({time.time()-t0:.1f}s)",
                    flush=True,
                )

        scheduler.step()
        val_metrics, val_cals = evaluate_val(net, val_loader, device)
        macro = float(val_metrics["_macro_auroc"])
        dt = time.time() - t0
        print(f"[Epoch {ep}/{epochs}] macro_AUROC={macro:.4f} ({dt:.1f}s)", flush=True)
        for fid in ("pneumothorax", "effusion", "cardiomegaly", "pneumonia", "tuberculosis", "nodule_mass", "atelectasis", "aortic_enlargement"):
            m = val_metrics[fid]
            print(f"    {fid:20s} AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} mask_dice={m['pos_mask_dice']}", flush=True)

        if macro > best_macro:
            best_macro = macro
            best_metrics = val_metrics
            best_cals = val_cals
            torch.save(net.state_dict(), weights_path)

    sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    ckpt_meta = {
        "weights_id": "convnext_tiny_fpn_chest16_512",
        "filename": weights_name,
        "sha256": sha,
        "img_size": IMG_SIZE,
        "seg_size": SEG_SIZE,
        "findings": list(CHEST_HD_FINDINGS),
        "best_macro_val_auroc": best_macro,
        "val_metrics": best_metrics,
        "platt_calibrators": best_cals,
        "train_counts_by_source": by_src_tr,
        "val_counts_by_source": by_src_va,
        "training_datasets": [
            "VinBigData Chest X-ray 512x512 (14 radiologist box classes + No finding)",
            "SIIM-ACR Pneumothorax Segmentation (exact pixel masks)",
            "RSNA Pneumonia Detection Challenge (radiologist opacity/pneumonia boxes)",
            "TB Chest Radiography Database (Tuberculosis + Normal)",
        ],
        "license": "Research prototype only — public datasets under respective research terms.",
    }
    (models_dir / CHEST_HD_RECORD).write_text(json.dumps(ckpt_meta, indent=2), encoding="utf-8")
    out_art = ROOT / "artifacts" / "chest_hd_val_metrics.json"
    out_art.parent.mkdir(parents=True, exist_ok=True)
    out_art.write_text(json.dumps(ckpt_meta, indent=2), encoding="utf-8")
    print(f"[ChestHD] Saved {weights_path.name} (sha256={sha[:16]}...) macro_AUROC={best_macro:.4f}", flush=True)


if __name__ == "__main__":
    main()
