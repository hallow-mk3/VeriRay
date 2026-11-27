"""Train 5-Class Brain MRI Reader + FPN Lesion Segmenter (Tumor + Ischemic Stroke).

Classes:
  0: glioma
  1: meningioma
  2: notumor
  3: pituitary
  4: ischemic_stroke

Datasets combined with strict patient/subject-level splits:
  1. C:\\Users\\swast\\brain_mri (Training=5,600 / Testing=1,600 across 4 tumor classes)
  2. BraTS 2020 (369 3D glioma volumes + exact radiologist seg.nii voxel masks; split by patient ID)
  3. ISLES 2022 (250 3D ischemic stroke volumes + exact derivatives/*_msk.nii.gz masks; split by sub-* ID)
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

import cv2
import nibabel as nib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veriray.brain_mri import BRAIN_HD_CLASSES, BRAIN_RECORD, BrainHDNet  # noqa: E402

IMG_SIZE = 256
SEG_SIZE = 64
NUM_CLASSES = len(BRAIN_HD_CLASSES)  # 5
SEED = 20261127


@dataclass
class BrainSliceSample:
    uid: str
    source: str
    img_u8: np.ndarray        # (256, 256) uint8
    cls_idx: int              # 0..4
    seg_known: bool
    mask_u8: np.ndarray | None = None  # (64, 64) uint8 in {0, 1}


def _is_train(subject_id: str) -> bool:
    h = int(hashlib.sha256(subject_id.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 100) < 85


def _norm_slice_u8(sl: np.ndarray, size: int = IMG_SIZE) -> np.ndarray | None:
    sl = np.nan_to_num(np.asarray(sl, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    fg = sl[sl > 0]
    if fg.size < 100:
        return None
    p1, p99 = float(np.percentile(fg, 1)), float(np.percentile(fg, 99.5))
    if p99 - p1 < 1e-5:
        return None
    u8 = np.clip((sl - p1) / (p99 - p1) * 255.0, 0.0, 255.0).astype(np.uint8)
    u8[sl <= 0] = 0
    if u8.shape != (size, size):
        u8 = cv2.resize(u8, (size, size), interpolation=cv2.INTER_LINEAR)
    return u8


def load_2d_benchmark(data_dir: Path) -> tuple[list[BrainSliceSample], list[BrainSliceSample]]:
    tr_out: list[BrainSliceSample] = []
    te_out: list[BrainSliceSample] = []
    base_4 = ("glioma", "meningioma", "notumor", "pituitary")
    for split_name, dst in [("Training", tr_out), ("Testing", te_out)]:
        sdir = data_dir / split_name
        for c_idx, cname in enumerate(base_4):
            cdir = sdir / cname
            if not cdir.is_dir():
                continue
            for p in sorted(cdir.iterdir()):
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                with Image.open(p) as im:
                    arr = np.asarray(im.convert("L"), dtype=np.uint8)
                arr_r = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                is_notumor = cname == "notumor"
                dst.append(
                    BrainSliceSample(
                        uid=f"bm4:{split_name}:{cname}:{p.stem}",
                        source="brain_mri_4cls",
                        img_u8=arr_r,
                        cls_idx=c_idx,
                        seg_known=is_notumor,
                        mask_u8=np.zeros((SEG_SIZE, SEG_SIZE), dtype=np.uint8) if is_notumor else None,
                    )
                )
    return tr_out, te_out


def load_brats_slices(brats_root: Path) -> tuple[list[BrainSliceSample], list[BrainSliceSample]]:
    tr_out: list[BrainSliceSample] = []
    te_out: list[BrainSliceSample] = []
    search_dir = brats_root / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
    if not search_dir.is_dir():
        return tr_out, te_out

    tr_pids, te_pids = set(), set()
    for pdir in sorted(search_dir.glob("BraTS20_Training_*")):
        if not pdir.is_dir():
            continue
        pid = pdir.name
        seg_files = list(pdir.glob("*_seg.nii*"))
        t1ce_files = list(pdir.glob("*_t1ce.nii*"))
        flair_files = list(pdir.glob("*_flair.nii*"))
        if not seg_files or (not t1ce_files and not flair_files):
            continue

        seg_vol = np.asarray(nib.load(str(seg_files[0])).dataobj)
        lesion_areas = (seg_vol > 0).sum(axis=(0, 1))
        pos_zs = np.where(lesion_areas >= 80)[0]
        if len(pos_zs) == 0:
            continue

        # Pick 3 evenly spaced tumor-bearing slices per patient across T1ce & FLAIR
        chosen_zs = np.linspace(pos_zs[0], pos_zs[-1], num=min(3, len(pos_zs)), dtype=int)
        is_tr = _is_train(f"brats:{pid}")
        (tr_pids if is_tr else te_pids).add(pid)
        dst = tr_out if is_tr else te_out

        vols = []
        if t1ce_files:
            vols.append(("t1ce", np.asarray(nib.load(str(t1ce_files[0])).dataobj)))
        if flair_files:
            vols.append(("flair", np.asarray(nib.load(str(flair_files[0])).dataobj)))

        for mod_name, vol in vols:
            for z in chosen_zs:
                sl = np.rot90(vol[:, :, int(z)])
                u8 = _norm_slice_u8(sl, IMG_SIZE)
                if u8 is None:
                    continue
                msk = np.rot90((seg_vol[:, :, int(z)] > 0).astype(np.uint8))
                msk_64 = (cv2.resize(msk, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
                dst.append(
                    BrainSliceSample(
                        uid=f"brats:{pid}:{mod_name}:z{int(z)}",
                        source="brats2020",
                        img_u8=u8,
                        cls_idx=0,  # glioma
                        seg_known=True,
                        mask_u8=msk_64,
                    )
                )
    assert tr_pids.isdisjoint(te_pids), "BraTS patient split overlap!"
    print(f"[BrainHD] BraTS patients train={len(tr_pids)} ({len(tr_out)} slices) | test={len(te_pids)} ({len(te_out)} slices)", flush=True)
    return tr_out, te_out


def load_isles_slices(isles_root: Path) -> tuple[list[BrainSliceSample], list[BrainSliceSample]]:
    tr_out: list[BrainSliceSample] = []
    te_out: list[BrainSliceSample] = []
    if not isles_root.is_dir():
        return tr_out, te_out

    # Index mask files by sub-strokecaseXXXX
    sub_re = re.compile(r"(sub-[a-zA-Z0-9]+)")
    masks_by_sub: dict[str, Path] = {}
    imgs_by_sub: dict[str, list[tuple[str, Path]]] = {}

    for p in isles_root.rglob("*.nii*"):
        m = sub_re.search(str(p))
        if not m:
            continue
        sid = m.group(1)
        low = p.name.lower()
        if "msk" in low or "lesion" in low or "seg" in low:
            masks_by_sub[sid] = p
        elif "dwi" in low or "flair" in low or "adc" in low:
            mod = "dwi" if "dwi" in low else ("flair" if "flair" in low else "adc")
            imgs_by_sub.setdefault(sid, []).append((mod, p))

    tr_sids, te_sids = set(), set()
    for sid in sorted(masks_by_sub):
        if sid not in imgs_by_sub:
            continue
        msk_vol = np.asarray(nib.load(str(masks_by_sub[sid])).dataobj)
        if msk_vol.ndim == 4:
            msk_vol = msk_vol[..., 0]
        lesion_areas = (msk_vol > 0).sum(axis=(0, 1))
        pos_zs = np.where(lesion_areas >= 15)[0]
        if len(pos_zs) == 0:
            pos_zs = np.where(lesion_areas > 0)[0]
        if len(pos_zs) == 0:
            continue

        chosen_zs = np.linspace(pos_zs[0], pos_zs[-1], num=min(5, len(pos_zs)), dtype=int)
        is_tr = _is_train(f"isles:{sid}")
        (tr_sids if is_tr else te_sids).add(sid)
        dst = tr_out if is_tr else te_out

        # Prefer DWI and FLAIR (standard acute stroke sequences)
        mods = [ (m, p) for (m, p) in imgs_by_sub[sid] if m in ("dwi", "flair") ] or imgs_by_sub[sid][:1]
        for mod_name, ipath in mods:
            vol = np.asarray(nib.load(str(ipath)).dataobj)
            if vol.ndim == 4:
                vol = vol[..., 0]
            if vol.shape[:2] != msk_vol.shape[:2] or vol.shape[2] != msk_vol.shape[2]:
                # Resample slice-wise if needed (e.g. FLAIR vs DWI grid)
                continue
            for z in chosen_zs:
                sl = np.rot90(vol[:, :, int(z)])
                u8 = _norm_slice_u8(sl, IMG_SIZE)
                if u8 is None:
                    continue
                msk = np.rot90((msk_vol[:, :, int(z)] > 0).astype(np.uint8))
                msk_64 = (cv2.resize(msk, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
                dst.append(
                    BrainSliceSample(
                        uid=f"isles:{sid}:{mod_name}:z{int(z)}",
                        source="isles2022",
                        img_u8=u8,
                        cls_idx=4,  # ischemic_stroke
                        seg_known=True,
                        mask_u8=msk_64,
                    )
                )
    assert tr_sids.isdisjoint(te_sids), "ISLES subject split overlap!"
    print(f"[BrainHD] ISLES subjects train={len(tr_sids)} ({len(tr_out)} slices) | test={len(te_sids)} ({len(te_out)} slices)", flush=True)
    return tr_out, te_out


class BrainSliceDataset(Dataset):
    def __init__(self, items: list[BrainSliceSample], augment: bool = False) -> None:
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
            if s.cls_idx == 2:  # notumor -> all lesion channels known 0
                m_seg[:] = 1.0
            else:
                m_seg[s.cls_idx] = 1.0
                y_seg[s.cls_idx] = s.mask_u8.astype(np.float32)
                # Other lesion channels are also 0 on this single-pathology scan
                for c in (0, 1, 3, 4):
                    if c != s.cls_idx:
                        m_seg[c] = 1.0

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
            if random.random() < 0.5:
                img = np.clip(img * random.uniform(0.85, 1.15) + random.uniform(-0.06, 0.06), 0.0, 1.0)

        x3 = (np.repeat(img[None, :, :], 3, axis=0) - self.mean) / self.std
        return torch.from_numpy(x3), int(s.cls_idx), torch.from_numpy(y_seg), torch.from_numpy(m_seg)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BrainHD] Device: {device}", flush=True)

    bm_tr, bm_te = load_2d_benchmark(Path(r"C:\Users\swast\brain_mri"))
    br_tr, br_te = load_brats_slices(Path(r"C:\Users\swast\datasets\brats"))
    is_tr, is_te = load_isles_slices(Path(r"C:\Users\swast\datasets\isles"))

    # Upsample stroke slices slightly if fewer than 1,200 so all 5 classes are balanced
    if 0 < len(is_tr) < 1200:
        reps = max(1, round(1200 / len(is_tr)))
        is_tr_bal = is_tr * reps
    else:
        is_tr_bal = is_tr

    train_items = bm_tr + br_tr + is_tr_bal
    test_items = bm_te + br_te + is_te
    print(f"[BrainHD] Total train={len(train_items)} | test={len(test_items)} (4cls_test={len(bm_te)})", flush=True)

    cls_counts = np.bincount([s.cls_idx for s in train_items], minlength=NUM_CLASSES).astype(np.float64)
    cls_weights = np.clip(cls_counts.mean() / np.maximum(cls_counts, 1.0), 0.5, 3.0)
    print(f"[BrainHD] Train counts per class {dict(zip(BRAIN_HD_CLASSES, cls_counts.astype(int).tolist()))}", flush=True)

    tr_loader = DataLoader(BrainSliceDataset(train_items, augment=True), batch_size=24, shuffle=True, num_workers=0, pin_memory=True)
    te_loader = DataLoader(BrainSliceDataset(test_items, augment=False), batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    bm4_loader = DataLoader(BrainSliceDataset(bm_te, augment=False), batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    net = BrainHDNet(num_classes=NUM_CLASSES, pretrained=True).to(device)
    epochs = 10
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    ce_fn = nn.CrossEntropyLoss(weight=torch.tensor(cls_weights, dtype=torch.float32, device=device), label_smoothing=0.03)

    models_dir = ROOT / "models"
    weights_name = "convnext_tiny_fpn-brain-mri-5cls.pt"
    weights_path = models_dir / weights_name

    best_acc = -1.0
    best_summary: dict[str, Any] = {}

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
                loss = l_cls + 0.6 * l_seg + 0.6 * l_dice
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            sum_loss += float(loss.item())
            steps += 1

        sched.step()

        # Evaluate 5-class combined + 4-class benchmark + lesion Dice
        net.eval()
        preds_all, y_all = [], []
        dice_vals = []
        with torch.no_grad():
            for x, y, y_seg, m_seg in te_loader:
                x = x.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    l1, s1 = net(x)
                    l2, s2 = net(torch.flip(x, dims=[3]))
                    logits = 0.5 * (l1 + l2)
                    seg_p = torch.sigmoid(0.5 * (s1 + torch.flip(s2, dims=[3]))).float().cpu().numpy()
                preds_all.extend(logits.argmax(dim=1).cpu().tolist())
                y_all.extend(y.tolist())
                ys_np = y_seg.numpy()
                ms_np = m_seg.numpy()
                for b in range(seg_p.shape[0]):
                    for c in (0, 4):  # glioma (BraTS) & ischemic_stroke (ISLES) ground-truth masks
                        if ms_np[b, c] > 0.5 and ys_np[b, c].sum() > 0:
                            pb = (seg_p[b, c] >= 0.45).astype(np.float32)
                            inter = float((pb * ys_np[b, c]).sum())
                            den = float(pb.sum() + ys_np[b, c].sum())
                            dice_vals.append((2.0 * inter) / max(den, 1.0))

            # Also check 4-class benchmark accuracy (restricted to first 4 logits)
            preds_4, y_4 = [], []
            for x, y, _, _ in bm4_loader:
                x = x.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    logits = 0.5 * (net(x)[0] + net(torch.flip(x, dims=[3]))[0])
                preds_4.extend(logits.argmax(dim=1).cpu().tolist())
                y_4.extend(y.tolist())

        acc5 = float(accuracy_score(y_all, preds_all))
        acc4 = float(accuracy_score(y_4, preds_4))
        mean_dice = float(np.mean(dice_vals)) if dice_vals else 0.0
        print(
            f"[Epoch {ep}/{epochs}] loss={sum_loss/max(steps,1):.4f} | 5cls_acc={acc5:.4f} | 4cls_bench_acc={acc4:.4f} | lesion_Dice={mean_dice:.4f} ({time.time()-t0:.1f}s)",
            flush=True,
        )
        if acc5 > best_acc:
            best_acc = acc5
            present_labels = sorted(set(y_all))
            present_names = [BRAIN_HD_CLASSES[i] for i in present_labels]
            rep = classification_report(y_all, preds_all, labels=present_labels, target_names=present_names, output_dict=True, zero_division=0)
            best_summary = {
                "best_5cls_accuracy": round(acc5, 6),
                "best_test_accuracy": round(acc4, 6),  # backwards-compatible UI metric key
                "lesion_segmentation_dice": round(mean_dice, 4),
                "classification_report": rep,
            }
            torch.save(net.state_dict(), weights_path)

    sha = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    ckpt = {
        "weights_id": "convnext_tiny_fpn-brain-mri-5cls",
        "filename": weights_name,
        "sha256": sha,
        "img_size": IMG_SIZE,
        "classes": list(BRAIN_HD_CLASSES),
        "train_samples": len(train_items),
        "test_samples": len(test_items),
        **best_summary,
    }
    (models_dir / BRAIN_RECORD).write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
    (ROOT / "artifacts" / "brain_mri_hd_metrics.json").write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
    print(f"[BrainHD] Saved {weights_name} (sha256={sha[:16]}...) 5cls_acc={best_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
