#!/usr/bin/env python
"""Fine-tune DenseNet-121 for Pneumothorax on NIH ChestX-ray14 'train' split ONLY (GPU).

Strict leakage control:
- Uses ONLY ``select_split(rows, "train")`` for training (up to all 69,302 train images).
- Evaluates every epoch on ``select_split(rows, "val")``.
- Never touches the locked ``test`` split.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
import torchxrayvision as xrv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veriray.splits import load_manifest, select_split  # noqa: E402


def _index_images(images_dir: Path) -> dict[str, Path]:
    idx = {p.name: p for p in images_dir.glob("*.png")}
    if not idx:
        idx = {p.name: p for p in images_dir.glob("images_*/images/*.png")}
    if not idx:
        idx = {p.name: p for p in images_dir.rglob("*.png")}
    return idx


def _read_one(args: tuple[Path, float]) -> tuple[np.ndarray, float]:
    path, label = args
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    resized = cv2.resize(arr, (224, 224), interpolation=cv2.INTER_AREA)
    return resized, label


def _load_cached_u8(
    rows: list[dict],
    index: dict[str, Path],
    *,
    all_rows: bool,
    max_per_class: int,
    seed: int,
    workers: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if all_rows:
        chosen = [r for r in rows if r["image"] in index]
    else:
        pos = [r for r in rows if r["pneumothorax"] == 1 and r["image"] in index]
        neg = [r for r in rows if r["pneumothorax"] == 0 and r["image"] in index]
        rng.shuffle(pos)
        rng.shuffle(neg)
        chosen = pos[:max_per_class] + neg[:max_per_class]
    rng.shuffle(chosen)

    tasks = [(index[r["image"]], float(r["pneumothorax"])) for r in chosen]
    imgs = np.empty((len(tasks), 224, 224), dtype=np.uint8)
    labels = np.empty((len(tasks),), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, (im, lbl) in enumerate(pool.map(_read_one, tasks, chunksize=64)):
            imgs[idx] = im
            labels[idx] = lbl
            if (idx + 1) % 10000 == 0:
                print(f"  cached {idx + 1}/{len(tasks)} images...", flush=True)
    return imgs, labels


def _to_xrv_gpu(batch_u8: np.ndarray, device: torch.device, *, augment: bool = False) -> torch.Tensor:
    t = torch.from_numpy(batch_u8).to(device, non_blocking=True).float().unsqueeze(1)
    if augment:
        # Random horizontal flip per image + mild brightness/contrast jitter
        flip = torch.rand((t.shape[0], 1, 1, 1), device=device) < 0.5
        t = torch.where(flip, torch.flip(t, dims=[3]), t)
        gain = 0.9 + 0.2 * torch.rand((t.shape[0], 1, 1, 1), device=device)
        bias = -12.0 + 24.0 * torch.rand((t.shape[0], 1, 1, 1), device=device)
        t = torch.clamp(t * gain + bias, 0.0, 255.0)
    return (2.0 * (t / 255.0) - 1.0) * 1024.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "nih_manifest.csv")
    ap.add_argument("--images-dir", type=Path, default=Path(r"C:\Users\swast\nih"))
    ap.add_argument("--all-train", action="store_true", help="train on ALL train-split images (69,302)")
    ap.add_argument("--max-per-class-train", type=int, default=1200)
    ap.add_argument("--max-per-class-val", type=int, default=442)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--resume-nih", action="store_true", help="resume from models/densenet121-nih-train-ptx.pt")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "models")
    args = ap.parse_args()

    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    rows = load_manifest(args.manifest)
    index = _index_images(args.images_dir)
    train_rows = select_split(rows, "train")
    val_rows = select_split(rows, "val")

    print("Caching train and val 224x224 uint8 tensors...", flush=True)
    x_tr, y_tr = _load_cached_u8(
        train_rows, index, all_rows=args.all_train, max_per_class=args.max_per_class_train, seed=20261126
    )
    x_va, y_va = _load_cached_u8(
        val_rows, index, all_rows=False, max_per_class=args.max_per_class_val, seed=20261127
    )
    n_pos = int(y_tr.sum())
    n_neg = int(len(y_tr) - n_pos)
    print(f"Train: {len(x_tr)} ({n_pos} pos, {n_neg} neg) | Val: {len(x_va)} ({int(y_va.sum())} pos)", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20261126)

    chex_rec = json.loads((args.out_dir / "CHECKPOINT.json").read_text(encoding="utf-8"))
    model = xrv.models.DenseNet(weights=None, num_classes=len(chex_rec["label_list"]), in_channels=1)
    nih_w = args.out_dir / "densenet121-nih-train-ptx.pt"
    if args.resume_nih and nih_w.is_file():
        print(f"Resuming from {nih_w.name}...", flush=True)
        model.load_state_dict(torch.load(nih_w, map_location="cpu", weights_only=True))
    else:
        chex_w = args.out_dir / chex_rec["filename"]
        if hashlib.sha256(chex_w.read_bytes()).hexdigest() != str(chex_rec["sha256"]).lower():
            raise RuntimeError(f"SHA-256 mismatch for {chex_w}")
        saved = torch.load(chex_w, map_location="cpu", weights_only=False)
        model.load_state_dict(saved.state_dict() if hasattr(saved, "state_dict") else dict(saved))
    model.to(device)

    ptx_idx = int(chex_rec["finding_index"])
    pos_weight = torch.tensor([min(15.0, max(1.0, n_neg / float(max(1, n_pos))))], device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    rng = np.random.default_rng(20261126)

    best_auc = -1.0
    best_state = None
    best_auprc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(len(x_tr))
        tr_losses = []
        for i in range(0, len(x_tr), args.batch_size):
            b = perm[i : i + args.batch_size]
            bx = _to_xrv_gpu(x_tr[b], device, augment=True)
            by = torch.from_numpy(y_tr[b]).to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                feats = torch.relu(model.features(bx))
                pooled = nn.functional.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
                logits = model.classifier(pooled)[:, ptx_idx]
                loss = loss_fn(logits, by)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_losses.append(float(loss.item()))
        sched.step()

        model.eval()
        val_probs = []
        with torch.inference_mode():
            for i in range(0, len(x_va), args.batch_size):
                bx = _to_xrv_gpu(x_va[i : i + args.batch_size], device)
                feats = torch.relu(model.features(bx))
                pooled = nn.functional.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
                logits = model.classifier(pooled)[:, ptx_idx]
                val_probs.append(torch.sigmoid(logits).cpu().numpy())
        probs = np.concatenate(val_probs)
        auc = float(roc_auc_score(y_va, probs))
        auprc = float(average_precision_score(y_va, probs))
        print(f"  epoch {epoch}/{args.epochs} train_loss={np.mean(tr_losses):.4f} val_auroc={auc:.4f} val_auprc={auprc:.4f}", flush=True)
        if auc > best_auc:
            best_auc = auc
            best_auprc = auprc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wpath = args.out_dir / "densenet121-nih-train-ptx.pt"
    torch.save(best_state, wpath)
    digest = hashlib.sha256(wpath.read_bytes()).hexdigest()
    (args.out_dir / "NIH_TRAINED_CHECKPOINT.json").write_text(
        json.dumps(
            {
                "weights_id": "densenet121-nih-train-ptx",
                "filename": wpath.name,
                "sha256": digest,
                "initialized_from": chex_rec["weights_id"],
                "trained_split": "NIH ChestX-ray14 train split ONLY (weak report-mined labels)",
                "train_samples": int(len(x_tr)),
                "val_samples": int(len(x_va)),
                "best_val_auroc": best_auc,
                "best_val_auprc": best_auprc,
                "label_list": chex_rec["label_list"],
                "finding_label": "Pneumothorax",
                "finding_index": ptx_idx,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {wpath} (best val AUROC={best_auc:.4f}, AUPRC={best_auprc:.4f}, sha256={digest[:12]}...)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
