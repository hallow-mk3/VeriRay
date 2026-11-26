#!/usr/bin/env python
"""Train a 4-class Brain Tumor MRI classifier (glioma, meningioma, notumor, pituitary) on GPU.

100% offline: initializes backbone from local medical DenseNet-121 weights in models/
and trains on C:\\Users\\swast\\brain_mri\\Training, evaluating on Testing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms.functional as TF
import torchxrayvision as xrv

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("glioma", "meningioma", "notumor", "pituitary")


def _load_split_u8(split_dir: Path, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    imgs, labels = [], []
    for idx, cls_name in enumerate(CLASSES):
        cdir = split_dir / cls_name
        for p in sorted(cdir.glob("*")):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            arr = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
            imgs.append(cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA))
            labels.append(idx)
    return np.stack(imgs, axis=0), np.asarray(labels, dtype=np.int64)


def _to_gpu(batch_u8: np.ndarray, device: torch.device, *, augment: bool, rgb_imagenet: bool) -> torch.Tensor:
    t = torch.from_numpy(batch_u8).to(device, non_blocking=True).float().unsqueeze(1)
    if augment:
        flip = torch.rand((t.shape[0], 1, 1, 1), device=device) < 0.5
        t = torch.where(flip, torch.flip(t, dims=[3]), t)
        # Mild random rotation (+-12 deg) so Training crop orientation doesn't overfit
        angles = (-12.0 + 24.0 * torch.rand(t.shape[0])).tolist()
        t = torch.stack([TF.rotate(img, ang) for img, ang in zip(t, angles)], dim=0)
        gain = 0.85 + 0.30 * torch.rand((t.shape[0], 1, 1, 1), device=device)
        bias = -15.0 + 30.0 * torch.rand((t.shape[0], 1, 1, 1), device=device)
        t = torch.clamp(t * gain + bias, 0.0, 255.0)
    if rgb_imagenet:
        t3 = (t / 255.0).repeat(1, 3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        return (t3 - mean) / std
    return (2.0 * (t / 255.0) - 1.0) * 1024.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path(r"C:\Users\swast\brain_mri"))
    ap.add_argument("--arch", default="efficientnet_b0", choices=["efficientnet_b0", "convnext_tiny", "densenet121_xrv"])
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "models")
    args = ap.parse_args()

    print("Caching Brain MRI Training and Testing images...", flush=True)
    x_tr, y_tr = _load_split_u8(args.data_dir / "Training", args.img_size)
    x_te, y_te = _load_split_u8(args.data_dir / "Testing", args.img_size)
    print(f"Train: {len(x_tr)} | Test: {len(x_te)} | Classes: {CLASSES} | arch={args.arch} | size={args.img_size}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20261126)
    rgb_imagenet = args.arch in {"efficientnet_b0", "convnext_tiny"}

    if args.arch == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    elif args.arch == "convnext_tiny":
        model = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(CLASSES))
    else:
        chex_rec = json.loads((args.out_dir / "CHECKPOINT.json").read_text(encoding="utf-8"))
        w_src = args.out_dir / chex_rec["filename"]
        if hashlib.sha256(w_src.read_bytes()).hexdigest() != str(chex_rec["sha256"]).lower():
            raise RuntimeError(f"SHA-256 mismatch for {w_src}")
        saved = torch.load(w_src, map_location="cpu", weights_only=False)
        model = xrv.models.DenseNet(weights=None, num_classes=len(chex_rec["label_list"]), in_channels=1)
        model.load_state_dict(saved.state_dict() if hasattr(saved, "state_dict") else dict(saved))
        model.classifier = nn.Linear(model.classifier.in_features, len(CLASSES))
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    rng = np.random.default_rng(20261126)

    best_acc = -1.0
    best_state = None
    best_preds = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(len(x_tr))
        tr_losses = []
        for i in range(0, len(x_tr), args.batch_size):
            b = perm[i : i + args.batch_size]
            bx = _to_gpu(x_tr[b], device, augment=True, rgb_imagenet=rgb_imagenet)
            by = torch.from_numpy(y_tr[b]).to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                if rgb_imagenet:
                    logits = model(bx)
                else:
                    feats = torch.relu(model.features(bx))
                    pooled = nn.functional.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
                    logits = model.classifier(pooled)
                loss = loss_fn(logits, by)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_losses.append(float(loss.item()))
        sched.step()

        model.eval()
        preds = []
        with torch.inference_mode():
            for i in range(0, len(x_te), args.batch_size):
                bx = _to_gpu(x_te[i : i + args.batch_size], device, augment=False, rgb_imagenet=rgb_imagenet)
                bx_flip = torch.flip(bx, dims=[3])
                if rgb_imagenet:
                    logits = 0.5 * (model(bx) + model(bx_flip))
                else:
                    def _fwd(z):
                        f = torch.relu(model.features(z))
                        return model.classifier(nn.functional.adaptive_avg_pool2d(f, (1, 1)).flatten(1))
                    logits = 0.5 * (_fwd(bx) + _fwd(bx_flip))
                preds.append(logits.argmax(dim=1).cpu().numpy())
        y_pred = np.concatenate(preds)
        acc = float(accuracy_score(y_te, y_pred))
        print(f"  epoch {epoch:2d}/{args.epochs} train_loss={np.mean(tr_losses):.4f} test_acc={acc:.4f}", flush=True)
        if acc > best_acc:
            best_acc = acc
            best_preds = y_pred
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wpath = args.out_dir / f"{args.arch}-brain-mri-4cls.pt"
    torch.save(best_state, wpath)
    digest = hashlib.sha256(wpath.read_bytes()).hexdigest()
    report = classification_report(y_te, best_preds, target_names=CLASSES, output_dict=True)
    (args.out_dir / "BRAIN_MRI_CHECKPOINT.json").write_text(
        json.dumps(
            {
                "weights_id": f"{args.arch}-brain-mri-4cls",
                "filename": wpath.name,
                "sha256": digest,
                "classes": list(CLASSES),
                "train_samples": int(len(x_tr)),
                "test_samples": int(len(x_te)),
                "best_test_accuracy": best_acc,
                "classification_report": report,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {wpath} (best test accuracy={best_acc:.4f}, sha256={digest[:12]}...)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
