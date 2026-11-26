#!/usr/bin/env python
"""Train the local Gated-Convolution U-Net Eraser on NIH 'train' split 'No Finding' images.

    python scripts/train_unet_eraser.py --manifest data/manifests/nih_manifest.csv --images-dir D:\\nih\\images
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
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veriray.splits import load_manifest, select_split  # noqa: E402
from veriray.unet_eraser import UNET_RECORD, GatedUNet  # noqa: E402


def _random_mask(rng: np.random.Generator, size: int = 224) -> np.ndarray:
    m = np.zeros((size, size), dtype=np.float32)
    cy, cx = rng.integers(32, size - 32, size=2)
    ry, rx = rng.integers(12, 42, size=2)
    yy, xx = np.ogrid[:size, :size]
    m[((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0] = 1.0
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max-images", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "models")
    args = ap.parse_args()

    if not args.manifest.is_file():
        print(f"Manifest not found: {args.manifest}. Run scripts/build_nih_manifest.py after downloading NIH data.", file=sys.stderr)
        return 2

    rows = load_manifest(args.manifest)
    train_rows = [r for r in select_split(rows, "train") if r["no_finding"] == 1][: args.max_images]
    if not train_rows:
        print("No 'No Finding' rows found in train split.", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20261126)
    rng = np.random.default_rng(20261126)
    model = GatedUNet(base_ch=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    index = {p.name: p for p in args.images_dir.glob("*.png")}
    if not index:
        index = {p.name: p for p in args.images_dir.glob("images_*/images/*.png")}
    if not index:
        index = {p.name: p for p in args.images_dir.rglob("*.png")}
    paths = [index[r["image"]] for r in train_rows if r["image"] in index]
    print(f"Loading and resizing {len(paths)} train 'No Finding' images to 224x224...", flush=True)
    cached_imgs = []
    for p in paths:
        arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        cached_imgs.append(cv2.resize(arr, (224, 224), interpolation=cv2.INTER_AREA))
    cached_arr = np.stack(cached_imgs, axis=0)
    print(f"Training GatedUNet on {len(cached_arr)} cached 224x224 images on {device}...", flush=True)

    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(len(cached_arr))
        losses = []
        for i in range(0, len(cached_arr), args.batch_size):
            batch_idx = perm[i : i + args.batch_size]
            imgs = cached_arr[batch_idx]
            masks = np.stack([_random_mask(rng, 224) for _ in range(len(batch_idx))])
            t_img = torch.from_numpy(imgs)[:, None].to(device)
            t_msk = torch.from_numpy(masks)[:, None].to(device)
            pred = model(t_img * (1.0 - t_msk), t_msk)
            # Weighted reconstruction loss: emphasize masked region + smooth outside
            loss = 6.0 * F.l1_loss(pred * t_msk, t_img * t_msk) + F.l1_loss(pred * (1.0 - t_msk), t_img * (1.0 - t_msk))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"  epoch {epoch}/{args.epochs} loss={np.mean(losses):.4f}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wpath = args.out_dir / "gated-unet-nofinding-224.pt"
    torch.save(model.state_dict(), wpath)
    digest = hashlib.sha256(wpath.read_bytes()).hexdigest()
    (args.out_dir / UNET_RECORD).write_text(
        json.dumps(
            {
                "filename": wpath.name,
                "sha256": digest,
                "base_ch": 32,
                "trained_on": "NIH ChestX-ray14 train split 'No Finding' only (weak label, not certified healthy)",
                "num_images": len(paths),
                "epochs": args.epochs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {wpath} (sha256={digest[:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
