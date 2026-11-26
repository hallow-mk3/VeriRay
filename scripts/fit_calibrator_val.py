#!/usr/bin/env python
"""Fit PlattCalibrator on NIH validation split ONLY and save to models/platt_calibrator.json."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.metrics import compute_reader_metrics  # noqa: E402
from eval.run_eval import _index_images  # noqa: E402
from scripts.val_diagnose_and_tune import _load_nih_trained_reader  # noqa: E402
from veriray.calibration import fit_platt_calibrator  # noqa: E402
from veriray.splits import load_manifest, select_split  # noqa: E402


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reader = _load_nih_trained_reader(device)
    rows = load_manifest(ROOT / "data" / "manifests" / "nih_manifest.csv")
    val_rows = select_split(rows, "val")
    index = _index_images(Path(r"C:\Users\swast\nih"))

    pos = [r for r in val_rows if r["pneumothorax"] == 1 and r["image"] in index]
    neg = [r for r in val_rows if r["pneumothorax"] == 0 and r["image"] in index]
    rng = np.random.default_rng(20261126)
    rng.shuffle(pos)
    rng.shuffle(neg)
    subset = pos[:200] + neg[:200]

    scores, y_true = [], []
    for r in subset:
        arr = np.asarray(Image.open(index[r["image"]]).convert("L"), dtype=np.float32)
        rc, _, _ = reader.score(arr)
        scores.append(rc.score_raw)
        y_true.append(int(r["pneumothorax"]))

    cal = fit_platt_calibrator(scores, y_true, split_name="val")
    out_path = cal.save(ROOT / "models" / "platt_calibrator.json")
    cal_scores = [cal.calibrate(s) for s in scores]

    raw_m = compute_reader_metrics(y_true, scores)
    cal_m = compute_reader_metrics(y_true, cal_scores)
    print(f"Saved {out_path} (n={len(y_true)}, coef={cal.coef:.4f}, intercept={cal.intercept:.4f})", flush=True)
    print(f"Raw val metrics:        {raw_m}", flush=True)
    print(f"Calibrated val metrics: {cal_m}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
