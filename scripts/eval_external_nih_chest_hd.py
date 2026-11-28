#!/usr/bin/env python
"""External Evaluation Script: NIH ChestX-ray14 -> 15-Finding De-Confounded ChestHD.

Evaluates ``models/CHEST_HD_CHECKPOINT.json`` (trained on cluster-grouped VinBigData + SIIM + RSNA,
with zero NIH ChestX-ray14 training exposure and TB-CXR dropped) as a true zero-shot external
evaluation on NIH ChestX-ray14.

Split safety:
  - Defaults to ``--split val`` from ``data/manifests/nih_manifest.csv`` (via ``veriray.splits``).
  - Locked ``test`` split requires explicit ``--split test --unlock-test`` flag.
  - Computes patient-level bootstrap 95% CIs for AUROC and AUPRC, plus localization IoU and
    Pointing Game hit rate on NIH ``BBox_List_2017.csv`` cases for mapped findings.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veriray.chest_hd import CHEST_HD_DISPLAY_NAMES, CHEST_HD_FINDINGS, CHEST_HD_RECORD, ChestHDNet, preprocess_hd  # noqa: E402
from veriray.splits import load_manifest, select_split  # noqa: E402

# Clean ontological mapping: ChestHD finding -> tuple of NIH ChestX-ray14 Finding Labels (OR-combined)
NIH_TO_CHEST_HD_MAP: dict[str, tuple[str, ...]] = {
    "atelectasis": ("Atelectasis",),
    "cardiomegaly": ("Cardiomegaly",),
    "effusion": ("Effusion",),
    "infiltration": ("Infiltration",),
    "nodule_mass": ("Mass", "Nodule"),  # VinBigData 'Nodule/Mass' = union of NIH 'Mass' | 'Nodule'
    "pneumonia": ("Pneumonia",),
    "pneumothorax": ("Pneumothorax",),
    "consolidation": ("Consolidation",),
    "pleural_thickening": ("Pleural_Thickening",),
    "pulmonary_fibrosis": ("Fibrosis",),
}

# Explicitly documented unmapped labels (never force incompatible definitions):
UNMAPPED_NIH_LABELS: dict[str, str] = {
    "Edema": "No dedicated pulmonary edema head in ChestHD.",
    "Emphysema": "No emphysema / hyperinflation head in ChestHD.",
    "Hernia": "No hiatal/diaphragmatic hernia head in ChestHD.",
}

UNMAPPED_CHEST_HD_FINDINGS: dict[str, str] = {
    "aortic_enlargement": "Not annotated in NIH ChestX-ray14.",
    "calcification": "Not annotated in NIH ChestX-ray14.",
    "ild": "Broader interstitial lung disease category than NIH 'Fibrosis' alone.",
    "other_lesion": "Residual catch-all category in VinBigData with no NIH counterpart.",
    "lung_opacity": "Super-class umbrella term not directly annotated as a standalone NIH label.",
}

# NIH BBox_List_2017.csv Finding Label -> ChestHD finding
NIH_BBOX_TO_CHEST_HD: dict[str, str] = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Effusion": "effusion",
    "Infiltrate": "infiltration",
    "Mass": "nodule_mass",
    "Nodule": "nodule_mass",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
}


def _index_nih_images(nih_dir: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for p in nih_dir.rglob("*.png"):
        idx[p.name] = p
    return idx


def _patient_bootstrap_ci(
    pids: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_boot: int = 500,
    seed: int = 20261127,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    uniq_p = np.unique(pids)
    p_to_idx = {p: np.where(pids == p)[0] for p in uniq_p}
    vals: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq_p, size=len(uniq_p), replace=True)
        idxs = np.concatenate([p_to_idx[p] for p in sampled])
        yb, sb = y_true[idxs], y_score[idxs]
        if 0 < yb.sum() < len(yb):
            vals.append(float(metric_fn(yb, sb)))
    if not vals:
        return (0.0, 0.0)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


class _NIHExternalDataset(Dataset):
    def __init__(self, items: list[tuple[str, str, Path]], img_size: int = 512) -> None:
        self.items = items
        self.img_size = img_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        im_name, pid, p = self.items[idx]
        with Image.open(p) as im:
            w0, h0 = im.size
            u8 = np.asarray(im.convert("L"), dtype=np.uint8)
        if u8.shape != (self.img_size, self.img_size):
            interp = cv2.INTER_AREA if (u8.shape[0] >= self.img_size and u8.shape[1] >= self.img_size) else cv2.INTER_LINEAR
            u8 = cv2.resize(u8, (self.img_size, self.img_size), interpolation=interp)
        x01 = u8.astype(np.float32) / 255.0
        x3 = (np.repeat(x01[None, :, :], 3, axis=0) - self.mean) / self.std
        return torch.from_numpy(x3), im_name, pid, int(w0), int(h0)


def _load_nih_bboxes_for_images(bbox_csv: Path, target_imgs: set[str]) -> dict[tuple[str, str], list[tuple[float, float, float, float]]]:
    """Return {(image_name, chest_hd_finding): [(x1_norm, y1_norm, x2_norm, y2_norm), ...]} on 1024x1024 NIH space."""
    out: dict[tuple[str, str], list[tuple[float, float, float, float]]] = defaultdict(list)
    if not bbox_csv.is_file():
        return out
    with bbox_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            im_name = row["Image Index"].strip()
            if im_name not in target_imgs:
                continue
            lbl = row["Finding Label"].strip()
            chd = NIH_BBOX_TO_CHEST_HD.get(lbl)
            if chd is None:
                continue
            x = float(row["Bbox [x"])
            y = float(row["y"])
            w = float(row["w"])
            h = float(row["h]"])
            out[(im_name, chd)].append((x, y, x + w, y + h))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="External NIH ChestX-ray14 evaluation for 15-finding ChestHD.")
    ap.add_argument("--nih-dir", type=Path, default=Path(r"C:\Users\swast\nih"))
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "nih_manifest.csv")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--unlock-test", action="store_true", help="Required if --split test is requested.")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "external_nih_eval.json")
    args = ap.parse_args()

    if args.split == "test" and not args.unlock_test:
        raise SystemExit("Refusing to read locked test split without explicit --unlock-test.")

    rows = load_manifest(args.manifest)
    split_rows = select_split(rows, args.split, allow_test=args.unlock_test)

    # Read NIH Finding Labels ONLY for images in the requested split
    target_imgs = {r["image"] for r in split_rows}
    labels_by_img: dict[str, set[str]] = {}
    with (args.nih_dir / "Data_Entry_2017.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            im_name = row["Image Index"].strip()
            if im_name in target_imgs:
                labels_by_img[im_name] = {x.strip() for x in row["Finding Labels"].split("|") if x.strip()}

    img_paths = _index_nih_images(args.nih_dir)
    bbox_map = _load_nih_bboxes_for_images(args.nih_dir / "BBox_List_2017.csv", target_imgs)
    bbox_img_set = {im_n for (im_n, _) in bbox_map}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rec = json.loads((args.models_dir / CHEST_HD_RECORD).read_text(encoding="utf-8"))
    findings = tuple(rec["findings"])
    img_sz = int(rec.get("img_size", 512))
    net = ChestHDNet(num_classes=len(findings), pretrained=False).to(device)
    net.load_state_dict(torch.load(args.models_dir / rec["filename"], map_location="cpu", weights_only=True))
    net.eval()

    valid_items = [
        (r["image"], str(r["patient_id"]), img_paths[r["image"]])
        for r in split_rows
        if r["image"] in img_paths and r["image"] in labels_by_img
    ]
    loader = DataLoader(_NIHExternalDataset(valid_items, img_size=img_sz), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    pids_list: list[str] = []
    scores_chunks: list[np.ndarray] = []
    y_mapped: dict[str, list[int]] = {k: [] for k in NIH_TO_CHEST_HD_MAP}
    bbox_ious: dict[str, list[float]] = defaultdict(list)
    bbox_hits: dict[str, list[int]] = defaultdict(list)

    for xb, im_names, pids_b, w0s, h0s in loader:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            cls_l, seg_l = net(xb.to(device, non_blocking=True))
            cls_p = torch.sigmoid(cls_l).float()
            seg_p = torch.sigmoid(seg_l).float()
            top_k = max(1, int(0.02 * seg_p.shape[2] * seg_p.shape[3]))
            seg_pk = torch.topk(seg_p.flatten(2), top_k, dim=2).values.mean(dim=2)
            fused = (0.80 * cls_p + 0.20 * seg_pk).cpu().numpy()

        scores_chunks.append(fused)
        for bi, im_name in enumerate(im_names):
            pids_list.append(pids_b[bi])
            nih_lbls = labels_by_img[im_name]
            for chd_fid, nih_set in NIH_TO_CHEST_HD_MAP.items():
                y_mapped[chd_fid].append(1 if any(nl in nih_lbls for nl in nih_set) else 0)

            if im_name in bbox_img_set:
                w0 = float(w0s[bi].item()) if float(w0s[bi].item()) > 0 else 1024.0
                h0 = float(h0s[bi].item()) if float(h0s[bi].item()) > 0 else 1024.0
                sp_128 = seg_p[bi].cpu().numpy()  # (C, 128, 128)
                H_s, W_s = sp_128.shape[1], sp_128.shape[2]
                for chd_fid in NIH_TO_CHEST_HD_MAP:
                    b_list = bbox_map.get((im_name, chd_fid))
                    if not b_list:
                        continue
                    c_idx = findings.index(chd_fid)
                    hm = sp_128[c_idx]
                    gt_mask = np.zeros((H_s, W_s), dtype=bool)
                    for x1, y1, x2, y2 in b_list:
                        ix1 = max(0, min(W_s - 1, int(np.floor(x1 / w0 * W_s))))
                        iy1 = max(0, min(H_s - 1, int(np.floor(y1 / h0 * H_s))))
                        ix2 = max(ix1 + 1, min(W_s, int(np.ceil(x2 / w0 * W_s))))
                        iy2 = max(iy1 + 1, min(H_s, int(np.ceil(y2 / h0 * H_s))))
                        gt_mask[iy1:iy2, ix1:ix2] = True
                    py, px = np.unravel_index(int(np.argmax(hm)), hm.shape)
                    bbox_hits[chd_fid].append(1 if gt_mask[py, px] else 0)
                    hm_norm = hm / max(float(hm.max()), 1e-6)
                    pred_m = hm_norm >= 0.45
                    inter = int(np.logical_and(pred_m, gt_mask).sum())
                    union = int(np.logical_or(pred_m, gt_mask).sum())
                    bbox_ious[chd_fid].append(inter / float(max(union, 1)))

    pids_arr = np.asarray(pids_list)
    scores_mat = np.concatenate(scores_chunks, axis=0)

    results: dict[str, Any] = {}
    for chd_fid, nih_tuple in NIH_TO_CHEST_HD_MAP.items():
        c_idx = findings.index(chd_fid)
        yt = np.asarray(y_mapped[chd_fid], dtype=np.int32)
        ys = scores_mat[:, c_idx]
        auroc = float(roc_auc_score(yt, ys))
        auprc = float(average_precision_score(yt, ys))
        auroc_ci = _patient_bootstrap_ci(pids_arr, yt, ys, roc_auc_score)
        auprc_ci = _patient_bootstrap_ci(pids_arr, yt, ys, average_precision_score)
        entry: dict[str, Any] = {
            "display_name": CHEST_HD_DISPLAY_NAMES.get(chd_fid, chd_fid),
            "nih_source_labels": list(nih_tuple),
            "n_images": int(len(yt)),
            "n_positives": int(yt.sum()),
            "auroc": round(auroc, 4),
            "auroc_95ci": [round(auroc_ci[0], 4), round(auroc_ci[1], 4)],
            "auprc": round(auprc, 4),
            "auprc_95ci": [round(auprc_ci[0], 4), round(auprc_ci[1], 4)],
            "brier_raw": round(float(brier_score_loss(yt, np.clip(ys, 0.0, 1.0))), 4),
        }
        if bbox_hits[chd_fid]:
            entry["nih_bbox_cases"] = len(bbox_hits[chd_fid])
            entry["pointing_game_hit_rate"] = round(float(np.mean(bbox_hits[chd_fid])), 4)
            entry["mean_bbox_iou"] = round(float(np.mean(bbox_ious[chd_fid])), 4)
        results[chd_fid] = entry

    payload = {
        "split": args.split,
        "checkpoint": rec["filename"],
        "checkpoint_sha256": rec["sha256"],
        "mapped_findings_count": len(NIH_TO_CHEST_HD_MAP),
        "mapping": {k: list(v) for k, v in NIH_TO_CHEST_HD_MAP.items()},
        "unmapped_nih_labels": UNMAPPED_NIH_LABELS,
        "unmapped_chest_hd_findings": UNMAPPED_CHEST_HD_FINDINGS,
        "metrics": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote external NIH evaluation report ({args.split}) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
