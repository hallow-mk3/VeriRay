#!/usr/bin/env python
"""One-Shot Locked Test Evaluation for De-Confounded Chest HD (15 findings) & Brain MRI HD (5 classes).

Requires explicit ``--unlock-test`` flag.
Verifies frozen SHA-256 hashes of:
  - configs/verifiable.yaml
  - data/manifests/chest_hd_manifest.csv
  - data/manifests/brain_mri_leakage_aware_manifest.csv

Evaluates each Chest HD finding against the unchanged frozen rule (positives >= 100,
box/mask Dice >= 0.50, no cross-source confound, and border_probe_auroc <= 0.80) on BOTH
dev_val and locked test to determine the final verified SUPPORTED_FINDINGS list.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import cv2
import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.train_brain_mri_hd import BrainCanonicalDataset, BrainSample, evaluate_within_source  # noqa: E402
from scripts.train_chest_hd import (  # noqa: E402
    DATASETS_DIR,
    PNEUMONIA_IDX,
    PNEUMOTHORAX_IDX,
    SEED,
    ChestMultiDataset,
    SampleMeta,
    _border_feats_256,
    evaluate_val,
)
from veriray.brain_mri import BRAIN_HD_CLASSES, BRAIN_RECORD, BrainHDNet, canonicalize_brain_slice  # noqa: E402
from veriray.chest_hd import CHEST_HD_DISPLAY_NAMES, CHEST_HD_FINDINGS, CHEST_HD_RECORD, ChestHDNet  # noqa: E402

BORDER_PROBE_MAX_AUROC = 0.80


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_frozen_hashes() -> dict[str, str]:
    checks = [
        (ROOT / "configs" / "verifiable.yaml", ROOT / "configs" / "verifiable.sha256.json"),
        (ROOT / "data" / "manifests" / "chest_hd_manifest.csv", ROOT / "data" / "manifests" / "chest_hd_manifest.sha256.json"),
        (ROOT / "data" / "manifests" / "brain_mri_leakage_aware_manifest.csv", ROOT / "data" / "manifests" / "brain_mri_manifest.sha256.json"),
    ]
    out: dict[str, str] = {}
    for target, sha_json in checks:
        expected = json.loads(sha_json.read_text(encoding="utf-8"))["sha256"].lower()
        actual = _sha256(target)
        if actual != expected:
            raise RuntimeError(f"Frozen hash mismatch for {target}: expected {expected}, got {actual}")
        out[str(target.relative_to(ROOT)).replace("\\", "/")] = actual
    return out


def _load_chest_splits_for_locked_eval() -> tuple[list[SampleMeta], list[SampleMeta]]:
    csv_p = ROOT / "data" / "manifests" / "chest_hd_manifest.csv"
    split_by_uid: dict[str, str] = {}
    with csv_p.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] in ("train", "test"):
                split_by_uid[row["uid"]] = row["split"]

    samples_tr: list[SampleMeta] = []
    samples_te: list[SampleMeta] = []

    # 1. VinBigData (source-stratified pneumothorax negatives with deterministic seed matching train/val)
    vbd_dir = DATASETS_DIR / "vindr_cxr" / "vinbigdata"
    df = pd.read_csv(vbd_dir / "train.csv")
    base_non_ptx = [c for c in range(14) if c != PNEUMOTHORAX_IDX]
    rng_strat = random.Random(SEED + 99)

    for img_id, grp in df.groupby("image_id"):
        uid = f"vbd:{img_id}"
        sp = split_by_uid.get(uid)
        if sp is None:
            continue
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
        include_ptx = (PNEUMOTHORAX_IDX in pos_set) or (rng_strat.random() < 0.023)
        known_cls = sorted(base_non_ptx + ([PNEUMOTHORAX_IDX] if include_ptx else []))
        sm = SampleMeta(
            uid=uid,
            source="vinbigdata",
            img_path=img_path,
            known_cls_indices=known_cls,
            pos_cls_indices=sorted(pos_set),
            known_seg_indices=known_cls,
            boxes_norm=boxes,
        )
        (samples_tr if sp == "train" else samples_te).append(sm)

    # 2. SIIM-ACR Pneumothorax
    siim_root = DATASETS_DIR / "siim_pneumothorax" / "siim-acr-pneumothorax"
    sdf = pd.read_csv(siim_root / "stage_1_train_images.csv")
    for row in sdf.itertuples(index=False):
        uid = f"siim:{row.ImageId}"
        sp = split_by_uid.get(uid)
        if sp is None:
            continue
        fname = str(row.new_filename)
        img_p = siim_root / "png_images" / fname
        msk_p = siim_root / "png_masks" / fname
        if not img_p.is_file():
            continue
        is_pos = int(row.has_pneumo) == 1
        sm = SampleMeta(
            uid=uid,
            source="siim_pneumo",
            img_path=img_p,
            known_cls_indices=[PNEUMOTHORAX_IDX],
            pos_cls_indices=[PNEUMOTHORAX_IDX] if is_pos else [],
            known_seg_indices=[PNEUMOTHORAX_IDX] if msk_p.is_file() else [],
            pneumo_mask_path=msk_p if (is_pos and msk_p.is_file()) else None,
        )
        (samples_tr if sp == "train" else samples_te).append(sm)

    # 3. RSNA Pneumonia
    rsna_root = DATASETS_DIR / "rsna_pneumonia" / "input"
    rdf = pd.read_csv(rsna_root / "stage_2_train_labels.csv")
    for pid, grp in rdf.groupby("patientId"):
        uid = f"rsna:{pid}"
        sp = split_by_uid.get(uid)
        if sp is None:
            continue
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
                        boxes_r.append((PNEUMONIA_IDX, x1, y1, x2, y2))
        sm = SampleMeta(
            uid=uid,
            source="rsna_pneumonia",
            img_path=img_p,
            known_cls_indices=[PNEUMONIA_IDX],
            pos_cls_indices=[PNEUMONIA_IDX] if is_pos else [],
            known_seg_indices=[PNEUMONIA_IDX],
            boxes_norm=boxes_r,
        )
        (samples_tr if sp == "train" else samples_te).append(sm)

    assert {s.uid for s in samples_tr}.isdisjoint({s.uid for s in samples_te})
    return samples_tr, samples_te


def _run_chest_border_probe_on_test(train_all: list[SampleMeta], test_all: list[SampleMeta]) -> dict[str, float]:
    rng = random.Random(SEED)
    tr_sub = list(train_all)
    rng.shuffle(tr_sub)
    tr_sub = tr_sub[:4500]
    X_tr = np.stack([_border_feats_256(s.img_path) for s in tr_sub])
    X_te = np.stack([_border_feats_256(s.img_path) for s in test_all])
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    out: dict[str, float] = {}
    for c_idx, fid in enumerate(CHEST_HD_FINDINGS):
        tr_idx = [i for i, s in enumerate(tr_sub) if c_idx in s.known_cls_indices]
        te_idx = [i for i, s in enumerate(test_all) if c_idx in s.known_cls_indices]
        y_tr = np.array([1 if c_idx in tr_sub[i].pos_cls_indices else 0 for i in tr_idx])
        y_te = np.array([1 if c_idx in test_all[i].pos_cls_indices else 0 for i in te_idx])
        if 5 <= y_tr.sum() <= len(y_tr) - 5 and 2 <= y_te.sum() <= len(y_te) - 2:
            clf = LogisticRegression(max_iter=800, C=0.5)
            clf.fit(X_tr_s[tr_idx], y_tr)
            pr = clf.predict_proba(X_te_s[te_idx])[:, 1]
            out[fid] = round(float(roc_auc_score(y_te, pr)), 4)
        else:
            out[fid] = 0.5
    return out


def _passes_split_gate(m: dict[str, Any], min_pos: int, min_dice: float, max_border: float) -> tuple[bool, list[str]]:
    fails: list[str] = []
    n_pos = int(m.get("val_positives") or 0)
    dice = m.get("pos_box_or_mask_dice") if m.get("pos_box_or_mask_dice") is not None else m.get("pos_mask_dice")
    bp = float(m.get("border_probe_auroc", m.get("border_probe_val_auroc", 0.5)))
    within = dict(m.get("within_source_auroc") or {})
    pooled = float(m.get("auroc") or 0.0)

    if n_pos < min_pos:
        fails.append(f"positives {n_pos} < {min_pos}")
    if dice is None or float(dice) < min_dice:
        fails.append(f"dice {dice} < {min_dice}")
    if len(within) > 1 and pooled > max(within.values()) + 0.02:
        fails.append(f"cross-source pooled AUROC inflation ({pooled:.4f} > max within-source {max(within.values()):.4f})")
    if bp > max_border:
        fails.append(f"border_probe_auroc {bp:.4f} > {max_border:.2f}")
    return (len(fails) == 0), fails


def _load_brain_locked_test() -> list[BrainSample]:
    csv_p = ROOT / "data" / "manifests" / "brain_mri_leakage_aware_manifest.csv"
    rows_te = []
    with csv_p.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == "test":
                rows_te.append(row)

    rows_te.sort(key=lambda r: r["locator_spec"])
    nifti_cache: dict[str, np.ndarray] = {}

    def _get_vol(vp: str) -> np.ndarray:
        if vp not in nifti_cache:
            if len(nifti_cache) > 12:
                nifti_cache.clear()
            arr = np.asarray(nib.load(vp).dataobj)
            if arr.ndim == 4:
                arr = arr[..., 0]
            nifti_cache[vp] = arr
        return nifti_cache[vp]

    out: list[BrainSample] = []
    for r in rows_te:
        kind, rest = r["locator_spec"].split("|", 1)
        c_idx = BRAIN_HD_CLASSES.index(r["class_name"])
        raw_img, raw_msk = None, None
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
            if c_idx == 2:
                raw_msk = np.zeros(raw_img.shape, dtype=np.uint8)
        elif kind == "nifti_slice":
            vol_p, msk_p, z_s = rest.split("|")
            z = int(z_s)
            vol = _get_vol(vol_p)
            raw_img = np.rot90(vol[:, :, z]).astype(np.float32)
            if msk_p == "NONE":
                raw_msk = np.zeros(raw_img.shape, dtype=np.uint8)
            else:
                mvol = _get_vol(msk_p)
                raw_msk = np.rot90((mvol[:, :, z] > 0).astype(np.uint8))

        if raw_img is None:
            continue
        can_img, can_msk = canonicalize_brain_slice(raw_img, mask=raw_msk, out_size=256)
        if float(can_img.max()) <= 0.0:
            continue
        msk_64 = None
        if seg_known and can_msk is not None:
            msk_64 = (cv2.resize(can_msk, (64, 64), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        out.append(
            BrainSample(
                sample_id=r["sample_id"],
                source=r["source"],
                sequence=r["sequence"],
                cls_idx=c_idx,
                img_u8=np.clip(np.rint(can_img), 0, 255).astype(np.uint8),
                seg_known=seg_known,
                mask_u8=msk_64,
            )
        )
    nifti_cache.clear()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unlock-test", action="store_true", help="Explicitly unlock test splits once.")
    args = ap.parse_args()
    if not args.unlock_test:
        raise SystemExit("Refusing to unlock test splits without --unlock-test.")

    hashes = _verify_frozen_hashes()
    ver_cfg = yaml.safe_load((ROOT / "configs" / "verifiable.yaml").read_text(encoding="utf-8"))
    min_pos = int(ver_cfg["rule"]["min_val_positives"])
    min_dice = float(ver_cfg["rule"]["min_pos_mask_dice"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "artifacts" / "eval_test_locked"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Chest HD Locked Test
    print("[LockedTest] Evaluating Chest HD (15 findings) on locked test split...", flush=True)
    chd_rec = json.loads((ROOT / "models" / CHEST_HD_RECORD).read_text(encoding="utf-8"))
    val_metrics_map = chd_rec["val_metrics"]
    chest_net = ChestHDNet(num_classes=len(CHEST_HD_FINDINGS), pretrained=False).to(device)
    chest_net.load_state_dict(torch.load(ROOT / "models" / chd_rec["filename"], map_location="cpu", weights_only=True))
    chest_net.eval()

    tr_chest, te_chest = _load_chest_splits_for_locked_eval()
    te_loader = DataLoader(ChestMultiDataset(te_chest, augment=False), batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    te_raw_metrics, _ = evaluate_val(chest_net, te_loader, te_chest, device)
    te_border = _run_chest_border_probe_on_test(tr_chest, te_chest)

    del chest_net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    per_finding_gate: dict[str, Any] = {}
    passing_both: list[str] = []
    for fid in CHEST_HD_FINDINGS:
        vm = dict(val_metrics_map[fid])
        tm_raw = dict(te_raw_metrics[fid])
        tm = {
            "display_name": tm_raw["display_name"],
            "supervision_type": tm_raw["supervision_type"],
            "test_total": tm_raw["val_total"],
            "test_positives": tm_raw["val_positives"],
            "auroc": tm_raw["auroc"],
            "within_source_auroc": tm_raw["within_source_auroc"],
            "auprc": tm_raw["auprc"],
            "pos_box_or_mask_dice": tm_raw["pos_box_or_mask_dice"],
            "border_probe_test_auroc": te_border[fid],
            "border_shortcut_flagged": bool(te_border[fid] > BORDER_PROBE_MAX_AUROC),
        }
        val_ok, val_fails = _passes_split_gate(vm, min_pos, min_dice, BORDER_PROBE_MAX_AUROC)
        test_ok, test_fails = _passes_split_gate(
            {**tm_raw, "border_probe_auroc": te_border[fid]},
            min_pos,
            min_dice,
            BORDER_PROBE_MAX_AUROC,
        )
        both_ok = bool(val_ok and test_ok)
        if both_ok:
            passing_both.append(fid)
        per_finding_gate[fid] = {
            "passed_both_val_and_locked_test": both_ok,
            "passed_dev_val": val_ok,
            "dev_val_fail_reasons": val_fails,
            "passed_locked_test": test_ok,
            "locked_test_fail_reasons": test_fails,
            "dev_val": {
                "positives": vm["val_positives"],
                "within_source_auroc": vm["within_source_auroc"],
                "pos_box_or_mask_dice": vm.get("pos_box_or_mask_dice", vm.get("pos_mask_dice")),
                "border_probe_auroc": vm.get("border_probe_val_auroc"),
            },
            "locked_test": tm,
        }

    chest_locked_payload = {
        "split": "locked_test",
        "verified_hashes": hashes,
        "checkpoint": chd_rec["filename"],
        "checkpoint_sha256": chd_rec["sha256"],
        "test_scans_count": len(te_chest),
        "test_counts_by_source": {
            src: sum(1 for s in te_chest if s.source == src)
            for src in ("vinbigdata", "siim_pneumo", "rsna_pneumonia")
        },
        "macro_locked_test_auroc_15_findings": te_raw_metrics["_macro_auroc"],
        "frozen_admission_rule": {
            "min_positives_per_split": min_pos,
            "min_pos_box_or_mask_dice": min_dice,
            "max_border_probe_auroc": BORDER_PROBE_MAX_AUROC,
            "require_both_dev_val_and_locked_test": True,
        },
        "verified_supported_findings": passing_both,
        "findings": per_finding_gate,
    }
    (out_dir / "chest_hd_locked_test.json").write_text(json.dumps(chest_locked_payload, indent=2) + "\n", encoding="utf-8")
    print(f"[LockedTest] Chest HD macro test AUROC = {te_raw_metrics['_macro_auroc']:.4f} | Verified passing findings ({len(passing_both)}): {passing_both}", flush=True)

    # 2. Brain MRI HD Locked Test
    print("[LockedTest] Evaluating Brain MRI HD (5 classes) on locked test split...", flush=True)
    bm_rec = json.loads((ROOT / "models" / BRAIN_RECORD).read_text(encoding="utf-8"))
    brain_net = BrainHDNet(num_classes=len(BRAIN_HD_CLASSES), pretrained=False).to(device)
    brain_net.load_state_dict(torch.load(ROOT / "models" / bm_rec["filename"], map_location="cpu", weights_only=True))
    brain_net.eval()

    te_brain = _load_brain_locked_test()
    te_b_loader = DataLoader(BrainCanonicalDataset(te_brain, augment=False), batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    bm_te_eval = evaluate_within_source(brain_net, te_brain, te_b_loader, device)
    brain_locked_payload = {
        "split": "locked_test",
        "verified_hashes": hashes,
        "checkpoint": bm_rec["filename"],
        "checkpoint_sha256": bm_rec["sha256"],
        "test_slices_count": len(te_brain),
        "note": "One-shot locked test evaluation (Tr-aug synthetic copies purged; Kaggle Testing folder never read). Headline metrics are within-source.",
        "headline_within_isles_stroke_vs_normal": bm_te_eval["headline_within_isles_stroke_vs_normal"],
        "within_brats_glioma_vs_normal": bm_te_eval["within_brats_glioma_vs_normal"],
        "within_2d_4class": bm_te_eval["within_2d_4class"],
        "pooled_5cls_locked_test_accuracy": bm_te_eval["pooled_5cls_dev_val_accuracy"],
        "overall_lesion_segmentation_dice_locked_test": bm_te_eval["overall_lesion_segmentation_dice"],
        "classification_report_locked_test": bm_te_eval["classification_report_dev_val"],
    }
    (out_dir / "brain_mri_locked_test.json").write_text(json.dumps(brain_locked_payload, indent=2) + "\n", encoding="utf-8")
    wi = bm_te_eval["headline_within_isles_stroke_vs_normal"]
    wb = bm_te_eval["within_brats_glioma_vs_normal"]
    w2 = bm_te_eval["within_2d_4class"]
    print(
        f"[LockedTest] Brain HD within-ISLES AUROC={wi['auroc_p_stroke']:.4f} (dice={wi['stroke_mask_dice']:.4f}) | "
        f"within-BraTS AUROC={wb['auroc_p_glioma']:.4f} | within-2D 4cls acc={w2['accuracy_4cls']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
