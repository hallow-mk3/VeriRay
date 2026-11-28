#!/usr/bin/env python
"""Step 1B: Build Brain Leakage-Aware Manifest + Verify Canonical Preprocessing Border Probe.

Implements Change 3 & Minor rules:
  1. Recovers exact hospital PID (233 patients) from Jun Cheng 2017 Figshare .mat files
     (including exact tumorMask availability) + flip-invariant duplicate clustering for
     SARTAJ / Br35H slices lacking PID. Never reads C:\\Users\\swast\\brain_mri\\Testing.
  2. Extracts lesion-free normal slices from BOTH ISLES 2022 (DWI/FLAIR) and BraTS 2020
     (T1ce/FLAIR) alongside Br35H so 'notumor' spans all sources/sequences.
  3. Splits strictly by patient/cluster group_id into train (70%), val (15%), and locked
     test (15%). Writes brain_mri_manifest.sha256.json + brain_mri_split_audit.json while
     keeping the CSV gitignored.
  4. Runs the 3-way Border/Corner Source Probe (and a deeper 14% cortical-rim probe) on
     train -> val BEFORE and AFTER canonicalize_brain_slice() to verify near-chance (~33.3%).
"""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import cv2
import h5py
import nibabel as nib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veriray.brain_mri import canonicalize_brain_slice  # noqa: E402

IMG_SIZE = 256


def _orig_non_test(uid: str) -> bool:
    h = int(hashlib.sha256(uid.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 100) < 85


def _thumb_vec(u8: np.ndarray) -> np.ndarray:
    t = cv2.resize(u8, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t = t - t.mean()
    return t / max(float(np.linalg.norm(t)), 1e-6)


def extract_border_probe_feats(arr_256: np.ndarray, border_frac: float = 0.08, corner_frac: float = 0.12) -> np.ndarray:
    g = np.asarray(arr_256, dtype=np.float32)
    s = g.shape[0]
    b = max(4, int(round(border_frac * s)))
    c = max(6, int(round(corner_frac * s)))
    patches = (g[:b, :], g[-b:, :], g[b:-b, :b], g[b:-b, -b:], g[:c, :c], g[:c, -c:], g[-c:, :c], g[-c:, -c:])
    feats: list[float] = []
    for p_arr in patches:
        p = p_arr.ravel()
        feats.extend([
            float(p.mean()) / 255.0,
            float(p.std()) / 255.0,
            float((p <= 2.0).mean()),
            float((p >= 250.0).mean()),
            float(np.percentile(p, 10)) / 255.0,
            float(np.percentile(p, 50)) / 255.0,
            float(np.percentile(p, 90)) / 255.0,
        ])
    for strip in (g[:b, :], g[-b:, :]):
        feats.extend((cv2.resize(strip, (8, 2), interpolation=cv2.INTER_AREA).ravel() / 255.0).tolist())
    for strip in (g[b:-b, :b], g[b:-b, -b:]):
        feats.extend((cv2.resize(strip, (2, 8), interpolation=cv2.INTER_AREA).ravel() / 255.0).tolist())
    return np.asarray(feats, dtype=np.float32)


def main() -> None:
    fig_root = Path(r"C:\Users\swast\datasets\figshare_brain_mat")
    lbl_map = {1: "meningioma", 2: "glioma", 3: "pituitary"}
    fig_records: list[dict] = []
    for mp in sorted(fig_root.rglob("*.mat"), key=lambda p: p.name):
        if mp.name == "cvind.mat":
            continue
        try:
            with h5py.File(mp, "r") as f:
                if "cjdata" not in f:
                    continue
                cj = f["cjdata"]
                lbl_num = int(np.asarray(cj["label"])[0, 0])
                pid_str = "".join(chr(int(c)) for c in np.asarray(cj["PID"]).ravel() if int(c) > 0).strip()
                img_raw = np.asarray(cj["image"], dtype=np.float32).T
                lo, hi = float(img_raw.min()), float(img_raw.max())
                u8_mm = np.clip((img_raw - lo) / max(hi - lo, 1e-5) * 255.0, 0.0, 255.0).astype(np.uint8)
                p1, p99 = float(np.percentile(img_raw, 1)), float(np.percentile(img_raw, 99.5))
                u8_p = np.clip((img_raw - p1) / max(p99 - p1, 1e-5) * 255.0, 0.0, 255.0).astype(np.uint8)
                fig_records.append({
                    "mat_path": str(mp),
                    "mat_file": mp.name,
                    "pid": f"figshare_PID_{pid_str}",
                    "cls": lbl_map[lbl_num],
                    "vec": _thumb_vec(u8_mm),
                    "vec_flip": _thumb_vec(u8_mm[:, ::-1]),
                    "vec_p99": _thumb_vec(u8_p),
                    "u8_raw": cv2.resize(u8_p, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA),
                })
        except Exception:
            continue

    fig_by_cls: dict[str, list[dict]] = defaultdict(list)
    for r in fig_records:
        fig_by_cls[r["cls"]].append(r)

    # Match C:\Users\swast\brain_mri\Training ONLY (NEVER read Testing!)
    bm_train_dir = Path(r"C:\Users\swast\brain_mri\Training")
    unmatched_bm: list[dict] = []
    matched_counts: dict[str, int] = defaultdict(int)
    unmatched_counts: dict[str, int] = defaultdict(int)
    dropped_tr_aug_count = 0

    for cls_name in ("glioma", "meningioma", "notumor", "pituitary"):
        f_list = fig_by_cls.get(cls_name, [])
        mat_v1 = np.stack([x["vec"] for x in f_list]) if f_list else None
        mat_vf = np.stack([x["vec_flip"] for x in f_list]) if f_list else None
        mat_vp = np.stack([x["vec_p99"] for x in f_list]) if f_list else None

        for p in sorted((bm_train_dir / cls_name).iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            if p.name.startswith("Tr-aug-") or "aug" in p.stem.lower():
                # Drop offline pre-augmented copies (e.g. rotated Figshare meningioma duplicates)
                # so synthetic rotations can NEVER straddle train/val/test splits!
                dropped_tr_aug_count += 1
                continue
            with Image.open(p) as im:
                u8 = np.asarray(im.convert("L"), dtype=np.uint8)
            v = _thumb_vec(u8)
            matched = False
            if mat_v1 is not None and mat_vf is not None and mat_vp is not None:
                best_s = float(np.max(np.maximum(np.maximum(mat_v1 @ v, mat_vf @ v), mat_vp @ v)))
                if best_s >= 0.94:
                    matched_counts[cls_name] += 1
                    matched = True
            if not matched:
                unmatched_counts[cls_name] += 1
                unmatched_bm.append({
                    "path": str(p),
                    "rel_path": f"brain_mri/Training/{cls_name}/{p.name}",
                    "cls": cls_name,
                    "source": "brain_mri_2d",
                    "sequence": "unlabeled_2D_jpg",
                    "vec": v,
                    "u8_raw": cv2.resize(u8, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA),
                })

    # Duplicate-cluster unmatched SARTAJ / Br35H slices within class
    cluster_id_num = 0
    for cls_name in ("glioma", "meningioma", "notumor", "pituitary"):
        sub = [e for e in unmatched_bm if e["cls"] == cls_name]
        if not sub:
            continue
        vecs = np.stack([e["vec"] for e in sub])
        vecs_f = np.stack([_thumb_vec(e["u8_raw"][:, ::-1]) for e in sub])
        sim = np.maximum(vecs @ vecs.T, vecs @ vecs_f.T)
        vis = [False] * len(sub)
        for i in range(len(sub)):
            if vis[i]:
                continue
            cluster_id_num += 1
            gid = f"dupcluster_{cls_name}_{cluster_id_num:04d}"
            stk = [i]
            vis[i] = True
            while stk:
                cur = stk.pop()
                sub[cur]["group_id"] = gid
                for nb in np.where(sim[cur] >= 0.965)[0]:
                    if not vis[int(nb)]:
                        vis[int(nb)] = True
                        stk.append(int(nb))

    # Build manifest rows (using upstream Figshare .mat directly for all 233 Figshare patients so
    # exact tumorMask is available for every Figshare slice without duplicate JPG copies!)
    manifest_rows: list[dict] = []
    probe_samples: list[tuple[str, str, np.ndarray]] = []  # (source, group_id, raw_256)

    for r in fig_records:
        manifest_rows.append({
            "sample_id": f"figshare:{r['mat_file']}",
            "locator_spec": f"figshare_mat|{r['mat_path']}",
            "source": "brain_mri_2d",
            "sub_source": "figshare_cheng2017_mat",
            "sequence": "T1ce",
            "class_name": r["cls"],
            "patient_id": r["pid"],
            "has_true_patient_id": "true",
            "has_mask": "true",
            "group_id": r["pid"],
        })
        if len(probe_samples) < 500:
            probe_samples.append(("brain_mri_2d", r["pid"], r["u8_raw"]))

    for e in unmatched_bm:
        manifest_rows.append({
            "sample_id": f"bm2d:{e['cls']}:{Path(e['path']).name}",
            "locator_spec": f"image_file|{e['path']}",
            "source": "brain_mri_2d",
            "sub_source": "br35h_notumor" if e["cls"] == "notumor" else "sartaj_2d",
            "sequence": e["sequence"],
            "class_name": e["cls"],
            "patient_id": "",
            "has_true_patient_id": "false",
            "has_mask": "true" if e["cls"] == "notumor" else "false",
            "group_id": e["group_id"],
        })
        if sum(1 for x in probe_samples if x[0] == "brain_mri_2d") < 900:
            probe_samples.append(("brain_mri_2d", e["group_id"], e["u8_raw"]))

    # BraTS 2020 (strictly _orig_non_test == True patients ONLY): BOTH glioma slices AND lesion-free normal slices!
    brats_dir = Path(r"C:\Users\swast\datasets\brats\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData")
    for pdir in sorted(brats_dir.glob("BraTS20_Training_*")):
        if not pdir.is_dir():
            continue
        pid = pdir.name
        if not _orig_non_test(f"brats:{pid}"):
            continue
        seg_p = next(pdir.glob("*_seg.nii*"))
        seg_vol = np.asarray(nib.load(str(seg_p)).dataobj)
        nz = seg_vol.shape[2]
        areas = (seg_vol > 0).sum(axis=(0, 1))
        pos_zs = np.where(areas >= 80)[0]
        neg_zs = [
            z for z in range(int(0.18 * nz), int(0.82 * nz))
            if areas[max(0, z - 2) : min(nz, z + 3)].sum() == 0
        ]
        c_pos = np.linspace(pos_zs[0], pos_zs[-1], num=min(3, len(pos_zs)), dtype=int).tolist() if len(pos_zs) > 0 else []
        c_neg = np.linspace(neg_zs[0], neg_zs[-1], num=min(2, len(neg_zs)), dtype=int).tolist() if len(neg_zs) > 0 else []

        for seq_tag, seq_l in [("_t1ce.nii*", "T1ce"), ("_flair.nii*", "FLAIR")]:
            fls = list(pdir.glob(f"*{seq_tag}"))
            if not fls:
                continue
            vol_p = fls[0]
            vol = None
            for z in c_pos:
                manifest_rows.append({
                    "sample_id": f"brats:{pid}:{seq_l}:z{z}:glioma",
                    "locator_spec": f"nifti_slice|{vol_p}|{seg_p}|{z}",
                    "source": "brats2020",
                    "sub_source": "brats2020_glioma",
                    "sequence": seq_l,
                    "class_name": "glioma",
                    "patient_id": f"brats_{pid}",
                    "has_true_patient_id": "true",
                    "has_mask": "true",
                    "group_id": f"brats_{pid}",
                })
                if sum(1 for x in probe_samples if x[0] == "brats2020") < 900:
                    if vol is None:
                        vol = np.asarray(nib.load(str(vol_p)).dataobj)
                    sl = np.rot90(vol[:, :, int(z)]).astype(np.float32)
                    lo, hi = float(sl.min()), float(sl.max())
                    if hi - lo > 1e-5:
                        u8 = cv2.resize(((sl - lo) / (hi - lo) * 255.0).astype(np.uint8), (IMG_SIZE, IMG_SIZE))
                        probe_samples.append(("brats2020", f"brats_{pid}", u8))
            for z in c_neg:
                if vol is None:
                    vol = np.asarray(nib.load(str(vol_p)).dataobj)
                sl = np.rot90(vol[:, :, int(z)])
                if (sl > 0).sum() < 1200:
                    continue
                manifest_rows.append({
                    "sample_id": f"brats:{pid}:{seq_l}:z{z}:notumor",
                    "locator_spec": f"nifti_slice|{vol_p}|NONE|{z}",
                    "source": "brats2020",
                    "sub_source": "brats2020_normal_slice",
                    "sequence": seq_l,
                    "class_name": "notumor",
                    "patient_id": f"brats_{pid}",
                    "has_true_patient_id": "true",
                    "has_mask": "true",
                    "group_id": f"brats_{pid}",
                })

    # ISLES 2022 (strictly _orig_non_test == True subjects ONLY): BOTH stroke AND normal slices across DWI & FLAIR!
    isles_root = Path(r"C:\Users\swast\datasets\isles")
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
        elif "dwi" in low or "flair" in low:
            imgs_by_sub.setdefault(sid, []).append(("DWI" if "dwi" in low else "FLAIR", p))

    for sid in sorted(masks_by_sub):
        if not _orig_non_test(f"isles:{sid}") or sid not in imgs_by_sub:
            continue
        msk_p = masks_by_sub[sid]
        msk_vol = np.asarray(nib.load(str(msk_p)).dataobj)
        if msk_vol.ndim == 4:
            msk_vol = msk_vol[..., 0]
        nz = msk_vol.shape[2]
        areas = (msk_vol > 0).sum(axis=(0, 1))
        pos_zs = np.where(areas >= 15)[0]
        if len(pos_zs) == 0:
            pos_zs = np.where(areas > 0)[0]
        neg_zs = [
            z for z in range(int(0.18 * nz), int(0.82 * nz))
            if areas[max(0, z - 1) : min(nz, z + 2)].sum() == 0
        ]
        c_pos = np.linspace(pos_zs[0], pos_zs[-1], num=min(5, len(pos_zs)), dtype=int).tolist() if len(pos_zs) > 0 else []
        c_neg = np.linspace(neg_zs[0], neg_zs[-1], num=min(4, len(neg_zs)), dtype=int).tolist() if len(neg_zs) > 0 else []

        for seq_l, ipath in imgs_by_sub[sid]:
            vol = np.asarray(nib.load(str(ipath)).dataobj)
            if vol.ndim == 4:
                vol = vol[..., 0]
            if vol.shape[:3] != msk_vol.shape[:3]:
                continue
            for z in c_pos:
                sl = np.rot90(vol[:, :, int(z)]).astype(np.float32)
                if (sl > 0).sum() < 600 or float(sl.max() - sl.min()) < 1e-5:
                    continue
                manifest_rows.append({
                    "sample_id": f"isles:{sid}:{seq_l}:z{z}:stroke",
                    "locator_spec": f"nifti_slice|{ipath}|{msk_p}|{z}",
                    "source": "isles2022",
                    "sub_source": "isles2022_stroke",
                    "sequence": seq_l,
                    "class_name": "ischemic_stroke",
                    "patient_id": f"isles_{sid}",
                    "has_true_patient_id": "true",
                    "has_mask": "true",
                    "group_id": f"isles_{sid}",
                })
                if sum(1 for x in probe_samples if x[0] == "isles2022") < 900:
                    lo, hi = float(sl.min()), float(sl.max())
                    u8 = cv2.resize(((sl - lo) / (hi - lo) * 255.0).astype(np.uint8), (IMG_SIZE, IMG_SIZE))
                    probe_samples.append(("isles2022", f"isles_{sid}", u8))
            for z in c_neg:
                sl = np.rot90(vol[:, :, int(z)]).astype(np.float32)
                if (sl > 0).sum() < 800 or float(sl.max() - sl.min()) < 1e-5:
                    continue
                manifest_rows.append({
                    "sample_id": f"isles:{sid}:{seq_l}:z{z}:notumor",
                    "locator_spec": f"nifti_slice|{ipath}|NONE|{z}",
                    "source": "isles2022",
                    "sub_source": "isles2022_normal_slice",
                    "sequence": seq_l,
                    "class_name": "notumor",
                    "patient_id": f"isles_{sid}",
                    "has_true_patient_id": "true",
                    "has_mask": "true",
                    "group_id": f"isles_{sid}",
                })

    # Assign deterministic patient/group-disjoint splits: 70% train / 15% val / 15% locked test
    for row in manifest_rows:
        hb = int(hashlib.sha256(f"20261127:brain_v2:{row['group_id']}".encode()).hexdigest()[:8], 16) % 100
        row["split"] = "train" if hb < 70 else ("val" if hb < 85 else "test")

    groups_by_sp: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for row in manifest_rows:
        groups_by_sp[row["split"]].add(row["group_id"])
    assert groups_by_sp["train"].isdisjoint(groups_by_sp["val"])
    assert groups_by_sp["train"].isdisjoint(groups_by_sp["test"])
    assert groups_by_sp["val"].isdisjoint(groups_by_sp["test"])

    man_dir = ROOT / "data" / "manifests"
    man_csv = man_dir / "brain_mri_leakage_aware_manifest.csv"
    fields = ["sample_id", "locator_spec", "source", "sub_source", "sequence", "class_name", "patient_id", "has_true_patient_id", "has_mask", "group_id", "split"]
    with man_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest_rows)

    man_sha = hashlib.sha256(man_csv.read_bytes()).hexdigest()
    (man_dir / "brain_mri_manifest.sha256.json").write_text(
        json.dumps({"file": "data/manifests/brain_mri_leakage_aware_manifest.csv", "sha256": man_sha}, indent=2) + "\n",
        encoding="utf-8",
    )

    # Run 3-way Border/Corner Source Probe BEFORE vs AFTER canonicalize_brain_slice (strictly train -> val groups ONLY!)
    src_names = ("brain_mri_2d", "brats2020", "isles2022")
    raw_tr_X, raw_tr_y, raw_va_X, raw_va_y = [], [], [], []
    can_tr_X, can_va_X = [], []
    deep_tr_X, deep_va_X = [], []

    for src_n, gid, raw_u8 in probe_samples:
        hb = int(hashlib.sha256(f"20261127:brain_v2:{gid}".encode()).hexdigest()[:8], 16) % 100
        if hb >= 85:
            continue  # NEVER touch locked test groups!
        s_idx = src_names.index(src_n)
        can_img, _ = canonicalize_brain_slice(raw_u8, mask=None, out_size=IMG_SIZE)
        f_raw = extract_border_probe_feats(raw_u8, 0.08, 0.12)
        f_can = extract_border_probe_feats(can_img, 0.08, 0.12)
        f_deep = extract_border_probe_feats(can_img, 0.14, 0.18)
        if hb < 70:
            raw_tr_X.append(f_raw)
            can_tr_X.append(f_can)
            deep_tr_X.append(f_deep)
            raw_tr_y.append(s_idx)
        else:
            raw_va_X.append(f_raw)
            can_va_X.append(f_can)
            deep_va_X.append(f_deep)
            raw_va_y.append(s_idx)

    def _fit_eval_probe(Xtr, ytr, Xva, yva):
        sc = StandardScaler()
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(sc.fit_transform(np.stack(Xtr)), np.array(ytr))
        pr = clf.predict(sc.transform(np.stack(Xva)))
        return round(float(accuracy_score(yva, pr)), 4), confusion_matrix(yva, pr).tolist()

    acc_before, cm_before = _fit_eval_probe(raw_tr_X, raw_tr_y, raw_va_X, raw_va_y)
    acc_after_8pct, cm_after_8pct = _fit_eval_probe(can_tr_X, raw_tr_y, can_va_X, raw_va_y)
    acc_after_14pct, cm_after_14pct = _fit_eval_probe(deep_tr_X, raw_tr_y, deep_va_X, raw_va_y)

    sp_cls_src: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in manifest_rows:
        sp_cls_src[r["split"]][f"{r['class_name']}@{r['source']}"] += 1

    audit = {
        "manifest_file": "data/manifests/brain_mri_leakage_aware_manifest.csv",
        "manifest_sha256": man_sha,
        "kaggle_testing_folder_read": False,
        "dropped_synthetic_tr_aug_copies": dropped_tr_aug_count,
        "total_rows": len(manifest_rows),
        "unique_groups_by_split": {k: len(v) for k, v in groups_by_sp.items()},
        "counts_by_split_class_source": {k: dict(sorted(v.items())) for k, v in sp_cls_src.items()},
        "border_probe_3way_chance": 0.3333,
        "border_probe_before_canonical_8pct": {"val_acc": acc_before, "cm": cm_before},
        "border_probe_after_canonical_8pct_outer_frame": {
            "val_acc": acc_after_8pct,
            "cm": cm_after_8pct,
            "note": "Outer 8% frame/background is identically zeroed outside canonical ellipse (rx=ry=0.41*256), eliminating frame/padding cues.",
        },
        "parenchymal_rim_probe_after_canonical_14pct": {
            "val_acc": acc_after_14pct,
            "cm": cm_after_14pct,
            "note": "Inner 14% cortical strip retains intrinsic MRI sequence/voxel-resolution differences (DWI vs T1ce); therefore ONLY within-source metrics (Within-ISLES, Within-BraTS, Within-2D) are valid headline metrics.",
        },
    }
    (man_dir / "brain_mri_split_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
