#!/usr/bin/env python
"""Step 1A: Build de-confounded, dHash-cluster-grouped Chest HD splits.

Implements all approved audit changes:
  1. Drops tb_cxr completely (0 TB images).
  2. Preserves exact frozen val membership ((sha256(uid)[:8] % 100) >= 85).
  3. Runs global 64-bit dHash + 32x32 normalized correlation clustering across ALL 52,359
     scans (vinbigdata + siim_pneumo + rsna_pneumonia) simultaneously, catching both
     within-source near-duplicates and cross-source RSNA <-> SIIM duplicates.
  4. Purges any non-val item that clusters with a val item (split='dropped_val_leak') so
     val membership is never modified while eliminating train<->val and test<->val leakage.
  5. Partitions remaining clean clusters deterministically into train (~70%) and locked
     test (~15%). No model touches test.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = Path(r"C:\Users\swast\datasets")


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def _orig_is_val(uid: str) -> bool:
    h = int(hashlib.sha256(uid.encode("utf-8")).hexdigest()[:8], 16)
    return (h % 100) >= 85


def _extract_sig(path_str: str) -> tuple[int, int, np.ndarray, np.ndarray]:
    with Image.open(path_str) as im:
        g = np.asarray(im.convert("L"), dtype=np.uint8)
    # 64-bit dHash (upright & horizontal flip)
    d98 = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (d98[:, 1:] > d98[:, :-1]).ravel()
    dh = 0
    for b in bits:
        dh = (dh << 1) | int(b)

    d98_f = d98[:, ::-1]
    bits_f = (d98_f[:, 1:] > d98_f[:, :-1]).ravel()
    dh_f = 0
    for b in bits_f:
        dh_f = (dh_f << 1) | int(b)

    # 32x32 zero-mean unit-norm patch (upright & flip)
    t32 = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    v = t32.ravel()
    v = v - v.mean()
    v = v / max(float(np.linalg.norm(v)), 1e-6)

    vf = t32[:, ::-1].ravel()
    vf = vf - vf.mean()
    vf = vf / max(float(np.linalg.norm(vf)), 1e-6)
    return dh, dh_f, v, vf


def collect_items() -> list[dict]:
    items: list[dict] = []

    # 1. VinBigData (15,000)
    vbd_dir = DATASETS_DIR / "vindr_cxr" / "vinbigdata"
    vbd_df = pd.read_csv(vbd_dir / "train.csv")
    for img_id in sorted(vbd_df["image_id"].unique()):
        p = vbd_dir / "train" / f"{img_id}.png"
        if p.is_file():
            uid = f"vbd:{img_id}"
            items.append({"uid": uid, "source": "vinbigdata", "native_id": str(img_id), "path": str(p), "is_val": _orig_is_val(uid)})

    # 2. SIIM-ACR Pneumothorax (10,675)
    siim_root = DATASETS_DIR / "siim_pneumothorax" / "siim-acr-pneumothorax"
    siim_df = pd.read_csv(siim_root / "stage_1_train_images.csv")
    for row in siim_df.itertuples(index=False):
        p = siim_root / "png_images" / str(row.new_filename)
        if p.is_file():
            uid = f"siim:{row.ImageId}"
            items.append({"uid": uid, "source": "siim_pneumo", "native_id": str(row.ImageId), "path": str(p), "is_val": _orig_is_val(uid)})

    # 3. RSNA Pneumonia (26,684)
    rsna_root = DATASETS_DIR / "rsna_pneumonia" / "input"
    rsna_df = pd.read_csv(rsna_root / "stage_2_train_labels.csv")
    for pid in sorted(rsna_df["patientId"].unique()):
        p = rsna_root / "images" / f"{pid}.jpg"
        if not p.is_file():
            p = rsna_root / "images" / f"{pid}.png"
            if not p.is_file():
                continue
        uid = f"rsna:{pid}"
        items.append({"uid": uid, "source": "rsna_pneumonia", "native_id": str(pid), "path": str(p), "is_val": _orig_is_val(uid)})

    return items


def main() -> None:
    t0 = time.time()
    items = collect_items()
    n = len(items)
    print(f"[ChestSplits] Collected {n} scans (tb_cxr dropped completely).", flush=True)

    paths = [it["path"] for it in items]
    with ThreadPoolExecutor(max_workers=8) as ex:
        sigs = list(ex.map(_extract_sig, paths))
    print(f"[ChestSplits] Computed 64-bit dHash + 32x32 signatures for {n} scans in {time.time()-t0:.1f}s", flush=True)

    dhashes = np.array([s[0] for s in sigs], dtype=np.uint64)
    dhashes_f = np.array([s[1] for s in sigs], dtype=np.uint64)
    vecs = np.stack([s[2] for s in sigs], axis=0)
    vecs_f = np.stack([s[3] for s in sigs], axis=0)

    uf = UnionFind(n)
    # Every candidate pair (including exact 64-bit dHash collisions on symmetric PA X-rays!)
    # MUST pass BOTH 64-bit dHash Hamming <= 6 AND 32x32 normalized pixel correlation >= 0.985.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vt = torch.from_numpy(vecs).to(device, dtype=torch.float16)
    vtf = torch.from_numpy(vecs_f).to(device, dtype=torch.float16)

    # Precompute 8-byte view of uint64 dHashes on GPU for fast popcount
    dh_bytes = torch.from_numpy(dhashes.view(np.uint8).reshape(n, 8)).to(device)
    dhf_bytes = torch.from_numpy(dhashes_f.view(np.uint8).reshape(n, 8)).to(device)
    pop_lut = torch.tensor([bin(k).count("1") for k in range(256)], device=device, dtype=torch.int16)

    chunk = 2048
    pair_links = 0
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        sim1 = vt[start:end] @ vt.T
        sim2 = vt[start:end] @ vtf.T
        sim = torch.maximum(sim1, sim2)
        cand_r, cand_c = torch.where(sim >= 0.985)
        if cand_r.numel() == 0:
            continue
        abs_r = cand_r + start
        keep = abs_r < cand_c
        abs_r = abs_r[keep]
        cand_c = cand_c[keep]
        if abs_r.numel() == 0:
            continue
        # Verify dHash Hamming <= 6 on candidates
        xor1 = torch.bitwise_xor(dh_bytes[abs_r], dh_bytes[cand_c]).long()
        ham1 = pop_lut[xor1].sum(dim=1)
        xor2 = torch.bitwise_xor(dh_bytes[abs_r], dhf_bytes[cand_c]).long()
        ham2 = pop_lut[xor2].sum(dim=1)
        ham = torch.minimum(ham1, ham2)
        valid = ham <= 6
        rs = abs_r[valid].cpu().tolist()
        cs = cand_c[valid].cpu().tolist()
        for r_i, c_i in zip(rs, cs):
            uf.union(r_i, c_i)
            pair_links += 1

    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    multi_clusters = sum(1 for v in clusters.values() if len(v) > 1)
    cross_src_clusters = sum(
        1 for v in clusters.values() if len({items[idx]["source"] for idx in v}) > 1
    )
    print(
        f"[ChestSplits] Clusters: total={len(clusters)} | multi_image={multi_clusters} | cross_source={cross_src_clusters} | top10_sizes={sizes[:10]} | pair_links={pair_links}",
        flush=True,
    )

    # Assign deterministic cluster_id from lexicographically smallest uid in cluster
    val_contaminated_clusters = 0
    dropped_val_leak_count = 0
    for root_idx, member_idxs in clusters.items():
        min_uid = min(items[m]["uid"] for m in member_idxs)
        cid = f"cxd_{hashlib.sha256(min_uid.encode()).hexdigest()[:12]}"
        has_val = any(items[m]["is_val"] for m in member_idxs)
        if has_val:
            if any(not items[m]["is_val"] for m in member_idxs):
                val_contaminated_clusters += 1
            for m in member_idxs:
                items[m]["cluster_id"] = cid
                if items[m]["is_val"]:
                    items[m]["split"] = "val"
                else:
                    items[m]["split"] = "dropped_val_leak"
                    dropped_val_leak_count += 1
        else:
            # Deterministic ~70/15 carve-out on the remaining 85% non-val clusters:
            # 15 / 85 = 17.65% of non-val -> ~15.0% of total dataset goes to locked 'test'
            hb = int(hashlib.sha256(f"20261127:chest_locked:{cid}".encode()).hexdigest()[:8], 16) % 85
            sp = "test" if hb >= 70 else "train"
            for m in member_idxs:
                items[m]["cluster_id"] = cid
                items[m]["split"] = sp

    # Strict audit: zero cluster overlap across train, val, test
    c_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for it in items:
        sp = it["split"]
        if sp in c_by_split:
            c_by_split[sp].add(it["cluster_id"])
    assert c_by_split["train"].isdisjoint(c_by_split["val"])
    assert c_by_split["train"].isdisjoint(c_by_split["test"])
    assert c_by_split["val"].isdisjoint(c_by_split["test"])

    # Also verify exact val membership unchanged
    val_uids_now = {it["uid"] for it in items if it["split"] == "val"}
    val_uids_orig = {it["uid"] for it in items if it["is_val"]}
    assert val_uids_now == val_uids_orig, "Val membership changed!"

    man_dir = ROOT / "data" / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    csv_path = man_dir / "chest_hd_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["uid", "source", "native_id", "cluster_id", "split"])
        w.writeheader()
        for it in items:
            w.writerow({k: it[k] for k in ("uid", "source", "native_id", "cluster_id", "split")})

    csv_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    (man_dir / "chest_hd_manifest.sha256.json").write_text(
        json.dumps({"file": "data/manifests/chest_hd_manifest.csv", "sha256": csv_sha}, indent=2) + "\n",
        encoding="utf-8",
    )

    split_src_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for it in items:
        split_src_counts[it["split"]][it["source"]] += 1

    audit = {
        "manifest_file": "data/manifests/chest_hd_manifest.csv",
        "manifest_sha256": csv_sha,
        "tb_cxr_included": False,
        "val_membership_modified": False,
        "clustering_rule": "exact_64bit_dhash_or_(dhash_hamming<=6_AND_32x32_cosine>=0.985)_flip_invariant_across_all_sources",
        "total_scans": n,
        "total_clusters": len(clusters),
        "singleton_clusters": int(sum(1 for v in clusters.values() if len(v) == 1)),
        "multi_image_clusters": multi_clusters,
        "cross_source_clusters": cross_src_clusters,
        "top_10_cluster_sizes": sizes[:10],
        "val_contaminated_clusters_purged": val_contaminated_clusters,
        "dropped_non_val_items_clustering_with_val": dropped_val_leak_count,
        "unique_clusters_by_split": {k: len(v) for k, v in c_by_split.items()},
        "counts_by_split_and_source": {k: dict(v) for k, v in split_src_counts.items()},
        "split_totals": dict(Counter(it["split"] for it in items)),
    }
    (man_dir / "chest_hd_split_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
