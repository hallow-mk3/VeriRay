#!/usr/bin/env python
"""Build data/manifests/nih_manifest.csv from the official NIH ChestX-ray14 files (local, offline).

    python scripts/build_nih_manifest.py --data-entry D:\\nih\\Data_Entry_2017.csv \\
        --train-val D:\\nih\\train_val_list.txt --test D:\\nih\\test_list.txt --bbox D:\\nih\\BBox_List_2017.csv
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veriray.splits import (  # noqa: E402
    LeakageError, SplitConfig, _read_list, audit_rows, build_rows, read_bbox_images, read_entries, write_manifest,
)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-entry", type=Path, required=True)
    ap.add_argument("--train-val", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--bbox", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/manifests"))
    args = ap.parse_args()
    cfg = SplitConfig()
    missing = [str(p) for p in (args.data_entry, args.train_val, args.test, args.bbox) if p and not p.is_file()]
    if missing:
        print('INPUT FILE(S) NOT FOUND - download the NIH files first:\n  ' + '\n  '.join(missing), file=sys.stderr)
        return 2
    try:
        rows = build_rows(read_entries(args.data_entry), _read_list(args.train_val), _read_list(args.test),
                          read_bbox_images(args.bbox), cfg)
        audit = audit_rows(rows)
    except LeakageError as exc:
        print(f"LEAKAGE AUDIT FAILED - no manifest written: {exc}", file=sys.stderr)
        return 1
    inputs = {p.name: _sha(p) for p in (args.data_entry, args.train_val, args.test, args.bbox) if p}
    path = write_manifest(rows, audit, args.out, cfg, inputs)
    print(f"Wrote {path}")
    for s, c in audit["split_counts"].items():
        print(f"  {s:5} images={c['images']:7} patients={c['patients']:6} pneumothorax={c['pneumothorax_images']}")
    print("  bbox roles:", audit["bbox_role_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
