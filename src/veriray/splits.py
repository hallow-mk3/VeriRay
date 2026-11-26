"""NIH ChestX-ray14 patient-level split manifest and leakage audit (Blueprint 26 Nov §10; v2 §10.3).

Pure local file processing; no network. Inputs are the official NIH files you downloaded yourself:
Data_Entry_2017.csv, train_val_list.txt, test_list.txt and (optional) BBox_List_2017.csv.

Rules implemented:
- Official split lists are used. Official test -> ``test``.
- train_val_list is split BY PATIENT into ``train`` / ``val`` with a deterministic seeded hash.
- Box images that fall in the official test split are split BY PATIENT into a ``tuning`` half and a
  ``reporting`` half (never tune on reported images). Box images outside test get role ``not_in_test``.
- Labels are weak, report-mined labels. "No Finding" is NOT a certified healthy cohort.
- Any leakage (a patient in two splits, an image in two lists, missing/unknown images) raises
  :class:`LeakageError`: fail closed, no manifest is written.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

FINDING = "Pneumothorax"
SPLITS = ("train", "val", "test")
LABEL_NOTE = "NIH labels are text-mined from reports (weak labels), not radiologist ground truth."


class LeakageError(Exception):
    """Raised when a split audit fails. Never write or use a manifest after this."""


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 20261126
    val_fraction: float = 0.2
    bbox_tuning_fraction: float = 0.5


def _bucket(patient_id: str, seed: int, salt: str) -> float:
    """Deterministic uniform [0, 1) value per (seed, salt, patient)."""
    h = hashlib.sha256(f"{seed}:{salt}:{patient_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def _read_list(path: Path) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def _patient_from_image(image_index: str) -> str:
    return image_index.split("_", 1)[0]


def read_entries(csv_path: Path) -> dict[str, dict]:
    """Image Index -> row dict with normalized fields."""
    out: dict[str, dict] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            img = row["Image Index"].strip()
            if img in out:
                raise LeakageError(f"duplicate image in Data_Entry csv: {img}")
            labels = [x.strip() for x in row["Finding Labels"].split("|")]
            out[img] = {
                "image": img,
                "patient_id": str(row["Patient ID"]).strip().zfill(8),
                "labels": labels,
                "age": row.get("Patient Age", "").strip(),
                "sex": row.get("Patient Gender", "").strip(),
                "view": row.get("View Position", "").strip().upper() or "UNKNOWN",
            }
    return out


def read_bbox_images(bbox_csv: Path | None) -> set[str]:
    if bbox_csv is None:
        return set()
    with Path(bbox_csv).open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return {row[0].strip() for row in reader if row and row[0].strip()}


def build_rows(
    entries: dict[str, dict],
    train_val: Iterable[str],
    test: Iterable[str],
    bbox_images: set[str],
    cfg: SplitConfig = SplitConfig(),
) -> list[dict]:
    tv, te = list(train_val), list(test)
    tv_set, te_set = set(tv), set(te)
    if len(tv) != len(tv_set) or len(te) != len(te_set):
        raise LeakageError("duplicate image ids inside an official list")
    both = tv_set & te_set
    if both:
        raise LeakageError(f"{len(both)} images are in BOTH train_val and test lists, e.g. {sorted(both)[:3]}")
    unknown = (tv_set | te_set) - set(entries)
    if unknown:
        raise LeakageError(f"{len(unknown)} listed images missing from Data_Entry csv, e.g. {sorted(unknown)[:3]}")
    unlisted = set(entries) - tv_set - te_set
    if unlisted:
        raise LeakageError(f"{len(unlisted)} csv images are in neither official list, e.g. {sorted(unlisted)[:3]}")
    missing_bbox = bbox_images - set(entries)
    if missing_bbox:
        raise LeakageError(f"{len(missing_bbox)} bbox images missing from Data_Entry csv")

    rows = []
    for img in sorted(entries):
        e = entries[img]
        pid = e["patient_id"]
        if img in te_set:
            split = "test"
        else:
            split = "val" if _bucket(pid, cfg.seed, "trainval") < cfg.val_fraction else "train"
        if img in bbox_images:
            if split == "test":
                role = "tuning" if _bucket(pid, cfg.seed, "bbox") < cfg.bbox_tuning_fraction else "reporting"
            else:
                role = "not_in_test"
        else:
            role = "none"
        rows.append(
            {
                "image": img,
                "patient_id": pid,
                "split": split,
                "pneumothorax": int(FINDING in e["labels"]),
                "no_finding": int(e["labels"] == ["No Finding"]),
                "finding_labels": "|".join(e["labels"]),
                "age": e["age"],
                "sex": e["sex"],
                "view": e["view"],
                "has_bbox": int(img in bbox_images),
                "bbox_role": role,
            }
        )
    return rows


def audit_rows(rows: list[dict]) -> dict:
    """Raise LeakageError on any leakage; otherwise return counts for the audit record."""
    patient_splits: dict[str, set[str]] = {}
    patient_bbox_roles: dict[str, set[str]] = {}
    seen: set[str] = set()
    for r in rows:
        if r["image"] in seen:
            raise LeakageError(f"image appears twice in manifest: {r['image']}")
        seen.add(r["image"])
        if r["split"] not in SPLITS:
            raise LeakageError(f"bad split value {r['split']!r}")
        if _patient_from_image(r["image"]) != r["patient_id"]:
            raise LeakageError(f"patient id mismatch for {r['image']}")
        patient_splits.setdefault(r["patient_id"], set()).add(r["split"])
        if r["bbox_role"] in ("tuning", "reporting"):
            patient_bbox_roles.setdefault(r["patient_id"], set()).add(r["bbox_role"])
    bad = {p: s for p, s in patient_splits.items() if len(s) > 1}
    if bad:
        raise LeakageError(f"{len(bad)} patients span multiple splits, e.g. {dict(list(bad.items())[:3])}")
    bad_b = {p: s for p, s in patient_bbox_roles.items() if len(s) > 1}
    if bad_b:
        raise LeakageError(f"{len(bad_b)} patients have images in both bbox tuning and reporting halves")
    counts = {s: {"images": 0, "patients": 0, "pneumothorax_images": 0} for s in SPLITS}
    pats: dict[str, set[str]] = {s: set() for s in SPLITS}
    for r in rows:
        c = counts[r["split"]]
        c["images"] += 1
        c["pneumothorax_images"] += r["pneumothorax"]
        pats[r["split"]].add(r["patient_id"])
    for s in SPLITS:
        counts[s]["patients"] = len(pats[s])
    box_counts: dict[str, int] = {}
    for r in rows:
        box_counts[r["bbox_role"]] = box_counts.get(r["bbox_role"], 0) + 1
    return {"split_counts": counts, "bbox_role_counts": box_counts, "patients_total": len(patient_splits),
            "images_total": len(rows), "leakage_found": False, "label_note": LABEL_NOTE}


def write_manifest(rows: list[dict], audit: dict, out_dir: Path, cfg: SplitConfig, inputs: dict[str, str]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "nih_manifest.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(csv_path)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    record = {"manifest_sha256": digest, "config": {"seed": cfg.seed, "val_fraction": cfg.val_fraction,
              "bbox_tuning_fraction": cfg.bbox_tuning_fraction}, "input_file_sha256": inputs, "audit": audit}
    (out_dir / "nih_split_audit.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return csv_path


def load_manifest(csv_path: Path) -> list[dict]:
    """Read a manifest back AND re-run the leakage audit (fail closed)."""
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("pneumothorax", "no_finding", "has_bbox"):
            r[k] = int(r[k])
    audit_rows(rows)
    return rows


def select_split(rows: list[dict], split: str, *, allow_test: bool = False) -> list[dict]:
    """Return rows of one split. The locked test split requires an explicit allow_test=True."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if split == "test" and not allow_test:
        raise PermissionError("locked test split: pass allow_test=True only after thresholds are frozen")
    return [r for r in rows if r["split"] == split]
