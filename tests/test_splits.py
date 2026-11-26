"""Patient-level split and leakage tests (synthetic NIH-format files; no real data needed)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from veriray.splits import (
    LeakageError, SplitConfig, audit_rows, build_rows, load_manifest, read_bbox_images, read_entries,
    select_split, write_manifest,
)

HEADER = ["Image Index", "Finding Labels", "Follow-up #", "Patient ID", "Patient Age", "Patient Gender", "View Position"]


def make_dataset(tmp_path: Path, n_patients=200, per_patient=3):
    rows, tv, te, bbox = [], [], [], []
    for p in range(1, n_patients + 1):
        for k in range(per_patient):
            img = f"{p:08d}_{k:03d}.png"
            lab = "Pneumothorax|Effusion" if (p + k) % 7 == 0 else ("No Finding" if p % 2 else "Atelectasis")
            rows.append([img, lab, k, p, 40 + p % 30, "M" if p % 2 else "F", "PA"])
            (te if p % 5 == 0 else tv).append(img)
            if p % 5 == 0 and k == 0:
                bbox.append(img)
    d = tmp_path / "Data_Entry_2017.csv"
    with d.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(rows)
    (tmp_path / "train_val_list.txt").write_text("\n".join(tv))
    (tmp_path / "test_list.txt").write_text("\n".join(te))
    b = tmp_path / "BBox_List_2017.csv"
    with b.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Image Index", "Finding Label", "Bbox [x", "y", "w", "h]"])
        w.writerows([[i, "Pneumothorax", 1, 2, 3, 4] for i in bbox])
    return d, tmp_path / "train_val_list.txt", tmp_path / "test_list.txt", b


def build(tmp_path, cfg=SplitConfig()):
    d, tv, te, b = make_dataset(tmp_path)
    from veriray.splits import _read_list

    return build_rows(read_entries(d), _read_list(tv), _read_list(te), read_bbox_images(b), cfg)


def test_no_patient_spans_two_splits(tmp_path):
    rows = build(tmp_path)
    audit = audit_rows(rows)
    by_patient = {}
    for r in rows:
        by_patient.setdefault(r["patient_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by_patient.values())
    assert audit["leakage_found"] is False
    assert {r["split"] for r in rows} == {"train", "val", "test"}


def test_official_test_list_is_respected_and_val_is_from_trainval(tmp_path):
    rows = build(tmp_path)
    assert all((r["split"] == "test") == (int(r["patient_id"]) % 5 == 0) for r in rows)
    val = [r for r in rows if r["split"] == "val"]
    assert 0.1 < len(val) / len([r for r in rows if r["split"] != "test"]) < 0.3


def test_deterministic_and_seed_dependent(tmp_path):
    a, b = build(tmp_path), build(tmp_path)
    assert a == b
    c = build(tmp_path, SplitConfig(seed=1))
    assert [r["split"] for r in a] != [r["split"] for r in c]


def test_bbox_halves_are_patient_level_and_only_for_test(tmp_path):
    rows = build(tmp_path)
    roles = {r["patient_id"]: r["bbox_role"] for r in rows if r["has_bbox"]}
    assert set(roles.values()) <= {"tuning", "reporting"}
    assert {"tuning", "reporting"} == set(roles.values())
    assert all(r["bbox_role"] == "none" for r in rows if not r["has_bbox"])


def test_overlapping_official_lists_fail_closed(tmp_path):
    d, tv, te, b = make_dataset(tmp_path)
    leaked = tv.read_text().splitlines()[0]
    te.write_text(te.read_text() + "\n" + leaked)
    from veriray.splits import _read_list

    with pytest.raises(LeakageError):
        build_rows(read_entries(d), _read_list(tv), _read_list(te), read_bbox_images(b))


def test_patient_in_two_splits_is_caught_by_audit(tmp_path):
    rows = build(tmp_path)
    victim = next(r for r in rows if r["split"] == "train")
    twin = next(r for r in rows if r["patient_id"] == victim["patient_id"] and r is not victim)
    twin["split"] = "test"
    with pytest.raises(LeakageError, match="multiple splits"):
        audit_rows(rows)


def test_missing_unlisted_and_duplicate_images_fail_closed(tmp_path):
    d, tv, te, b = make_dataset(tmp_path)
    from veriray.splits import _read_list

    ent = read_entries(d)
    with pytest.raises(LeakageError):
        build_rows(ent, _read_list(tv)[:-1], _read_list(te), set())  # one csv image in neither list
    with pytest.raises(LeakageError):
        build_rows(ent, _read_list(tv) + ["99999999_000.png"], _read_list(te), set())  # listed but not in csv


def test_manifest_roundtrip_reaudits_and_locks_test(tmp_path):
    rows = build(tmp_path)
    cfg = SplitConfig()
    p = write_manifest(rows, audit_rows(rows), tmp_path / "out", cfg, {"x": "0" * 64})
    loaded = load_manifest(p)
    assert len(loaded) == len(rows)
    assert (tmp_path / "out" / "nih_split_audit.json").exists()
    assert select_split(loaded, "train") and select_split(loaded, "val")
    with pytest.raises(PermissionError):
        select_split(loaded, "test")
    assert select_split(loaded, "test", allow_test=True)
    # Tamper with the file: reload must fail the audit.
    text = p.read_text().splitlines()
    header = text[0].split(",")
    i_split, i_patient = header.index("split"), header.index("patient_id")
    first = text[1].split(",")
    target = first[i_patient]
    for n, line in enumerate(text[1:], 1):
        cells = line.split(",")
        if cells[i_patient] == target and n > 1:
            cells[i_split] = "test" if cells[i_split] != "test" else "train"
            text[n] = ",".join(cells)
            break
    p.write_text("\n".join(text) + "\n")
    with pytest.raises(LeakageError):
        load_manifest(p)


def test_labels_derived_from_finding_labels(tmp_path):
    rows = build(tmp_path)
    for r in rows:
        assert r["pneumothorax"] == int("Pneumothorax" in r["finding_labels"].split("|"))
        assert r["no_finding"] == int(r["finding_labels"] == "No Finding")
