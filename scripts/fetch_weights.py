#!/usr/bin/env python
"""Download ONE torchxrayvision checkpoint into models/ and record its provenance.

THIS IS THE ONLY FILE IN THE REPOSITORY ALLOWED TO TOUCH THE NETWORK (Blueprint
v2.0 Â§10.4, Â§15.1). It downloads model weights only; no image or metadata is
ever sent anywhere. Runtime code (src/veriray, app/) loads from models/ only.

Usage (from the repository root):
    python scripts/fetch_weights.py --list
    python scripts/fetch_weights.py                       # default: densenet121-res224-chex
    python scripts/fetch_weights.py --weights densenet121-res224-pc
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

import torchxrayvision as xrv

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
FINDING_LABEL = "Pneumothorax"
DEFAULT_WEIGHTS_ID = "densenet121-res224-chex"

# Dataset tokens as used in torchxrayvision weight file names -> human-readable names.
DATASET_NAMES = {
    "nih": "NIH ChestX-ray14",
    "pc": "PadChest",
    "chex": "CheXpert",
    "mimic_ch": "MIMIC-CXR (CheXpert-labeler labels)",
    "mimic_nb": "MIMIC-CXR (NegBio labels)",
    "google": "Google (token as named by torchxrayvision; not audited)",
    "openi": "OpenI",
    "kaggle": "RSNA Pneumonia (Kaggle)",
    "rsna": "RSNA Pneumonia",
    "siim": "SIIM-ACR Pneumothorax",
    "vin": "VinDr-CXR",
}

PROVENANCE_NOTES = (
    "Training-dataset list is derived from the weight file name published by torchxrayvision "
    "(file-name prefix tokens); it was not independently audited. Overlap with any evaluation "
    "dataset must still be audited per Blueprint section 5 and 10.3. If provenance is unclear, "
    "disclose it and avoid claims of fully independent evaluation."
)
LICENSE_NOTE = (
    "torchxrayvision code: Apache-2.0 (package metadata). No separate licence file for the "
    "weights was located by this script; the terms of the training datasets (e.g. the CheXpert "
    "research-use agreement) may apply to the weights. UNVERIFIED - confirm before any redistribution."
)


def training_datasets(weights_id: str) -> list[str]:
    url = xrv.models.model_urls[weights_id]["weights_url"]
    stem = url.rsplit("/", 1)[-1]
    prefix = stem.split("-densenet121")[0].split("-resnet50")[0]
    tokens = prefix.split("-")
    return [t for t in tokens if t]


def list_checkpoints() -> list[dict]:
    rows = []
    for key, info in xrv.models.model_urls.items():
        if not key.startswith("densenet121-res224"):
            continue
        toks = training_datasets(key)
        rows.append(
            {
                "weights_id": key,
                "datasets": toks,
                "has_pneumothorax": FINDING_LABEL in info["labels"],
                "trained_on_nih": "nih" in toks,
            }
        )
    return rows


def print_listing() -> None:
    print(f"{'weights_id':32} {'PTX label':10} {'NIH in training':16} training datasets")
    for r in list_checkpoints():
        names = ", ".join(DATASET_NAMES.get(t, t) for t in r["datasets"])
        print(
            f"{r['weights_id']:32} {str(r['has_pneumothorax']):10} {str(r['trained_on_nih']):16} {names}"
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print checkpoints and exit (no network)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS_ID)
    ap.add_argument("--medgemma", action="store_true", help="download gated MedGemma into models/medgemma (requires HF login & licence)")
    ap.add_argument("--medgemma-id", default="google/medgemma-4b-it")
    ap.add_argument("--lung-seg", action="store_true", help="download ChestX-Det PSPNet lung segmentation weights into models/")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args()

    if args.lung_seg:
        url = "https://github.com/mlmed/torchxrayvision/releases/download/v1/pspnet_chestxray_best_model_4.pth"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dest = MODELS_DIR / url.rsplit("/", 1)[-1]
        if not dest.exists() or args.force:
            part = dest.with_suffix(dest.suffix + ".part")
            print(f"Downloading lung segmentation weights from {url}")
            with urllib.request.urlopen(url, timeout=120) as resp, part.open("wb") as out:  # noqa: S310
                while chunk := resp.read(1024 * 1024):
                    out.write(chunk)
            part.replace(dest)
        digest = sha256_file(dest)
        (MODELS_DIR / "LUNG_SEG_CHECKPOINT.json").write_text(
            json.dumps(
                {
                    "model_id": "chestx-det-pspnet",
                    "filename": dest.name,
                    "source_url": url,
                    "license": "torchxrayvision Apache-2.0; ChestX-Det research dataset terms may apply (UNVERIFIED).",
                    "sha256": digest,
                    "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"SHA-256 {digest}\nWrote {MODELS_DIR / 'LUNG_SEG_CHECKPOINT.json'}")
        return 0

    if args.medgemma:
        from huggingface_hub import snapshot_download

        dest_dir = MODELS_DIR / "medgemma"
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {args.medgemma_id} into {dest_dir} (weights only; requires accepted Health AI Developer Foundations licence)...")
        snapshot_download(repo_id=args.medgemma_id, local_dir=str(dest_dir))
        hashes = {
            p.relative_to(dest_dir).as_posix(): sha256_file(p)
            for p in sorted(dest_dir.rglob("*"))
            if p.is_file() and p.name != "CHECKPOINT.json"
        }
        (dest_dir / "CHECKPOINT.json").write_text(
            json.dumps(
                {
                    "model_id": args.medgemma_id,
                    "license": "Health AI Developer Foundations Terms (gated on Hugging Face)",
                    "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                    "file_sha256": hashes,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {dest_dir / 'CHECKPOINT.json'} ({len(hashes)} files hashed)")
        return 0

    print_listing()
    if args.list:
        return 0

    if args.weights not in xrv.models.model_urls:
        print(f"ERROR: unknown weights id {args.weights!r}", file=sys.stderr)
        return 2
    info = xrv.models.model_urls[args.weights]
    labels = list(info["labels"])
    if FINDING_LABEL not in labels:
        print(f"ERROR: {FINDING_LABEL!r} is not in the label list of {args.weights}.", file=sys.stderr)
        return 3
    toks = training_datasets(args.weights)
    if "nih" in toks:
        print(
            f"ERROR: {args.weights} was trained on NIH ChestX-ray14, which we evaluate on. "
            "Refusing; choose a checkpoint without NIH.",
            file=sys.stderr,
        )
        return 4
    print(f"\nChosen: {args.weights}; {FINDING_LABEL!r} is at label index {labels.index(FINDING_LABEL)}.")

    url = info["weights_url"]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / url.rsplit("/", 1)[-1]
    if dest.exists() and not args.force:
        print(f"Already present, not re-downloading: {dest}")
    else:
        part = dest.with_suffix(dest.suffix + ".part")
        print(f"Downloading weights only (no user data) from {url}")
        with urllib.request.urlopen(url, timeout=120) as resp, part.open("wb") as out:  # noqa: S310
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        part.replace(dest)

    digest = sha256_file(dest)
    record = {
        "weights_id": args.weights,
        "filename": dest.name,
        "source_url": url,
        "license": LICENSE_NOTE,
        "training_datasets": [DATASET_NAMES.get(t, t) for t in toks],
        "training_dataset_tokens": toks,
        "label_list": labels,
        "finding_label": FINDING_LABEL,
        "finding_index": labels.index(FINDING_LABEL),
        "input_resolution": int(info.get("input_resolution", 224)),
        "sha256": digest,
        "size_bytes": dest.stat().st_size,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "torchxrayvision_version": getattr(xrv, "__version__", "unknown"),
        "provenance_notes": PROVENANCE_NOTES,
    }
    (MODELS_DIR / "CHECKPOINT.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"SHA-256 {digest}\nWrote {MODELS_DIR / 'CHECKPOINT.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
