# Data Card — VeriRay v2.0 (research prototype, not for clinical use)

> Research and educational demonstration on public, de-identified chest X-ray examples only.

## Current status
Patient-disjoint NIH ChestX-ray14 splits are recorded in `data/manifests/nih_manifest.csv` and `data/manifests/nih_split_audit.json`. Validation tuning (`configs/frozen.yaml`, `models/platt_calibrator.json`) and locked test evaluation (`artifacts/eval_test_locked/test_evidence_pack.json`) have been executed locally. All uploaded images in the live app are held in memory only and never saved to disk.

## Planned sources (Blueprint §10)
| Role | Dataset | Notes |
|---|---|---|
| Development | NIH ChestX-ray14 | Subject to dataset terms. Labels are report-mined **weak labels**; "No Finding" is **not** a certified healthy cohort. |
| External evaluation | VinDr-CXR (to be considered) | Only after label taxonomy, view and task compatibility are audited. No training or threshold tuning on it. Do not force a mapping where definitions are not comparable. |
| Capture robustness (optional) | Phone photographs of displayed public X-rays | Controlled engineering experiment only; not evidence for clinical or Indian-hospital performance. |

## Split and leakage rules
- Split by **patient**, not image. Group phone photos and derived versions with their source image.
- Manifest fields (protected manifest for identifiers): dataset source, patient identifier, split, view, label mapping, annotation availability, preprocessing hash.
- Freeze model selection, preprocessing and thresholds before opening locked-test results. The locked test set is never used for tuning.
- Audit pretrained-checkpoint provenance and possible overlap (see `docs/MODEL_CARD.md`: the chosen checkpoint was trained on CheXpert, not NIH; overlap with VinDr-CXR is unaudited).

## Privacy
Public de-identified data only. Everything is processed locally; no image or metadata is sent to any external API. Only allowlisted metadata keys are carried through the pipeline. Burned-in pixel identifiers may exist and are only flagged, not removed. Uploads are not retained beyond the session. No legal or regulatory compliance is claimed.

## NIH split procedure (implemented in `src/veriray/splits.py`; run via `scripts/build_nih_manifest.py`)
- Uses NIH's official `train_val_list.txt` / `test_list.txt` (official test -> `test`).
- `train_val_list` is split **by patient** into `train` / `val` (seeded SHA-256 hash, seed 20261126, 20 % val). The seed is recorded in `data/manifests/nih_split_audit.json` together with input-file hashes and the manifest hash.
- Box images in the test split are split **by patient** into a `tuning` half and a `reporting` half; never tune on `reporting`. Box images outside the test split get `not_in_test`.
- The audit fails closed (`LeakageError`, no manifest) if a patient spans two splits, an image is in both lists, or images are missing/unlisted. `load_manifest` re-runs the audit, and `select_split(..., "test")` raises unless `allow_test=True`.
- Status: `data/manifests/nih_manifest.csv` and `data/manifests/nih_split_audit.json` have been generated and audited against `C:\Users\swast\nih`.
