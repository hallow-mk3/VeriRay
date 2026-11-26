# VeriRay (Version 2.0 â€” Research Prototype)

> **IMPORTANT: Research and competition prototype only. Not for diagnosis, triage, treatment or clinical use.**

VeriRay tests whether a chest X-ray classifier's prediction for **pneumothorax** is sensitive to a proposed image region under a documented perturbation protocol, matched comparison regions, integrity checks, and uncertainty-aware abstention. It does not prove that a region contains disease or that a diagnosis is correct.

## Non-Negotiable Constraints (Blueprint v2.0)
- **Single Finding:** `pneumothorax` only.
- **Allowed Verdicts Only:** `MODEL-SUPPORTED`, `NOT SUPPORTED`, `SHORTCUT RISK`, `INDETERMINATE`. Never `PROVEN`, `DISEASE CONFIRMED`, `NORMAL`, `CLEAR`, `RULED OUT`, or diagnostic/triage/urgency language.
- **Local / Offline Only:** Zero patient images or metadata are sent to external APIs.
- **Fail Closed:** Any input, localization, reconstruction, stability, or runtime failure returns `INDETERMINATE` with explicit machine-readable reason codes.
- **Exact Outside-Mask Pixel Preservation:** Every synthetic edit preserves outside-mask pixels identically and carries `SYNTHETIC COUNTERFACTUAL â€” NOT A REAL SCAN`.
- **Patient-Level Splits:** Locked test set remains untouched during model, preprocessing, and threshold selection.

## Repository Layout (Appendix B)
```text
veriray/
  app/             # Streamlit UI
  configs/         # model, preprocessing, Judge and threshold configs
  data/manifests/  # split, label, source and capture manifests
  models/          # gate, reader, locator, eraser checkpoints
  src/veriray/     # contracts, manifest (gate, cleaner [unused], reader, locator, pipeline; eraser/judge/safety later)
  eval/            # metrics, risk_coverage, ablations, robustness
  tests/           # unit, integration, leakage and fail-closed tests
  docs/            # protocol, model card, data card, ethics, demo
  artifacts/       # versioned plots, configs and approved demo outputs
  README.md
  requirements.txt
```

## Current Status: Step 2 (Reader + Grad-CAM++ on one image)
- Step 1: layout, pinned requirements, typed contracts, run-manifest writer.
- Step 2: Input Gate (`gate.py`), Reader (`reader.py`, torchxrayvision DenseNet-121, CheXpert-trained, local weights only), Grad-CAM++ Locator (`locator.py`), `pipeline.py`, and a minimal Streamlit page. Parameters: `docs/PROTOCOL.md`. Checkpoint: `docs/MODEL_CARD.md`.
- `cleaner.py` (Capture Cleaner) exists but is deferred and imported nowhere.
- Not built: Eraser, Judge, matched controls, U-Net, narration, calibration.

## Install (Python 3.12, from the repository root)
Create a virtual environment, then install **one** of:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# CPU-only PyTorch:
python -m pip install -r requirements-cpu.txt
# OR NVIDIA GPU PyTorch (CUDA 12.8 wheels, verified on the RTX 5070 with torch 2.11.0+cu128; run `python scripts/check_gpu.py` afterwards):
python -m pip install -r requirements-gpu.txt
```

`requirements.txt` pins everything except the PyTorch build variant (no `+cpu` local-version pins).

## Get the weights (the ONLY step that uses the network)
```powershell
python scripts/fetch_weights.py --list   # list checkpoints and their training datasets (no download)
python scripts/fetch_weights.py          # downloads densenet121-res224-chex into models/, writes models/CHECKPOINT.json
```
Everything else runs offline. The app and library refuse to run (INDETERMINATE) if the weights file is missing or its SHA-256 differs from `models/CHECKPOINT.json`.

## Run the app
```powershell
streamlit run app/streamlit_app.py
```
Upload one public, de-identified chest X-ray. The page shows the gate result, the original, the Grad-CAM++ overlay, the candidate mask, the model score and checkpoint ID. Score and heatmap describe model behaviour, not a diagnosis. Research prototype — not for clinical use.

## Run the tests
```powershell
python -m pytest
```
