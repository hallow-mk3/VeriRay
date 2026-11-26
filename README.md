# VeriRay (Version 2.0 — Research Prototype)

> **IMPORTANT: Research and competition prototype only. Not for diagnosis, triage, treatment or clinical use.**

VeriRay tests whether a chest X-ray classifier's prediction for **pneumothorax** is sensitive to a proposed image region under a documented perturbation protocol, matched comparison regions, integrity checks, and uncertainty-aware abstention. It does not prove that a region contains disease or that a diagnosis is correct.

## Non-Negotiable Constraints (Blueprint v2.0)
- **Single Finding:** `pneumothorax` only.
- **Allowed Verdicts Only:** `MODEL-SUPPORTED`, `NOT SUPPORTED`, `SHORTCUT RISK`, `INDETERMINATE`. Never `PROVEN`, `DISEASE CONFIRMED`, `NORMAL`, `CLEAR`, `RULED OUT`, or diagnostic/triage/urgency language.
- **Local / Offline Only:** Zero patient images or metadata are sent to external APIs.
- **Fail Closed:** Any input, localization, reconstruction, stability, or runtime failure returns `INDETERMINATE` with explicit machine-readable reason codes.
- **Exact Outside-Mask Pixel Preservation:** Every synthetic edit preserves outside-mask pixels identically and carries `SYNTHETIC COUNTERFACTUAL — NOT A REAL SCAN`.
- **Patient-Level Splits:** Locked test set remains untouched during model, preprocessing, and threshold selection.

## Repository Layout (Appendix B)
```text
veriray/
  app/             # Streamlit UI
  api/             # FastAPI REST service (GET /health, POST /verify, POST /verify-batch, POST /chat)
  configs/         # model, preprocessing, Judge and threshold configs
  data/manifests/  # split, label, source and capture manifests
  models/          # gate, reader, locator, eraser checkpoints
  src/veriray/     # contracts, gate, reader, locator, eraser, controls, judge, pipeline, explainer, verifier, safety
  eval/            # metrics, risk_coverage, ablations, robustness, load_test
  redteam/         # attack suites and runners
  tests/           # unit, integration, leakage and fail-closed tests
  docs/            # protocol, model card, data card, ethics, demo
  artifacts/       # versioned plots, configs and approved demo outputs
  README.md
  requirements.txt
```

## Full Pipeline Status (implemented and evaluated)
Full pipeline is implemented and evaluated on the locked NIH test split (`artifacts/eval_test_locked/test_evidence_pack.json`):

**Gate → Reader → Grad-CAM++ Locator → Eraser (3 methods) + Matched Controls (K=19) → Judge → Grounded Explainer + Claim Verifier + Safety guardrails**

All components (`gate.py`, `reader.py`, `locator.py`, `eraser.py`, `controls.py`, `judge.py`, `pipeline.py`, `explainer.py`, `verifier.py`, `safety.py`) are present and tested.

## Known Limitations
- **Coverage:** 17/445 cases accepted as `MODEL-SUPPORTED` (3.82% coverage). The pipeline abstains on the other 96.18% (`INDETERMINATE`: 241, `NOT SUPPORTED`: 123, `SHORTCUT RISK`: 64).
- **Precision comparison:** 94.12% (16/17) vs. 88.24% (15/17) with overlapping 95% CIs ([0.821, 1.0] vs. [0.725, 1.0]). The +5.88% delta is not statistically distinguishable at this sample size.
- **Fill-method flip rate:** 20.82% (81/389 tested cases). Verdicts can change across `telea`, `gaussian_blur`, and `mean_fill` eraser methods, which the protocol flags as `INDETERMINATE` via the fill-spread gate.
- **Rotated/inverted images:** 90°-rotated and colour-inverted X-rays currently pass the input gate (returned as `NOT SUPPORTED` or `INDETERMINATE`, not explicitly `REJECTED`). They should ideally be rejected at the gate stage.
- **Localization quality:** Mean IoU 0.093, pointing-game accuracy 5.9% against expert boxes. Grad-CAM++ provides candidate regions only; localization quality is poor relative to annotated ground truth.
- **No backup video:** A pre-recorded backup demo video does not yet exist in the repository.
- **MedGemma 4B Claimant / U-Net Eraser:** These components (`vlm.py` adapter and U-Net eraser) are not implemented. `cleaner.py` (Capture Cleaner) exists but is deferred and imported nowhere.
- **No real NIH data ingested:** Splits and manifests are defined but no real NIH CXR14 dataset has been ingested. `scripts/fetch_weights.py` is the only networked file.

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
Upload one public, de-identified chest X-ray. The page shows the gate result, the original, the Grad-CAM++ overlay, the candidate mask, the model score, matched controls, judge verdict, and evidence-grounded explanation. Score and heatmap describe model behaviour, not a diagnosis. Research prototype — not for clinical use.

## Run the API
```powershell
uvicorn api.app:app --host 127.0.0.1 --port 8000
# GET  /health
# POST /verify        (single image)
# POST /verify-batch  (multi-case queue)
# POST /chat          (evidence-grounded Q&A)
```

## Run the tests
```powershell
python -m pytest
```
141 tests pass on Python 3.12 with the pinned dependencies in `requirements.txt`.
