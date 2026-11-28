# VeriRay (Version 2.0 — Research Prototype)

> **IMPORTANT: Research and competition prototype only. Not for diagnosis, triage, treatment or clinical use.**

VeriRay stress-tests whether a medical imaging classifier's prediction is sensitive to a proposed image region under a documented perturbation protocol, matched comparison regions, integrity checks, and uncertainty-aware abstention. It does not prove that a region contains disease or that a diagnosis is correct.

## Non-Negotiable Constraints (Blueprint v2.0)
- **Verified Thoracic Findings (`SUPPORTED_FINDINGS`):** Restricted strictly to the **3 findings** that passed the frozen admission rule (`positives >= 100`, `box/mask Dice >= 0.50`, `no cross-source confound`, and `border_probe_auroc <= 0.80`) on **both** `dev_val` and `locked_test`: `pneumothorax`, `effusion`, and `pneumonia`. (Unverified heads remain excluded from verification contracts; `tuberculosis` was dropped entirely due to severe sub-source framing confound.)
- **Allowed Verdicts Only:** `MODEL-SUPPORTED`, `NOT SUPPORTED`, `SHORTCUT RISK`, `INDETERMINATE`. Never `PROVEN`, `DISEASE CONFIRMED`, `NORMAL`, `CLEAR`, `RULED OUT`, or diagnostic/triage/urgency language.
- **Local / Offline Only:** Zero patient images or metadata are sent to external APIs; live uploads and NIfTI volumes are decoded strictly in memory.
- **Fail Closed:** Any input, localization, reconstruction, stability, or runtime failure returns `INDETERMINATE` with explicit machine-readable reason codes.
- **Exact Outside-Mask Pixel Preservation:** Every synthetic edit preserves outside-mask pixels identically and carries `SYNTHETIC COUNTERFACTUAL — NOT A REAL SCAN`.
- **De-Confounded Patient/Cluster Splits:** Cluster-grouped (`dHash` + `32×32` cosine) chest splits and PID/cluster-grouped brain splits keep locked `test` untouched during model, preprocessing, and threshold selection.

## Repository Layout (Appendix B)
```text
veriray/
  app/             # Streamlit UI (Chest X-ray Judge + Brain MRI explorer + Evidence Pack)
  api/             # FastAPI REST service (GET /health, POST /verify, POST /verify-batch, POST /verify-brain, POST /chat)
  configs/         # frozen Judge/Locator config (frozen.yaml) + frozen admission rule (verifiable.yaml)
  data/manifests/  # split audit JSONs and SHA-256 digests (large CSV manifests gitignored)
  models/          # SHA-256-verified local checkpoints (ChestHD 512px, BrainHD 256px, PSPNet, Gated U-Net, MedGemma)
  src/veriray/     # contracts, gate, reader, chest_hd, brain_mri, locator, eraser, unet_eraser, controls, judge, pipeline, explainer, verifier, safety, vlm
  eval/            # metrics, risk_coverage, ablations, boundary_test, shortcut_audit, load_test
  redteam/         # primary (150 attacks) and independent audit (130 attacks) suites & runners
  tests/           # 188 unit, integration, security, and fail-closed tests
  docs/            # protocol, model card, data card, claims & demo policy
  artifacts/       # locked-test evidence packs, risk-coverage plots, demo cases, latency/boundary/red-team reports
  README.md
  requirements.txt
```

## Full Pipeline Status (Implemented & Evaluated on Locked Test Splits)
**Gate → 512×512 ConvNeXt-Tiny + FPN Reader/Locator → Eraser (`telea`, `gaussian_blur`, `mean_fill`, optional `gated_unet`) + Matched Controls (`K=19`) → Verification Judge → Grounded Explainer + Claim Verifier + Safety Guardrails**

Measured locked-test highlights (`artifacts/eval_test_locked/`):
- **Chest X-Ray Full Judge (`test_evidence_pack.json`, `n=445` zero-shot NIH locked test):**
  - Reader AUROC `0.8941` (`AUPRC 0.9062`, `Brier 0.1402`).
  - Full Judge achieved **98.08% precision** (`51/52`, `95% CI [0.9394, 1.0000]`) at **11.69% coverage** vs. **96.15%** (`50/52`, `95% CI [0.9012, 1.0000]`) for Reader-only at matched coverage (`+1.92%` delta; overlapping CIs).
  - Reporting-half expert bounding-box localization (`n=51`): **76.47% pointing-game accuracy** (`39/51`), **49.02% IoU>=0.10 success rate**, mean IoU `0.1540`.
  - Experiment 5 (MedGemma 4B 4-bit Claim Verifier): unsupported claim rate reduced from `87.5%` before verifier to **0.0%** after verifier.
- **Chest HD Locked Test (`chest_hd_locked_test.json`, `n=7,721` scans) & Zero-Shot External NIH Test (`external_nih_test.json`, `n=25,596` scans):**
  - `pneumothorax` (`pixel_mask_plus_bbox`): within-source AUROC `0.9475` (SIIM) / `0.9833` (VinBigData), mask+box Dice `0.5191`, border probe `0.5778`; external NIH AUROC `0.8829` (`68.37%` pointing game).
  - `effusion` (`bbox_ellipse`): within-source AUROC `0.9831` (VinBigData), box Dice `0.6006`, border probe `0.7683`; external NIH AUROC `0.7752` (`69.93%` pointing game).
  - `pneumonia` (`bbox_ellipse`): within-source AUROC `0.8966` (RSNA), box Dice `0.6528`, border probe `0.7099`; external NIH AUROC `0.7041` (`85.83%` pointing game, mean bbox IoU `0.4530`).
- **Brain MRI HD Within-Source Locked Test (`brain_mri_locked_test.json`, `n=1,378` slices across `197` patient/cluster groups, `Tr-aug` synthetic copies purged):**
  - Within-ISLES (`ischemic_stroke` vs `notumor` slices from ISLES): AUROC `0.8315` (`0.8516` relative), stroke pixel-mask Dice `0.5181`, false-positive stroke rate on normal ISLES slices `33.83%`.
  - Within-BraTS (`glioma` vs `notumor` slices from BraTS): AUROC `0.8719`, glioma mask Dice `0.4405`.
  - Within-2D (`glioma`, `meningioma`, `notumor`, `pituitary`): 4-class accuracy `85.95%` (Figshare tumor mask Dice: meningioma `0.8041`, pituitary `0.7937`, glioma `0.5600`).
- **Safety & Boundary Suites:** Primary red-team `150/150 (100.0%)` (`artifacts/redteam_report.json`), independent audit red-team `130/130 (100.0%)` end-to-end (`70.8%` Layer 1 alone; `100.0%` homoglyph obfuscation; `artifacts/audit_redteam_report.json`), boundary fail-closed rate `1.0` (`11/11`, `artifacts/boundary_report.json`).

## Known Limitations
- **Selective abstention:** On the NIH locked test sample (`n=445`), the Judge accepts `11.69%` (`52/445`) as `MODEL-SUPPORTED` and abstains or rejects on `88.31%` (`NOT SUPPORTED`: `334`, `INDETERMINATE`: `42`, `SHORTCUT RISK`: `17`), with overlapping 95% CIs against Reader-only at matched coverage.
- **Fill-method sensitivity:** `30.69%` verdict flip rate (`58/189` evaluated cases) across `telea`, `gaussian_blur`, and `mean_fill`, gated by the Judge's fill-spread check.
- **Box vs. pixel-mask supervision:** On Chest HD, `effusion` and `pneumonia` spatial supervision comes from rasterized radiologist bounding boxes (`bbox_ellipse`), whereas only `pneumothorax` (SIIM-ACR) includes exact pixel-contour masks.
- **Excluded findings & border framing cues:** `cardiomegaly` (`0.8506` test border-probe AUROC) and `aortic_enlargement` (`0.8649`) were excluded from `SUPPORTED_FINDINGS` because border/aspect-ratio framing cues inside VinBigData predicted them above the `0.80` cutoff; `infiltration` was excluded after dropping to `95 < 100` locked-test positives.
- **Brain MRI modality scope:** Brain MRI mode provides within-source model scores, FPN/Grad-CAM++ candidate masks, and Telea counterfactual drops, not the full 7-gate thoracic Verification Judge; cortical MRI sequence differences (`T1ce` vs `DWI`) remain detectable inside parenchyma (`78.85%` at `14%` depth), so only within-source metrics are valid headlines.

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
188 tests pass on Python 3.12 with the pinned dependencies in `requirements.txt`.
