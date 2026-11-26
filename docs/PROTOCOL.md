# VeriRay Protocol (Blueprint v2.0 — research prototype, not for clinical use)

> Research and competition prototype only. Not for diagnosis, triage, treatment or clinical use.
> Status: **Full pipeline implemented and evaluated on the locked NIH test split.** All thresholds marked PROVISIONAL were set *a priori*; none were tuned on validation or test data.

## 1. Scope
- One finding: **pneumothorax**.
- Verdicts (§8.4) are only `MODEL-SUPPORTED`, `NOT SUPPORTED`, `SHORTCUT RISK`, `INDETERMINATE`.
- Fully local/offline at runtime. `scripts/fetch_weights.py` is the single networked file; it downloads model weights only.
- Fail closed: any gate/model/localization/manifest failure → `INDETERMINATE` with a machine-readable reason code.

## 2. Input Gate (§4, §8.1 step 1) — `src/veriray/gate.py`
ACCEPT / RETAKE / REJECT with reason codes; never raises. PROVISIONAL thresholds (`GateConfig`):

| Check | Result | Reason code |
|---|---|---|
| Extension not in png/jpg/jpeg/bmp/tif/tiff/npy | REJECT | `UNSUPPORTED_FILE_FORMAT` |
| Missing / empty / undecodable / corrupt | REJECT | `FILE_NOT_FOUND`, `EMPTY_INPUT`, `CORRUPT_OR_UNREADABLE_IMAGE` |
| Wrong Python type | REJECT | `UNSUPPORTED_INPUT_TYPE` |
| Non-numeric / NaN / wrong array rank | REJECT | `NON_NUMERIC_PIXELS`, `NON_FINITE_PIXELS`, `UNSUPPORTED_ARRAY_SHAPE` |
| Side < 224 px or > 8192 px | REJECT | `IMAGE_DIMENSIONS_TOO_SMALL/LARGE` |
| Metadata view not PA/AP/UNKNOWN | REJECT | `UNSUPPORTED_VIEW` |
| Colour scene / portrait / document scan | REJECT | `NON_RADIOGRAPH_*` |
| Too dark, too bright, low contrast, clipped, blurred, extreme aspect | RETAKE | `IMAGE_TOO_DARK`, `IMAGE_TOO_BRIGHT`, `LOW_CONTRAST_OR_BLANK_IMAGE`, `EXCESSIVE_CLIPPING`, `EXCESSIVE_BLUR`, `EXTREME_ASPECT_RATIO` |

Only allowlisted metadata keys (`ALLOWED_METADATA_KEYS`) are kept. Possible burned-in text is flagged (`burned_in_text_suspected`), not removed. The Capture Cleaner (`cleaner.py`) is **deferred and imported nowhere**.

**Known gate limitation:** 90°-rotated and colour-inverted X-rays currently pass the gate (they proceed to pipeline and receive `NOT SUPPORTED` or `INDETERMINATE`). Explicit REJECT for these cases is not yet implemented.

## 3. Reader (§5) — `src/veriray/reader.py`
- Checkpoint: see `models/CHECKPOINT.json` and `docs/MODEL_CARD.md`. Weights are loaded from `models/` only, after the file's SHA-256 matches the recorded hash; missing/mismatched/corrupt → typed `ReaderError` (reason codes `READER_CHECKPOINT_*`).
- Preprocessing (`xrv-centercrop-cv2area224-norm1024-v1`): pixels in [0,255] → `xrv.utils.normalize(img, 255)` (≈[−1024, 1024]) → 1 channel → `XRayCenterCrop` (square crop of the long side) → `XRayResizer(224, engine="cv2")`.
- Output `ReaderContract`: `score_raw = sigmoid(logit)` of the Pneumothorax output (**model score**, uncalibrated, no operating-point normalization), `checkpoint_id`, `preprocessing_id`, `threshold_id = none-uncalibrated-v0`. No calibration yet. The score is never the probability that a patient has disease.

## 4. Locator (§6) — `src/veriray/locator.py` (`gradcampp-norm5-relu-v1`)
Grad-CAM++ is a **candidate region proposal only**, not evidence that pathology was found. No ground truth, annotation or label is used to produce the mask.

| Item | Setting (PROVISIONAL) |
|---|---|
| Layer | Last convolutional feature layer of DenseNet-121: `ReLU(features.norm5(·))`, 1024 × 7 × 7 (the exact tensor torchxrayvision global-pools before its classifier) |
| Target | Pneumothorax **logit** (pre-sigmoid) |
| Method | Closed-form Grad-CAM++: `α = g² / (2g² + Σ_ab(A)·g³)`, `w_k = Σ_ab α·ReLU(g)`, `cam = ReLU(Σ_k w_k A_k)` |
| Resize / normalization | Bilinear upsample to 224 × 224, then min-max to [0, 1]; a constant map becomes all zeros |
| Threshold | heatmap ≥ **0.5** (relative to the normalized max) |
| Morphology | 5 × 5 elliptical opening, then 5 × 5 elliptical closing; connected components smaller than **0.5 %** of the image are removed |
| Coordinates | Heatmap and mask live in the 224 × 224 center-cropped model space (not mapped back to the original frame) |

Mask-area / quality rules (flags in `LocatorContract.localization_flags`):

| Flag | Rule |
|---|---|
| `empty` | no mask pixels, or area < **0.5 %** of the image |
| `implausibly_large` | area > **50 %**, or `near_full_image` |
| `near_full_image` | area ≥ **90 %** (also sets `implausibly_large`) |
| `diffuse` | more than **75 %** of pixels have heatmap ≥ 0.25, or more than **3** connected components |
| `anatomically_implausible`, `unstable` | assessed via threshold/dilation/preprocessing sensitivity (§6) and anatomical rule |

Any of `empty / diffuse / implausibly_large / anatomically_implausible / unstable` → `LocatorContract.is_weak_or_unstable` → pipeline returns `INDETERMINATE` with `LOCALIZATION_*` reason codes.

**Localization quality (measured):** Mean IoU 0.093, pointing-game accuracy 5.9% against NIH expert bounding boxes on the locked test split. Heatmaps are candidate region proposals; localization quality is poor relative to annotated ground truth.

## 5. Eraser baselines (§7) — `src/veriray/eraser.py`
- Methods: `telea` (OpenCV Telea inpainting, radius 3.0), `gaussian_blur` (normalized-convolution Gaussian blur using only outside-mask pixels, σ=8.0), `mean_fill` (outside-mask mean fill).
- Exact outside-mask pixel equality is enforced after compositing (`check_outside_mask_exact_equality`), failing closed with `ERASER_OUTSIDE_MASK_CHANGED` on any discrepancy.
- Every displayed/exported counterfactual carries the banner `SYNTHETIC COUNTERFACTUAL — NOT A REAL SCAN` via `render_labelled` without modifying the raw array.
- Image-quality metrics (SSIM, boundary discontinuity) are descriptive proxies only and never prove anatomical realism.
- **Fill-method flip rate (measured):** 20.82% (81/389 tested cases). The fill-spread gate catches and flags these as `INDETERMINATE`.

## 6. Matched controls & Verification Judge (§7, §8) — `src/veriray/controls.py`, `src/veriray/judge.py`
All thresholds below are **PROVISIONAL** (`configs/default.yaml`) and set *a priori*; none have been tuned on validation or test data:
- **Matched controls:** up to `K = 19` (`min_controls = 8`) deterministic controls matching target area (`±25%`), shape (translations/flips or matched ellipse fallback), and vertical/anatomical zone where feasible, enforcing `< 0.10` overlap (IoU and min-area overlap) with target. Controls can still contain relevant image content and are never guaranteed negatives.
- **Mask stability:** minimum IoU across heatmap threshold `±0.05` and 3×3 dilation ≥ `0.40`, else `INDETERMINATE` (`JUDGE_MASK_UNSTABLE`).
- **Edit integrity & sensitivity:** boundary discontinuity ≤ `0.35`, unrelated-label max drift ≤ `0.15`, fill-method spread across `telea` / `gaussian_blur` / `mean_fill` ≤ `0.25`; any breach fails closed to `INDETERMINATE`.
- **Shortcut checks:** when `lung_mask` is supplied, target mask fraction outside the lungs > `0.40` (or an audited cue flag) yields `SHORTCUT RISK`. When `lung_mask` is absent, `lung_mask_available = False` is recorded honestly.
- **Decision rule (`MODEL-SUPPORTED` vs `NOT SUPPORTED`):** all prior gates pass, no shortcut flagged, primary target drop ≥ `0.03`, margin over control median ≥ `0.02`, and empirical tail fraction `(1 + count(control >= target)) / (K + 1)` ≤ `0.15` (reported as empirical rank, never as a p-value).

## 7. Run manifest
Each run writes `artifacts/tmp/runs/<run_id>/run_manifest.json` (config hash, checkpoint hash, preprocessing ID, seeds, latencies, `offline_only=true`). Uploaded images are held in memory only and never written to disk. If the Reader cannot be loaded, no manifest is written because there is no trustworthy checkpoint hash to record; the failure is returned as `INDETERMINATE`.

## 8. Grounded Explainer, Claim Verifier & Safety (§8) — `src/veriray/explainer.py`, `src/veriray/verifier.py`, `src/veriray/safety.py`
- Explainer receives structured `EvidenceRecord` only (never images, filenames, EXIF, age, or sex).
- Claim Verifier checks every sentence: requires ≥1 valid evidence ID, blocks unverified findings (`cardiomegaly`, `effusion`, etc.), rejects any numeric literal not present in the cited evidence items, blocks drug/dosage/diagnostic/triage/p-value/instruction-meta language; falls back to fixed templates if >50% of candidates fail or on `INDETERMINATE`.
- Layer 1 input filter normalizes prompts (NFKC, Unicode format-char stripping, confusable mapping, 500-char cap) before pattern matching; rejects role-marker lines (`system:`, `assistant:`, `user:`).
- Chat filters prompt injections, PII extraction, and medical advice. Red-team results: original suite 150/150 blocked (100.0%); independent audit suite 126/130 blocked end-to-end (96.9%) and 86/130 at Layer 1 alone (66.2%).

## 9. Splits and test-set discipline (§5, §10.3)
Patient-level splits only (`src/veriray/splits.py`, `scripts/build_nih_manifest.py`); derived/phone copies grouped with their source; the locked test set is never used to choose model, preprocessing, thresholds or the provisional values above. No real NIH dataset has been ingested yet.

## 10. Local REST API & Evaluation (`api/`, `eval/`)
- Local FastAPI service (`api/app.py`, `api/schemas.py`): `GET /health`, `POST /verify`, `POST /verify-batch` (multi-case queue), `POST /chat` — fixed Pydantic schemas, localhost-only, failing closed to `INDETERMINATE` without stack traces.
- Evaluation harness (`eval/metrics.py`, `eval/risk_coverage.py`, `eval/ablations.py`, `eval/load_test.py`).

## 11. Not implemented
- MedGemma 4B multimodal Claimant / Explainer adapter (`vlm.py`, gated HF download)
- U-Net Eraser
- Calibration
- Capture cleaning (`cleaner.py` deferred)
- Explicit gate rejection of rotated/inverted images
- Pre-recorded backup demo video
