# Model Card â€” Reader (Blueprint v2.0 Â§5) â€” research prototype, not for clinical use

> Research and competition prototype only. Not for diagnosis, triage, treatment or clinical use. Outputs are *model scores*, never the probability that a patient has disease.

## Checkpoint
| Field | Value |
|---|---|
| Weights ID | `densenet121-res224-chex` (torchxrayvision 1.5.7) |
| File | `models/chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt` (git-ignored) |
| Source URL | https://github.com/mlmed/torchxrayvision/releases/download/v1/chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt |
| SHA-256 | `1a914ae802170187a4852e594708206c12468eeeefd66aa4e8e26141649d61a4` |
| Provenance record | `models/CHECKPOINT.json` (written by `scripts/fetch_weights.py`) |
| Architecture | DenseNet-121, 1 input channel, 18 outputs |
| Training data (per torchxrayvision file name) | **CheXpert only**. Not NIH ChestX-ray14. |
| Label mapping | Output index 3 = `Pneumothorax` (CheXpert-labeler style label). Full list in `CHECKPOINT.json`; blank entries are unused outputs. |
| Input | 224 Ã— 224, single channel, scaled to â‰ˆ[âˆ’1024, 1024] via `xrv.utils.normalize`, after a center crop (see `docs/PROTOCOL.md` Â§3) |
| Output | `sigmoid(logit)` for Pneumothorax; uncalibrated; no threshold |
| Licence | torchxrayvision code is Apache-2.0. **No separate weights licence was found; CheXpert data terms may apply to the weights. UNVERIFIED â€” confirm before any redistribution.** |

## Why this checkpoint
We evaluate on NIH ChestX-ray14, so a checkpoint **not trained on NIH** was preferred (Blueprint Â§5 pretraining-overlap audit). Candidates with a Pneumothorax label and no NIH in training: PadChest (`pc`), CheXpert (`chex`), MIMIC-CXR (`mimic_nb`, `mimic_ch`). `chex` was picked because it is single-source (simplest provenance statement), its label definition comes from the CheXpert labeler, and it is not one of the NIH-derived multi-dataset models. This is a PROVISIONAL judgement call, not a measured result. The 26 Nov blueprint says the primary Reader is to be chosen by pneumothorax AUROC on the NIH *validation* split only, written down before the test set is opened; that comparison has not been run, so this choice may change.

## Known limitations / disclosures
- The training-dataset list is taken from the published file name; it was **not independently audited**. Overlap with future external evaluation sets (e.g. VinDr-CXR) has **not** been audited. If provenance remains unclear, evaluation must not be described as fully independent.
- CheXpert labels are report-mined (weak) labels. The model score reflects that label definition, which may differ from a radiologist's pneumothorax definition.
- Performance on NIH, VinDr-CXR or any other set is **unmeasured**. No calibration has been done.
- The checkpoint is a pickled PyTorch module; it is unpickled only after its SHA-256 matches the recorded hash.
- The Reader has not been validated for phone photographs or Indian hospital populations.
