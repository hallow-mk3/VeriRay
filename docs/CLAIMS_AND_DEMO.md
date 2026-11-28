# Claims and Demo Policy (Blueprint v2.0 §12.3, §17, §18)

> Research and competition prototype only. Not for diagnosis, triage, treatment or clinical use.

## Positioning (one line)
VeriRay stress-tests a chest X-ray model's prediction using controlled perturbations and matched controls, measures instability and abstains when the evidence is insufficient — without claiming to diagnose disease.

## Claims allowed — only if supported by measured results (§18)
Each claim may be made only using measured numbers in `artifacts/eval_test_locked/test_evidence_pack.json`, `artifacts/eval_test_locked/chest_hd_locked_test.json`, `artifacts/eval_test_locked/external_nih_test.json`, `artifacts/eval_test_locked/brain_mri_locked_test.json`, `artifacts/boundary_report.json`, and `artifacts/redteam_report.json`.
- **Supported thoracic findings (`SUPPORTED_FINDINGS`):** Strictly `pneumothorax`, `effusion`, and `pneumonia` — the 3 findings passing `positives >= 100`, `box/mask Dice >= 0.50`, `no cross-source confound`, and `border_probe_auroc <= 0.80` on **both** `dev_val` and `locked_test` (`artifacts/eval_test_locked/chest_hd_locked_test.json`).
- **NIH locked-test Judge comparison (`n=445`, zero-shot external evaluation):** Full Judge achieved **98.08% precision** (`51/52`, `95% CI [0.9394, 1.0000]`) at **11.69% coverage** vs. **96.15%** (`50/52`, `95% CI [0.9012, 1.0000]`) for Reader-only at matched coverage (`+1.92%` delta; overlapping 95% CIs).
- **Abstention & verdict breakdown (`n=445`):** `MODEL-SUPPORTED`: `52`, `NOT SUPPORTED`: `334`, `INDETERMINATE`: `42`, `SHORTCUT RISK`: `17` (`88.31%` non-accepted).
- **Reporting-half expert box localization (`n=51`):** `76.47%` pointing-game accuracy (`39/51`) and `0.1540` mean IoU (`artifacts/eval_test_locked/test_evidence_pack.json`); across full external NIH locked test (`n=25,596`), zero-shot AUROC was `0.8829` (`pneumothorax`), `0.7752` (`effusion`), and `0.7041` (`pneumonia`).
- **Brain MRI within-source locked test (`n=1,378` slices, `Tr-aug` purged):** Within-ISLES stroke-vs-normal AUROC `0.8315` (pixel-mask Dice `0.5181`), Within-BraTS glioma-vs-normal AUROC `0.8719`, and Within-2D 4-class accuracy `85.95%` (`artifacts/eval_test_locked/brain_mri_locked_test.json`).
- **Boundary & Red-team guardrails:** Boundary fail-closed rate **11/11 (100.0%)** (`artifacts/boundary_report.json`); primary red-team **150/150 (100.0%)** (`artifacts/redteam_report.json`); independent audit red-team **130/130 (100.0%)** end-to-end and **92/130 (70.8%)** at Layer 1 alone (`15/15` Unicode homoglyphs blocked at Layer 1; `artifacts/audit_redteam_report.json`); MedGemma 4B Claim Verifier reduced unsupported sentences from `87.5%` to **0.0%**.

## Claims NOT allowed (§18)
- VeriRay proves whether a patient has a disease.
- VeriRay is clinically validated, medically safe or ready for hospital use.
- VeriRay works for Indian patients based only on public datasets and a small phone-photo pilot.
- The generated image shows what the patient's lung would look like without disease.
- The system detects all shortcuts.

## Vocabulary
- Verdicts: `MODEL-SUPPORTED`, `NOT SUPPORTED`, `SHORTCUT RISK`, `INDETERMINATE` only. Never PROVEN, DISEASE CONFIRMED, NORMAL, CLEAR or RULED OUT.
- Say "model score", never "probability of disease". Say "heatmap shows a candidate region", never "shows the disease".
- Do not call empirical control ranks p-values. Anatomical matching does not make a control region a guaranteed negative.
- Every edited image is labelled **SYNTHETIC COUNTERFACTUAL — NOT A REAL SCAN**.
- Mandatory on-screen text in the interface: "Score and heatmap describe model behaviour, not a diagnosis. Research prototype — not for clinical use."

## Demo rules (§12.3)
Use 3–5 precomputed public, de-identified examples plus a backup video; label precomputed outputs honestly; run a live case only if local hardware is stable. Finish with the *measured* Reader-versus-Judge risk-coverage comparison and the limitations. Phone photographs are a controlled capture-robustness pilot, not evidence of performance in Indian hospitals.

## Current honest status
Full pipeline (Gate -> 512×512 ConvNeXt-Tiny+FPN Reader/Locator -> Eraser + Matched Controls -> Judge -> Grounded Explainer + Claim Verifier) is implemented, tested offline (`188/188` pytest), and evaluated once on the locked test splits (`artifacts/eval_test_locked/`), with `SUPPORTED_FINDINGS` restricted strictly to `pneumothorax`, `effusion`, and `pneumonia`. Spatial maps remain candidate region proposals and scores describe model behaviour, never clinical diagnoses.
