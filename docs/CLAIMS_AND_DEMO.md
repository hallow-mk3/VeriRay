# Claims and Demo Policy (Blueprint v2.0 §12.3, §17, §18)

> Research and competition prototype only. Not for diagnosis, triage, treatment or clinical use.

## Positioning (one line)
VeriRay stress-tests a chest X-ray model's prediction using controlled perturbations and matched controls, measures instability and abstains when the evidence is insufficient — without claiming to diagnose disease.

## Claims allowed — only if supported by measured results (§18)
Each claim may be made only using measured numbers in `artifacts/eval_test_locked/test_evidence_pack.json`, `artifacts/boundary_report.json`, and `artifacts/redteam_report.json`.
- On the held-out NIH locked test sample (`n=445`), the full Judge achieved **94.12% precision** at **3.82% coverage** vs. **88.24%** for Reader-only at matched coverage (`+5.88%` precision delta).
- Matched controls and stability/integrity gates abstained on `96.18%` of cases (`INDETERMINATE`: `241`, `NOT SUPPORTED`: `123`, `SHORTCUT RISK`: `64`, `MODEL-SUPPORTED`: `17`).
- The input-quality gate abstained on controlled boundary/capture failure cases (measured in `artifacts/boundary_report.json`).
- The shortcut audit flagged out-of-lung attribution (`64/445` on locked test).

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
- Every edited image is labelled **SYNTHETIC COUNTERFACTUAL — NOT A REAL SCAN** (no edits exist yet).
- Mandatory on-screen text in the interface: "Score and heatmap describe model behaviour, not a diagnosis. Research prototype — not for clinical use."

## Demo rules (§12.3)
Use 3–5 precomputed public, de-identified examples plus a backup video; label precomputed outputs honestly; run a live case only if local hardware is stable. Finish with the *measured* Reader-versus-Judge risk-coverage comparison and the limitations. Phone photographs are a controlled capture-robustness pilot, not evidence of performance in Indian hospitals.

## Current honest status
Full pipeline (Gate -> Reader -> Grad-CAM++ -> Eraser + Matched Controls -> Judge -> Grounded Explainer + Claim Verifier) is implemented, tested offline, and evaluated on the locked NIH test split (`artifacts/eval_test_locked/test_evidence_pack.json`). Grad-CAM++ remains a candidate region proposal and scores are model behaviour, not diagnoses.
