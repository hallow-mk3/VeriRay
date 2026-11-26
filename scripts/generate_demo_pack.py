#!/usr/bin/env python
"""Generate honestly labelled precomputed demo bundles for the 3-minute stage demo (Blueprint §4, §12).

Covers all four verdicts: MODEL-SUPPORTED, SHORTCUT RISK, NOT SUPPORTED, INDETERMINATE.
Run:
    python scripts/generate_demo_pack.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veriray.contracts import SYNTHETIC_COUNTERFACTUAL_LABEL, EvidenceRecord, Verdict  # noqa: E402
from veriray.eraser import erase, render_labelled  # noqa: E402
from veriray.explainer import explain_verified  # noqa: E402
from veriray.pipeline import overlay_heatmap  # noqa: E402


def _synthetic_chest(seed: int, size: int = 224) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    lung_l = np.exp(-(((xx - size * 0.34) / 34) ** 2 + ((yy - size * 0.48) / 56) ** 2))
    lung_r = np.exp(-(((xx - size * 0.66) / 34) ** 2 + ((yy - size * 0.48) / 56) ** 2))
    img = 165.0 - 85.0 * (lung_l + lung_r) + rng.normal(0, 4.0, size=(size, size))
    return np.clip(img, 0, 255).astype(np.float32)


def generate_demo_pack(out_dir: Path = ROOT / "artifacts" / "demo_cases") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32)

    specs = [
        {
            "case_id": "case_a_model_supported",
            "title": "Case A — MODEL-SUPPORTED (synthetic demo example)",
            "verdict": Verdict.MODEL_SUPPORTED,
            "reason_codes": ("JUDGE_ALL_GATES_PASSED",),
            "center": (75, 148),
            "score_raw": 0.7420,
            "target_drop": 0.2410,
            "control_median": 0.0145,
            "tail_frac": 0.05,
            "stability_iou": 0.87,
            "drift": 0.006,
        },
        {
            "case_id": "case_b_shortcut_risk",
            "title": "Case B — SHORTCUT RISK (out-of-lung cue; explanation withheld)",
            "verdict": Verdict.SHORTCUT_RISK,
            "reason_codes": ("SHORTCUT_OUT_OF_LUNG_ATTRIBUTION",),
            "center": (32, 32),
            "score_raw": 0.6810,
            "target_drop": 0.1950,
            "control_median": 0.0180,
            "tail_frac": 0.05,
            "stability_iou": 0.81,
            "drift": 0.008,
        },
        {
            "case_id": "case_c_not_supported",
            "title": "Case C — NOT SUPPORTED (target drop indistinguishable from controls)",
            "verdict": Verdict.NOT_SUPPORTED,
            "reason_codes": ("TARGET_DROP_BELOW_THRESHOLD", "MARGIN_OVER_CONTROLS_BELOW_THRESHOLD"),
            "center": (118, 80),
            "score_raw": 0.4120,
            "target_drop": 0.0150,
            "control_median": 0.0140,
            "tail_frac": 0.45,
            "stability_iou": 0.79,
            "drift": 0.005,
        },
        {
            "case_id": "case_d_indeterminate",
            "title": "Case D — INDETERMINATE (fail-closed on weak/diffuse localization)",
            "verdict": Verdict.INDETERMINATE,
            "reason_codes": ("LOCALIZATION_DIFFUSE", "JUDGE_LOCATOR_WEAK_OR_UNSTABLE"),
            "center": (112, 112),
            "score_raw": 0.3050,
            "target_drop": None,
            "control_median": None,
            "tail_frac": None,
            "stability_iou": None,
            "drift": None,
        },
    ]

    index = []
    for i, sp in enumerate(specs):
        cdir = out_dir / sp["case_id"]
        cdir.mkdir(parents=True, exist_ok=True)
        img = _synthetic_chest(seed=10 + i)
        cy, cx = sp["center"]
        hm = np.clip(1.0 - np.hypot(yy - cy, xx - cx) / 42.0, 0.0, 1.0).astype(np.float32)
        mask = np.hypot(yy - cy, xx - cx) <= 24.0

        Image.fromarray(img.astype(np.uint8)).save(cdir / "original.png")
        Image.fromarray(overlay_heatmap(img, hm)).save(cdir / "overlay.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(cdir / "mask.png")

        has_edit = sp["verdict"] is not Verdict.INDETERMINATE
        if has_edit:
            edit = erase(img, mask, "telea", seed=i)
            Image.fromarray(render_labelled(edit.edited_image, scale=2)).save(cdir / "counterfactual_labelled.png")

        ev = EvidenceRecord(
            record_id=f"demo-{sp['case_id']}",
            finding_id="pneumothorax",
            verdict=sp["verdict"],
            reason_codes=sp["reason_codes"],
            items={
                "E01_MODEL_SCORE_RAW": sp["score_raw"],
                "E02_VERDICT": sp["verdict"].value,
                "E03_REASON_CODES": list(sp["reason_codes"]),
                "E04_TARGET_SCORE_DROP_PRIMARY": sp["target_drop"],
                "E06_CONTROL_COUNT": 19 if has_edit else 0,
                "E07_CONTROL_SCORE_DROP_MEDIAN": sp["control_median"],
                "E08_EMPIRICAL_TAIL_FRACTION": sp["tail_frac"],
                "E11_MASK_MIN_STABILITY_IOU": sp["stability_iou"],
                "E12_UNRELATED_LABEL_MAX_DRIFT": sp["drift"],
            },
        )
        expl = explain_verified(ev)
        bundle = {
            "case_id": sp["case_id"],
            "title": sp["title"],
            "precomputed_notice": "PRECOMPUTED DEMO OUTPUT (synthetic example — not a real patient scan)",
            "synthetic_label": SYNTHETIC_COUNTERFACTUAL_LABEL,
            "verdict": sp["verdict"].value,
            "reason_codes": list(sp["reason_codes"]),
            "has_counterfactual": has_edit,
            "evidence_items": dict(ev.items),
            "verified_explanation": [
                {"text": s.text, "evidence_ids": list(s.evidence_ids)} for s in expl.sentences
            ],
        }
        (cdir / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
        index.append({"case_id": sp["case_id"], "title": sp["title"], "verdict": sp["verdict"].value})

    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return out_dir


if __name__ == "__main__":
    p = generate_demo_pack()
    print(f"Wrote precomputed demo pack to {p}")
