#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veriray.contracts import EvidenceRecord, Verdict  # noqa: E402
from veriray.splits import load_manifest, select_split  # noqa: E402
from veriray.vlm import MedGemmaRunner, VLMConfig, build_grounded_explainer_prompt  # noqa: E402

rows = load_manifest(ROOT / "data" / "manifests" / "nih_manifest.csv")
pos = next(r for r in select_split(rows, "val") if r["pneumothorax"] == 1)
img_path = next(Path(r"C:\Users\swast\nih").glob(f"images_*/images/{pos['image']}"))
arr = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)

runner = MedGemmaRunner.load(VLMConfig(model_dir=ROOT / "models" / "medgemma", quantization="4bit", max_new_tokens=256))
print("Real NIH image:", pos["image"], flush=True)
try:
    print("Claims:", runner.claim_on_image(arr), flush=True)
except Exception as exc:  # noqa: BLE001
    print("Claim error:", type(exc).__name__, exc, flush=True)

ev = EvidenceRecord(
    "t",
    "pneumothorax",
    Verdict.MODEL_SUPPORTED,
    ("JUDGE_ALL_GATES_PASSED",),
    {
        "E01_MODEL_SCORE_RAW": 0.74,
        "E02_VERDICT": "MODEL-SUPPORTED",
        "E03_REASON_CODES": ["JUDGE_ALL_GATES_PASSED"],
        "E04_TARGET_SCORE_DROP_PRIMARY": 0.24,
        "E06_CONTROL_COUNT": 19,
        "E07_CONTROL_SCORE_DROP_MEDIAN": 0.015,
    },
)
print("Raw explainer text:", repr(runner.generate_text(build_grounded_explainer_prompt(ev))), flush=True)
runner.unload()
