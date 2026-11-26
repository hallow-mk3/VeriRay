#!/usr/bin/env python
"""Verify local offline 4-bit MedGemma 4B loading, image+text inference, and peak VRAM on RTX 5070."""

from __future__ import annotations

from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests.conftest import synthetic_xray  # noqa: E402
from veriray.contracts import EvidenceRecord, Verdict  # noqa: E402
from veriray.vlm import MedGemmaRunner, VLMConfig, run_grounded_vlm_explainer  # noqa: E402


def main() -> int:
    t0 = time.perf_counter()
    runner = MedGemmaRunner.load(VLMConfig(model_dir=ROOT / "models" / "medgemma", quantization="4bit", max_new_tokens=512))
    load_s = time.perf_counter() - t0
    print(f"Loaded MedGemma 4-bit in {load_s:.2f}s | load_peak_vram_mb={runner.peak_vram_mb:.1f}", flush=True)

    # 1. Image mode claim
    t1 = time.perf_counter()
    try:
        claims = runner.claim_on_image(synthetic_xray(224, 224))
        print(f"Image claim ({time.perf_counter() - t1:.2f}s): {claims}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Image claim output/error ({time.perf_counter() - t1:.2f}s): {type(exc).__name__}: {exc}", flush=True)

    # 2. Grounded text explainer + Claim Verifier
    ev = EvidenceRecord(
        record_id="vram-test",
        finding_id="pneumothorax",
        verdict=Verdict.MODEL_SUPPORTED,
        reason_codes=("JUDGE_ALL_GATES_PASSED",),
        items={
            "E01_MODEL_SCORE_RAW": 0.74,
            "E02_VERDICT": "MODEL-SUPPORTED",
            "E03_REASON_CODES": ["JUDGE_ALL_GATES_PASSED"],
            "E04_TARGET_SCORE_DROP_PRIMARY": 0.24,
            "E06_CONTROL_COUNT": 19,
            "E07_CONTROL_SCORE_DROP_MEDIAN": 0.015,
        },
    )
    t2 = time.perf_counter()
    rep = run_grounded_vlm_explainer(ev, runner.generate_text)
    print(
        f"Grounded explainer ({time.perf_counter() - t2:.2f}s): total={rep.total_candidates} "
        f"dropped={rep.dropped_count} fallback={rep.used_template_fallback}",
        flush=True,
    )
    for s in rep.verified_sentences:
        print("  VERIFIED:", s.text, s.evidence_ids, flush=True)

    peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else 0.0
    runner.unload()
    after_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0) if torch.cuda.is_available() else 0.0
    print(f"Peak VRAM: {peak_mb:.1f} MB | After unload: {after_mb:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
