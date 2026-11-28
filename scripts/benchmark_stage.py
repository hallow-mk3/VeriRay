#!/usr/bin/env python
"""Measure real cold-start, per-stage p50/p95 latencies, and API load/fault stats (Exp 9 & 10)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.load_test import run_api_load_and_fault_test  # noqa: E402
from eval.metrics import summarize_latencies  # noqa: E402
from tests.conftest import synthetic_xray  # noqa: E402
from veriray.anatomy import LungSegmenter  # noqa: E402
from veriray.pipeline import run_verification  # noqa: E402
from veriray.reader import Reader  # noqa: E402


def main() -> int:
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    reader = Reader.load_for_finding(ROOT / "models", "pneumothorax", device=dev)
    reader_cold_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    seg = LungSegmenter.load_optional(ROOT / "models", device=dev)
    seg_cold_ms = (time.perf_counter() - t1) * 1000.0

    img = synthetic_xray(512, 512, seed=7)
    runs = []
    for _ in range(10):
        res = run_verification(img, reader=reader, manifest_dir=None)
        runs.append({"latencies_ms": res.latencies_ms})

    latency_summary = summarize_latencies(runs)
    with tempfile.TemporaryDirectory() as td:
        api_summary = run_api_load_and_fault_test(
            reader=reader,
            missing_models_dir=Path(td),
            n_sequential=6,
            n_concurrent=4,
        )

    report = {
        "hardware": f"local workstation ({dev.upper()} inference for {reader.checkpoint_id} + PSPNet)",
        "cold_start_ms": {
            "reader_load": round(reader_cold_ms, 2),
            "lung_segmenter_load": round(seg_cold_ms, 2),
            "total_cold_start": round(reader_cold_ms + seg_cold_ms, 2),
        },
        "stage_latencies": latency_summary,
        "api_load_and_fault_test": api_summary,
        "lung_segmenter_active": seg is not None,
    }
    out = ROOT / "artifacts" / "latency_and_api_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Cold start: {report['cold_start_ms']}")
    print(f"End-to-end p50/p95: {latency_summary.get('end_to_end')}")
    print(f"API load & faults: {api_summary}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
