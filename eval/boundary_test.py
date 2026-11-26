#!/usr/bin/env python
"""Experiment 11: Boundary robustness evaluation (Blueprint 26 Nov §9, §12).

Tests corrupt, non-X-ray, tiny, huge, rotated/extreme-aspect, low-contrast, blank,
and missing-weights inputs against ``run_verification`` and records the exact share
ending in a defined fail-closed ``INDETERMINATE`` outcome.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veriray.contracts import Verdict  # noqa: E402
from veriray.pipeline import run_verification  # noqa: E402
from veriray.reader import Reader  # noqa: E402


def run_boundary_suite(*, models_dir: Path = ROOT / "models", empty_dir: Path | None = None) -> dict[str, Any]:
    reader = Reader.load(models_dir)
    rng = np.random.default_rng(20261126)

    cases: list[tuple[str, Any, dict[str, Any]]] = [
        ("corrupt_bytes", b"not-a-valid-image-stream", {"filename_hint": "bad.png", "reader": reader}),
        ("unsupported_extension", b"%PDF-1.4", {"filename_hint": "report.pdf", "reader": reader}),
        ("non_xray_color_scene", rng.integers(0, 255, size=(300, 300, 3), dtype=np.uint8), {"reader": reader}),
        ("tiny_dimensions", np.full((64, 64), 120, dtype=np.uint8), {"reader": reader}),
        ("huge_dimensions", np.full((9000, 300), 120, dtype=np.uint8), {"reader": reader}),
        ("extreme_aspect_ratio", np.tile(np.linspace(20, 220, 300, dtype=np.uint8), (900, 1)), {"reader": reader}),
        ("too_dark_underexposed", np.full((300, 300), 5, dtype=np.uint8), {"reader": reader}),
        ("blank_low_contrast", np.full((300, 300), 128, dtype=np.uint8), {"reader": reader}),
        ("nan_pixels", np.full((300, 300), np.nan, dtype=np.float32), {"reader": reader}),
        ("lateral_view_metadata", np.tile(np.linspace(20, 220, 300, dtype=np.uint8), (300, 1)),
         {"raw_metadata": {"view": "LATERAL"}, "reader": reader}),
    ]
    if empty_dir is not None:
        cases.append(("missing_model_weights", np.full((300, 300), 120, dtype=np.uint8),
                      {"models_dir": empty_dir, "reader": None}))

    results = []
    passed = 0
    for name, src, kw in cases:
        res = run_verification(src, manifest_dir=None, **kw)
        ok = res.verdict is Verdict.INDETERMINATE and bool(res.reason_codes)
        passed += int(ok)
        results.append({"case": name, "verdict": res.verdict.value, "reason_codes": list(res.reason_codes), "fail_closed": ok})

    total = len(results)
    return {
        "total_boundary_cases": total,
        "defined_fail_closed_count": passed,
        "fail_closed_rate": passed / float(max(1, total)),
        "cases": results,
    }


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rep = run_boundary_suite(empty_dir=Path(td))
    out = ROOT / "artifacts" / "boundary_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(f"Boundary suite: {rep['defined_fail_closed_count']}/{rep['total_boundary_cases']} fail-closed ({rep['fail_closed_rate']:.1%}) -> {out}")
