"""Tests for precomputed 3-minute stage demo pack."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_demo_pack import generate_demo_pack
from veriray.safety import check_output_safety


def test_generate_demo_pack_covers_all_four_verdicts_safely(tmp_path: Path):
    out = generate_demo_pack(tmp_path / "demo_cases")
    idx = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert {row["verdict"] for row in idx} == {
        "MODEL-SUPPORTED",
        "SHORTCUT RISK",
        "NOT SUPPORTED",
        "INDETERMINATE",
    }
    for row in idx:
        cdir = out / row["case_id"]
        bundle = json.loads((cdir / "bundle.json").read_text(encoding="utf-8"))
        assert "PRECOMPUTED DEMO OUTPUT" in bundle["precomputed_notice"]
        assert (cdir / "original.png").is_file()
        assert (cdir / "overlay.png").is_file()
        assert (cdir / "mask.png").is_file()
        if bundle["has_counterfactual"]:
            assert (cdir / "counterfactual_labelled.png").is_file()
        for sent in bundle["verified_explanation"]:
            assert sent["evidence_ids"]
            assert check_output_safety(sent["text"]).safe
