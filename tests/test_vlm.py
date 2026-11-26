"""Tests for MedGemma offline adapter, strict JSON claim parsing, grounded VLM explainer, and fail-closed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from veriray.contracts import EvidenceRecord, Verdict
from veriray.vlm import (
    VLMConfig,
    VLMError,
    MedGemmaRunner,
    box_mask_iou,
    build_grounded_explainer_prompt,
    parse_explainer_candidates,
    parse_vlm_claims,
    run_grounded_vlm_explainer,
)


def _ev() -> EvidenceRecord:
    return EvidenceRecord(
        record_id="ev-vlm",
        finding_id="pneumothorax",
        verdict=Verdict.MODEL_SUPPORTED,
        reason_codes=("JUDGE_ALL_GATES_PASSED",),
        items={
            "E01_MODEL_SCORE_RAW": 0.78,
            "E02_VERDICT": "MODEL-SUPPORTED",
            "E03_REASON_CODES": ["JUDGE_ALL_GATES_PASSED"],
            "E04_TARGET_SCORE_DROP_PRIMARY": 0.24,
            "E06_CONTROL_COUNT": 19,
            "E07_CONTROL_SCORE_DROP_MEDIAN": 0.02,
        },
    )


def test_medgemma_fails_closed_when_local_dir_missing(tmp_path: Path):
    with pytest.raises(VLMError) as exc:
        MedGemmaRunner.load(VLMConfig(model_dir=tmp_path / "missing_medgemma"))
    assert exc.value.reason_code == "VLM_CHECKPOINT_MISSING"


def test_parse_vlm_claims_valid_and_invalid_json():
    raw = """```json
    {"claims": [
      {"finding": "Pneumothorax", "status": "present", "region_description": "right apex", "box_xyxy": [0.1, 0.1, 0.5, 0.5]},
      {"finding": "Effusion", "status": "uncertain", "region_description": "left base"}
    ]}
    ```"""
    claims = parse_vlm_claims(raw)
    assert len(claims) == 2
    assert claims[0].is_verifiable_finding and not claims[1].is_verifiable_finding

    for bad in ("", "not json at all", '{"claims": [{"finding": "x", "status": "confirmed", "region_description": "y"}]}'):
        with pytest.raises(VLMError):
            parse_vlm_claims(bad)


def test_grounded_vlm_explainer_drops_hallucinations_and_passes_clean_sentences():
    import re

    ev = _ev()
    prompt = build_grounded_explainer_prompt(ev)
    assert "E01_MODEL_SCORE_RAW" in prompt and not re.search(r"\b(?:age|sex|patient)\b", prompt.lower())

    def mock_vlm_with_one_hallucination(_prompt: str) -> str:
        return json.dumps({
            "sentences": [
                {"text": "The Reader model score was 0.78.", "evidence_ids": ["E01_MODEL_SCORE_RAW"]},
                {"text": "The patient has a 99% chance of disease and needs 500 mg aspirin.", "evidence_ids": ["E01_MODEL_SCORE_RAW"]},
            ]
        })

    rep = run_grounded_vlm_explainer(ev, mock_vlm_with_one_hallucination)
    assert rep.total_candidates == 2 and rep.dropped_count == 1
    assert not rep.used_template_fallback
    assert len(rep.verified_sentences) == 1 and "0.78" in rep.verified_sentences[0].text

    # Malformed JSON from VLM falls back cleanly to fixed templates and counts the failure
    rep_bad = run_grounded_vlm_explainer(ev, lambda _p: "unparseable gibberish")
    assert rep_bad.used_template_fallback and len(rep_bad.verified_sentences) == 1
    assert rep_bad.total_candidates == 1 and rep_bad.dropped_count == 1 and rep_bad.unsupported_claim_rate == 1.0


def test_box_mask_iou_and_parse_explainer_candidates():
    mask = np.zeros((100, 100), bool)
    mask[10:50, 10:50] = True
    assert box_mask_iou(mask, (0.1, 0.1, 0.5, 0.5)) == pytest.approx(1.0)
    assert box_mask_iou(mask, (0.6, 0.6, 0.9, 0.9)) == pytest.approx(0.0)

    cands = parse_explainer_candidates('[{"text": "Score 0.78.", "evidence_ids": ["E01_MODEL_SCORE_RAW"]}]')
    assert len(cands) == 1 and not cands[0].verified
    with pytest.raises(VLMError):
        parse_explainer_candidates('[{"text": "missing ids"}]')


def test_vlm_module_has_no_network_library():
    import re

    src = (Path(__file__).resolve().parents[1] / "src" / "veriray" / "vlm.py").read_text(encoding="utf-8")
    assert not re.search(r"^\s*(?:import|from)\s+(?:urllib|requests|socket|http)", src, re.M)
    assert "local_files_only=True" in src
