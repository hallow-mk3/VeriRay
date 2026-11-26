"""Tests for matched controls, Verification Judge, full pipeline, and new 26 Nov contracts."""

from __future__ import annotations

import socket

import numpy as np
import pytest

from tests.conftest import synthetic_xray
from veriray.contracts import EvidenceRecord, ExplanationSentence, LocatorContract, VLMClaim, Verdict
from veriray.controls import ControlConfig, generate_matched_controls
from veriray.judge import JudgeConfig, run_judge
from veriray.locator import LocatorConfig
from veriray.pipeline import run_verification
from veriray.reader import Reader


def _compact_locator(cy: int = 112, cx: int = 112, r: float = 28.0) -> LocatorContract:
    yy, xx = np.mgrid[0:224, 0:224].astype(np.float32)
    hm = np.clip(1.0 - np.hypot(yy - cy, xx - cx) / (r * 1.6), 0.0, 1.0)
    mask = np.hypot(yy - cy, xx - cx) <= r
    return LocatorContract(
        heatmap=hm,
        target_mask=mask,
        localization_flags={
            "empty": False, "diffuse": False, "implausibly_large": False,
            "near_full_image": False, "anatomically_implausible": False, "unstable": False,
        },
    )


def test_matched_controls_satisfy_overlap_area_and_determinism():
    loc = _compact_locator()
    cfg = ControlConfig(k=19, min_controls=8, max_target_overlap=0.10, area_tolerance=0.25, seed=7)
    r1 = generate_matched_controls(loc.target_mask, cfg)
    r2 = generate_matched_controls(loc.target_mask, cfg)
    assert r1.valid and len(r1.masks) == 19
    assert all(np.array_equal(a, b) for a, b in zip(r1.masks, r2.masks))
    assert all(ov < 0.10 for ov in r1.overlaps_with_target)
    assert all(0.75 <= ratio <= 1.25 for ratio in r1.area_ratios)


def test_matched_controls_fail_closed_on_huge_or_empty_target():
    empty = generate_matched_controls(np.zeros((224, 224), bool))
    assert not empty.valid and empty.reason_codes == ("CONTROLS_EMPTY_TARGET",)
    huge = np.zeros((224, 224), bool)
    huge[:, :180] = True  # leaves nowhere for non-overlapping same-area copies
    res = generate_matched_controls(huge, ControlConfig(k=19, min_controls=8))
    assert not res.valid and res.reason_codes == ("INSUFFICIENT_MATCHED_CONTROLS",)


def test_judge_returns_model_supported_when_target_drop_dominates_controls(fake_reader: Reader, monkeypatch):
    img = synthetic_xray(224, 224, seed=3).astype(np.float32)
    base_rc, _, _ = fake_reader.score(img)
    loc = _compact_locator()

    # Controlled Reader response: target edit drops score by 0.25; control edits drop by ~0.01.
    orig_score = fake_reader.score

    def controlled_score(arr):
        rc, x, disp = orig_score(arr)
        if np.array_equal(arr, img):
            return rc, x, disp
        changed = arr != img
        overlap_target = float(np.logical_and(changed, loc.target_mask).sum()) / max(1, int(loc.target_mask.sum()))
        fake_val = max(0.0, base_rc.score_raw - (0.25 if overlap_target > 0.5 else 0.01))
        from veriray.contracts import ReaderContract

        return (
            ReaderContract(fake_val, None, rc.threshold_id, rc.checkpoint_id, rc.preprocessing_id, rc.unrelated_scores),
            x,
            disp,
        )

    monkeypatch.setattr(fake_reader, "score", controlled_score)
    out = run_judge(fake_reader, img, base_rc, loc, cfg=JudgeConfig(num_controls=19, min_controls=8))
    assert out.contract.verdict is Verdict.MODEL_SUPPORTED
    assert all(out.contract.gate_results.values())
    assert out.evidence_record.items["E08_EMPIRICAL_TAIL_FRACTION"] == pytest.approx(1.0 / 20.0)
    assert out.primary_target_edit is not None and out.primary_target_edit.integrity_report.outside_mask_exact_equality


def test_judge_returns_not_supported_when_target_drop_matches_controls(fake_reader: Reader, monkeypatch):
    img = synthetic_xray(224, 224, seed=3).astype(np.float32)
    base_rc, _, _ = fake_reader.score(img)
    loc = _compact_locator()
    orig_score = fake_reader.score

    def flat_score(arr):
        rc, x, disp = orig_score(arr)
        from veriray.contracts import ReaderContract

        # Zero score drop on both target and controls => adequate test, NOT SUPPORTED
        return (
            ReaderContract(base_rc.score_raw, None, rc.threshold_id, rc.checkpoint_id, rc.preprocessing_id, rc.unrelated_scores),
            x,
            disp,
        )

    monkeypatch.setattr(fake_reader, "score", flat_score)
    out = run_judge(fake_reader, img, base_rc, loc, cfg=JudgeConfig(num_controls=12, min_controls=8))
    assert out.contract.verdict is Verdict.NOT_SUPPORTED
    assert "TARGET_DROP_BELOW_THRESHOLD" in out.contract.reason_codes


def test_judge_returns_shortcut_risk_when_attribution_is_outside_lung(fake_reader: Reader):
    img = synthetic_xray(224, 224, seed=3).astype(np.float32)
    base_rc, _, _ = fake_reader.score(img)
    loc = _compact_locator(cy=20, cx=20, r=16)  # top-left corner
    lung_mask = np.zeros((224, 224), bool)
    lung_mask[60:200, 60:180] = True  # target mask is 100% outside lung_mask
    out = run_judge(
        fake_reader, img, base_rc, loc,
        cfg=JudgeConfig(num_controls=10, min_controls=8, max_boundary_discontinuity=1.0),
        lung_mask=lung_mask,
    )
    assert out.contract.verdict is Verdict.SHORTCUT_RISK
    assert "SHORTCUT_OUT_OF_LUNG_ATTRIBUTION" in out.contract.reason_codes


def test_judge_fails_closed_on_unstable_mask_and_fill_spread(fake_reader: Reader, monkeypatch):
    img = synthetic_xray(224, 224, seed=3).astype(np.float32)
    base_rc, _, _ = fake_reader.score(img)
    # Steeply spiking mask: threshold + 0.05 wipes it out => unstable
    hm = np.zeros((224, 224), np.float32)
    hm[100:130, 100:130] = 0.52
    mask = hm >= 0.5
    unstable_loc = LocatorContract(hm, mask, {"empty": False, "diffuse": False, "implausibly_large": False})
    out = run_judge(fake_reader, img, base_rc, unstable_loc, locator_cfg=LocatorConfig())
    assert out.contract.verdict is Verdict.INDETERMINATE and out.contract.reason_codes == ("JUDGE_MASK_UNSTABLE",)

    # Now test fill-method spread exceeding max_fill_method_spread
    good_loc = _compact_locator()
    orig_score = fake_reader.score
    calls = {"n": 0}

    def wildly_varying_score(arr):
        rc, x, disp = orig_score(arr)
        calls["n"] += 1
        val = max(0.0, min(1.0, base_rc.score_raw - (0.6 if calls["n"] == 1 else 0.0)))
        from veriray.contracts import ReaderContract

        return ReaderContract(val, None, rc.threshold_id, rc.checkpoint_id, rc.preprocessing_id, rc.unrelated_scores), x, disp

    monkeypatch.setattr(fake_reader, "score", wildly_varying_score)
    out2 = run_judge(fake_reader, img, base_rc, good_loc, cfg=JudgeConfig(max_fill_method_spread=0.10))
    assert out2.contract.verdict is Verdict.INDETERMINATE and out2.contract.reason_codes == ("JUDGE_FILL_METHOD_INSTABILITY",)


def test_full_verification_pipeline_offline_and_manifest(monkeypatch, fake_models_dir, tmp_path):
    def boom(*a, **k):
        raise AssertionError("network disabled")

    monkeypatch.setattr(socket, "socket", boom)
    res = run_verification(
        synthetic_xray(320, 320),
        models_dir=fake_models_dir,
        manifest_dir=tmp_path / "runs",
        judge_cfg=JudgeConfig(num_controls=10, min_controls=8),
    )
    assert res.verdict in {Verdict.MODEL_SUPPORTED, Verdict.NOT_SUPPORTED, Verdict.SHORTCUT_RISK, Verdict.INDETERMINATE}
    assert res.output is not None and res.manifest_path is not None and res.manifest_path.exists()
    assert "not for diagnosis" in res.output.disclaimer.lower()


def test_26_nov_contracts_vlm_evidence_explanation():
    claim = VLMClaim("Pneumothorax", "present", "right apex", (0.1, 0.1, 0.5, 0.5))
    assert claim.is_verifiable_finding and claim.finding == "pneumothorax"
    with pytest.raises(ValueError):
        VLMClaim("pneumothorax", "confirmed", "right apex")
    with pytest.raises(ValueError):
        VLMClaim("pneumothorax", "present", "right apex", (0.6, 0.2, 0.3, 0.5))

    ev = EvidenceRecord("ev-1", "pneumothorax", Verdict.NOT_SUPPORTED, ("TARGET_DROP_BELOW_THRESHOLD",), {"E01": 0.12})
    assert ev.items["E01"] == 0.12

    sent = ExplanationSentence("The model score was 0.12 and the test did not support it.", ("E01",), verified=True)
    assert sent.verified
    with pytest.raises(ValueError):
        ExplanationSentence("No evidence cited.", (), verified=True)
    with pytest.raises(ValueError):
        ExplanationSentence("You have a diagnosis of pneumothorax.", ("E01",), verified=True)


def test_judge_and_screening_summary_on_low_score_scan(fake_reader: Reader):
    from veriray.contracts import ReaderContract
    from veriray.reader import summarize_reader_findings

    img = synthetic_xray(224, 224, seed=4).astype(np.float32)
    low_rc = ReaderContract(
        score_raw=0.04,
        score_calibrated_optional=None,
        threshold_id="t0",
        checkpoint_id=fake_reader.checkpoint_id,
        preprocessing_id=fake_reader.preprocessing_id,
        unrelated_scores={"Atelectasis": 0.08, "Effusion": 0.05},
    )
    summary = summarize_reader_findings(low_rc, target_threshold=0.70, other_threshold=0.50)
    assert summary.all_scores_low and not summary.target_flagged and summary.flagged_other_findings == ()

    other_high_rc = ReaderContract(
        score_raw=0.04,
        score_calibrated_optional=None,
        threshold_id="t0",
        checkpoint_id=fake_reader.checkpoint_id,
        preprocessing_id=fake_reader.preprocessing_id,
        unrelated_scores={"Atelectasis": 0.08, "Effusion": 0.72},
    )
    summary2 = summarize_reader_findings(other_high_rc, target_threshold=0.70, other_threshold=0.50)
    assert not summary2.all_scores_low and summary2.flagged_other_findings == (("Effusion", 0.72),)

    empty_loc = LocatorContract(
        heatmap=np.zeros((224, 224), np.float32),
        target_mask=np.zeros((224, 224), bool),
        localization_flags={"empty": True, "diffuse": False, "implausibly_large": False},
    )
    out = run_judge(
        fake_reader,
        img,
        low_rc,
        empty_loc,
        cfg=JudgeConfig(min_reader_score=0.70),
    )
    assert out.contract.verdict is Verdict.NOT_SUPPORTED
    assert "READER_SCORE_BELOW_THRESHOLD" in out.contract.reason_codes
    assert "NO_FINDING_FLAGGED_BY_READER" in out.contract.reason_codes
