"""Step 1 pytest suite verifying Appendix B layout, Section 4.1 contracts, non-negotiables, and RunManifest writer."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import numpy as np
import pytest

from veriray import (
    RESEARCH_DISCLAIMER,
    SUPPORTED_FINDING,
    SYNTHETIC_COUNTERFACTUAL_LABEL,
    VERDICT_USER_LANGUAGE,
    CaptureCleanerContract,
    EraserContract,
    InputContract,
    InputGateContract,
    InputGateStatus,
    IntegrityReport,
    JudgeContract,
    LocatorContract,
    OutputContract,
    ReaderContract,
    Verdict,
    check_outside_mask_exact_equality,
    create_run_manifest,
    read_run_manifest,
    sha256_bytes,
    sha256_file,
    validate_safe_language,
    write_run_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SHA256 = sha256_bytes(b"documented-densenet121-checkpoint-placeholder")


def _sample_manifest():
    return create_run_manifest(
        run_config={"threshold_quantile": 0.85, "num_controls": 19},
        model_id="densenet121-res224-nih",
        checkpoint_hashes={"reader": SAMPLE_SHA256},
        preprocessing_id="xrv_res224_norm1024",
        dataset_split_id="nih_tuning",
        random_seeds={"global": 42, "controls": 20261125},
        stage_latencies_ms={"reader_ms": 35.2},
    )


def test_appendix_b_repository_layout_exists() -> None:
    expected_dirs = [
        "app",
        "configs",
        "data/manifests",
        "models",
        "src/veriray",
        "eval",
        "tests",
        "docs",
        "artifacts",
    ]
    for rel in expected_dirs:
        assert (REPO_ROOT / rel).is_dir(), f"Missing Appendix B directory: {rel}"
    assert (REPO_ROOT / "README.md").is_file()
    assert (REPO_ROOT / "requirements.txt").is_file()


def test_input_contract_enforces_single_finding_and_metadata_allowlist() -> None:
    img = np.zeros((224, 224), dtype=np.float32)
    valid = InputContract(
        image_tensor=img,
        image_metadata_allowlist={"view": "PA", "source_dataset": "NIH_ChestXray14"},
        finding_id="Pneumothorax",
        model_id="densenet121-res224-nih",
        preprocessing_id="xrv_res224_norm1024",
    )
    assert valid.finding_id == SUPPORTED_FINDING

    # Frozen dataclass immutability
    with pytest.raises(FrozenInstanceError):
        valid.finding_id = "edema"  # type: ignore[misc]

    # Unsupported finding rejected
    with pytest.raises(ValueError, match="finding_id"):
        InputContract(
            image_tensor=img,
            image_metadata_allowlist={"view": "PA"},
            finding_id="edema",
            model_id="densenet121-res224-nih",
            preprocessing_id="xrv_res224_norm1024",
        )

    # Unapproved / PII metadata keys rejected
    with pytest.raises(ValueError, match="Disallowed metadata keys"):
        InputContract(
            image_tensor=img,
            image_metadata_allowlist={"patient_name": "Jane Doe"},
            finding_id="pneumothorax",
            model_id="densenet121-res224-nih",
            preprocessing_id="xrv_res224_norm1024",
        )

    # Non-finite image pixels rejected
    bad_img = img.copy()
    bad_img[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        InputContract(
            image_tensor=bad_img,
            image_metadata_allowlist={},
            finding_id="pneumothorax",
            model_id="densenet121-res224-nih",
            preprocessing_id="xrv_res224_norm1024",
        )


def test_input_gate_and_capture_cleaner_contracts() -> None:
    gate = InputGateContract(
        status=InputGateStatus.REJECT,
        reason_codes=("UNSUPPORTED_VIEW_LATERAL",),
        quality_flags={"supported_view": False},
    )
    assert not gate.accepted

    # Empty reason codes must fail
    with pytest.raises(ValueError, match="reason_code"):
        InputGateContract(status=InputGateStatus.RETAKE, reason_codes=())

    # Capture cleaner must retain original & transformed separately (never same object reference)
    orig = np.ones((64, 64), dtype=np.float32)
    transformed = orig.copy() * 0.9
    cc = CaptureCleanerContract(
        original_image=orig,
        transformed_image=transformed,
        transform_log={"deskew_deg": 2.0},
    )
    assert cc.transformed_image is not None
    with pytest.raises(ValueError, match="separately"):
        CaptureCleanerContract(original_image=orig, transformed_image=orig)


def test_reader_and_locator_contracts() -> None:
    reader = ReaderContract(
        score_raw=0.18,
        score_calibrated_optional=None,
        threshold_id="tuning_frozen_v1",
        checkpoint_id="densenet121-res224-nih",
        preprocessing_id="xrv_res224_norm1024",
        unrelated_scores={"effusion": 0.04},
    )
    assert reader.score_raw == pytest.approx(0.18)

    with pytest.raises(ValueError, match="score_raw"):
        ReaderContract(score_raw=1.5, score_calibrated_optional=None, threshold_id="t1")

    hm = np.linspace(0.0, 1.0, 224 * 224, dtype=np.float32).reshape(224, 224)
    mask = hm >= 0.85
    loc_ok = LocatorContract(heatmap=hm, target_mask=mask, localization_flags={"unstable": False})
    assert not loc_ok.is_weak_or_unstable

    loc_unstable = LocatorContract(heatmap=hm, target_mask=mask, localization_flags={"unstable": True})
    assert loc_unstable.is_weak_or_unstable


def test_eraser_outside_mask_exact_pixel_equality_and_synthetic_label() -> None:
    rng = np.random.default_rng(7)
    original = rng.uniform(10.0, 240.0, size=(64, 64)).astype(np.float32)
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:32, 20:40] = True

    # Valid composite keeping outside-mask pixels bit-identical
    edited_valid = original.copy()
    edited_valid[mask] = 100.0
    exact, max_diff = check_outside_mask_exact_equality(original, edited_valid, mask)
    assert exact is True
    assert max_diff == 0.0

    report_ok = IntegrityReport(
        outside_mask_exact_equality=exact,
        outside_mask_max_abs_diff=max_diff,
        boundary_discontinuity=1.2,
        unrelated_output_max_drift=0.02,
        ssim_descriptive_proxy=0.94,
    )
    eraser_ok = EraserContract(
        edited_image=edited_valid,
        mask=mask,
        fill_method="telea",
        seed=0,
        integrity_report=report_ok,
    )
    assert eraser_ok.synthetic_label == SYNTHETIC_COUNTERFACTUAL_LABEL

    # Even a single outside-mask pixel drift must fail check_outside_mask_exact_equality & EraserContract
    edited_leaked = edited_valid.copy()
    edited_leaked[0, 0] += np.float32(1e-5)
    exact_bad, max_diff_bad = check_outside_mask_exact_equality(original, edited_leaked, mask)
    assert exact_bad is False
    assert max_diff_bad > 0.0

    report_bad = IntegrityReport(
        outside_mask_exact_equality=exact_bad,
        outside_mask_max_abs_diff=max_diff_bad,
        boundary_discontinuity=1.2,
        unrelated_output_max_drift=0.02,
    )
    with pytest.raises(ValueError, match="exact pixel equality"):
        EraserContract(
            edited_image=edited_leaked,
            mask=mask,
            fill_method="telea",
            seed=0,
            integrity_report=report_bad,
        )

    # Wrong or missing synthetic label must fail
    with pytest.raises(ValueError, match="SYNTHETIC COUNTERFACTUAL"):
        IntegrityReport(
            outside_mask_exact_equality=True,
            outside_mask_max_abs_diff=0.0,
            boundary_discontinuity=1.0,
            unrelated_output_max_drift=0.01,
            synthetic_label="Edited Scan",
        )


@pytest.mark.parametrize(
    "forbidden_text",
    [
        "Finding is PROVEN by counterfactual deletion.",
        "Status: DISEASE CONFIRMED.",
        "Lungs are NORMAL and CLEAR.",
        "Pneumothorax RULED OUT.",
        "Urgent triage priority: seek emergency treatment.",
        "Clinical diagnosis of pneumothorax with p-value = 0.04.",
        "Generated disease-free reconstruction of true anatomy.",
        "Probability that the patient has pneumothorax is 0.80.",
    ],
)
def test_verdicts_and_forbidden_language_guardrails(forbidden_text: str) -> None:
    assert {v.value for v in Verdict} == {
        "MODEL-SUPPORTED",
        "NOT SUPPORTED",
        "SHORTCUT RISK",
        "INDETERMINATE",
    }
    for v in Verdict:
        validate_safe_language(VERDICT_USER_LANGUAGE[v])
    validate_safe_language(RESEARCH_DISCLAIMER)

    with pytest.raises(ValueError, match="Unsafe or non-compliant language"):
        validate_safe_language(forbidden_text)


def test_judge_and_fail_closed_output_contracts() -> None:
    manifest = _sample_manifest()

    # MODEL-SUPPORTED fails if any gate is False
    with pytest.raises(ValueError, match="every gate True"):
        JudgeContract(
            target_effects={"median_drop": 0.12},
            control_effects={"num_controls": 19},
            stability_metrics={"seed_std": 0.002},
            shortcut_flags={"border": False},
            gate_results={"outside_mask_exact": True, "beats_controls": False},
            verdict=Verdict.MODEL_SUPPORTED,
            reason_codes=("GATE_MISMATCH",),
        )

    judge_ok = JudgeContract(
        target_effects={"median_drop": 0.12},
        control_effects={"num_controls": 19, "empirical_tail_fraction": 0.05},
        stability_metrics={"seed_std": 0.002},
        shortcut_flags={"border": False},
        gate_results={"outside_mask_exact": True, "beats_controls": True},
        verdict=Verdict.MODEL_SUPPORTED,
        reason_codes=("ALL_GATES_PASSED",),
    )
    out_ok = OutputContract(
        verdict=judge_ok.verdict,
        reason_codes=judge_ok.reason_codes,
        evidence_summary={"empirical_tail_fraction": 0.05},
        disclaimer=RESEARCH_DISCLAIMER,
        run_manifest=manifest,
    )
    assert out_ok.verdict is Verdict.MODEL_SUPPORTED

    # Fail-closed helper always produces INDETERMINATE with reason code
    abstained = OutputContract.fail_closed(
        reason_codes=("LOCATOR_UNSTABLE_MASK",),
        run_manifest=manifest,
        failed_stage="locator",
    )
    assert abstained.verdict is Verdict.INDETERMINATE
    assert abstained.reason_codes == ("LOCATOR_UNSTABLE_MASK",)


def test_run_manifest_writer_roundtrip_and_hashing(tmp_path: Path) -> None:
    manifest = _sample_manifest()
    out_file = tmp_path / "manifests" / "run_001.json"
    written = write_run_manifest(manifest, out_file)
    assert written == out_file
    assert written.is_file()
    assert len(sha256_file(written)) == 64

    reloaded = read_run_manifest(written)
    assert reloaded.run_id == manifest.run_id
    assert reloaded.config_sha256 == manifest.config_sha256
    assert reloaded.checkpoint_hashes == manifest.checkpoint_hashes
    assert reloaded.dataset_split_id == "nih_tuning"
    assert reloaded.random_seeds == {"global": 42, "controls": 20261125}
    assert reloaded.offline_only is True

    # Offline-only violation must fail closed
    payload = manifest.to_dict()
    payload["offline_only"] = False
    with pytest.raises(ValueError, match="offline_only=True"):
        from veriray.manifest import RunManifest

        RunManifest.from_dict(payload)
