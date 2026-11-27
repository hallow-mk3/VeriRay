"""Verification Judge (Blueprint 26 Nov §7, v2 §8): matched controls, stability, integrity, shortcut gates.

Four verdicts only: MODEL-SUPPORTED, NOT SUPPORTED, SHORTCUT RISK, INDETERMINATE.
Fail closed: any missing/invalid input, locator instability, edit/integrity error,
fill-method instability, or insufficient matched controls => INDETERMINATE with reason codes.

All thresholds in :class:`JudgeConfig` are PROVISIONAL, set a priori before viewing test data.
Empirical ranks / tail fractions are NEVER called p-values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from veriray.contracts import (
    SUPPORTED_FINDING,
    EraserContract,
    EvidenceRecord,
    JudgeContract,
    LocatorContract,
    ReaderContract,
    Verdict,
)
from veriray.controls import ControlConfig, generate_matched_controls
from veriray.eraser import METHODS as ERASER_METHODS, EraseError, erase
from veriray.locator import LocatorConfig, heatmap_to_mask
from veriray.reader import Reader, ReaderError

_UNET_METHOD_NAME = "gated_unet"


@dataclass(frozen=True)
class JudgeConfig:
    primary_fill_method: str = "telea"
    comparison_fill_methods: tuple[str, ...] = ("gaussian_blur", "mean_fill")
    num_controls: int = 19
    min_controls: int = 8
    max_target_overlap: float = 0.10
    control_area_tolerance: float = 0.25
    min_target_drop: float = 0.03
    min_margin_over_controls: float = 0.02
    max_empirical_tail_fraction: float = 0.15
    min_reader_score: float = 0.0  # optional positive-prediction floor (tuned on val)
    max_boundary_discontinuity: float = 0.35
    max_unrelated_drift: float = 0.15
    max_fill_method_spread: float = 0.25
    min_mask_threshold_iou: float = 0.40
    threshold_delta: float = 0.05
    max_out_of_lung_fraction: float = 0.40
    seed: int = 20261126


@dataclass(frozen=True)
class JudgeOutcome:
    contract: JudgeContract
    evidence_record: EvidenceRecord
    primary_target_edit: EraserContract | None = None
    comparison_target_edits: tuple[EraserContract, ...] = ()
    control_masks: tuple[np.ndarray, ...] = ()


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return 1.0 if union == 0 else inter / float(union)


def _locator_stability_metrics(
    locator: LocatorContract,
    locator_cfg: LocatorConfig,
    delta: float,
) -> dict[str, float]:
    """Measure mask IoU across small threshold shifts (threshold +- delta) and 3x3 dilation."""
    hm = locator.heatmap
    base = locator.target_mask
    lo_cfg = LocatorConfig(
        threshold=max(0.05, locator_cfg.threshold - delta),
        opening_kernel=locator_cfg.opening_kernel,
        closing_kernel=locator_cfg.closing_kernel,
        min_component_frac=locator_cfg.min_component_frac,
    )
    hi_cfg = LocatorConfig(
        threshold=min(0.95, locator_cfg.threshold + delta),
        opening_kernel=locator_cfg.opening_kernel,
        closing_kernel=locator_cfg.closing_kernel,
        min_component_frac=locator_cfg.min_component_frac,
    )
    m_lo = heatmap_to_mask(hm, lo_cfg)
    m_hi = heatmap_to_mask(hm, hi_cfg)
    dilated = cv2.dilate(base.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    iou_lo = _mask_iou(base, m_lo)
    iou_hi = _mask_iou(base, m_hi)
    iou_dil = _mask_iou(base, dilated)
    return {
        "mask_threshold_minus_iou": float(iou_lo),
        "mask_threshold_plus_iou": float(iou_hi),
        "mask_dilation_3x3_iou": float(iou_dil),
        "mask_min_stability_iou": float(min(iou_lo, iou_hi, iou_dil)),
    }


def _max_unrelated_drift(before: Mapping[str, float], after: Mapping[str, float]) -> float:
    common = set(before) & set(after)
    if not common:
        return 0.0
    return float(max(abs(float(after[k]) - float(before[k])) for k in common))


def _indeterminate_outcome(
    reason_codes: tuple[str, ...],
    gate_results: dict[str, bool],
    stability_metrics: dict[str, float],
    shortcut_flags: dict[str, bool],
    target_effects: dict[str, object],
    control_effects: dict[str, object],
    baseline_score: float,
    record_id: str,
    primary_edit: EraserContract | None = None,
    comp_edits: tuple[EraserContract, ...] = (),
    control_masks: tuple[np.ndarray, ...] = (),
) -> JudgeOutcome:
    jc = JudgeContract(
        target_effects=target_effects,
        control_effects=control_effects,
        stability_metrics=stability_metrics,
        shortcut_flags=shortcut_flags,
        gate_results=gate_results,
        verdict=Verdict.INDETERMINATE,
        reason_codes=reason_codes,
    )
    ev = _build_evidence_record(record_id, jc, baseline_score)
    return JudgeOutcome(jc, ev, primary_edit, comp_edits, control_masks)


def _build_evidence_record(record_id: str, jc: JudgeContract, baseline_score: float) -> EvidenceRecord:
    te = dict(jc.target_effects)
    ce = dict(jc.control_effects)
    sm = dict(jc.stability_metrics)
    sf = dict(jc.shortcut_flags)
    gr = dict(jc.gate_results)
    items: dict[str, object] = {
        "E01_MODEL_SCORE_RAW": round(float(baseline_score), 6),
        "E02_VERDICT": jc.verdict.value,
        "E03_REASON_CODES": list(jc.reason_codes),
        "E04_TARGET_SCORE_DROP_PRIMARY": te.get("primary_score_drop"),
        "E05_TARGET_SCORE_DROP_MEDIAN": te.get("median_score_drop"),
        "E06_CONTROL_COUNT": ce.get("num_controls", 0),
        "E07_CONTROL_SCORE_DROP_MEDIAN": ce.get("median_score_drop"),
        "E08_EMPIRICAL_TAIL_FRACTION": te.get("empirical_tail_fraction"),
        "E09_MARGIN_OVER_CONTROLS": te.get("margin_over_control_median"),
        "E10_FILL_METHOD_SPREAD": sm.get("fill_method_spread"),
        "E11_MASK_MIN_STABILITY_IOU": sm.get("mask_min_stability_iou"),
        "E12_UNRELATED_LABEL_MAX_DRIFT": sm.get("unrelated_output_max_drift"),
        "E13_BOUNDARY_DISCONTINUITY": sm.get("primary_boundary_discontinuity"),
        "E14_SHORTCUT_FLAGS": sf,
        "E15_GATE_RESULTS": gr,
    }
    return EvidenceRecord(
        record_id=record_id,
        finding_id=SUPPORTED_FINDING,
        verdict=jc.verdict,
        reason_codes=jc.reason_codes,
        items=items,
    )


def run_judge(
    reader: Reader,
    image_224: np.ndarray,
    baseline_reader: ReaderContract,
    locator: LocatorContract,
    *,
    cfg: JudgeConfig = JudgeConfig(),
    locator_cfg: LocatorConfig = LocatorConfig(),
    lung_mask: np.ndarray | None = None,
    audited_shortcut_cue: bool = False,
    record_id: str = "ev-step3",
    unet_eraser: object | None = None,
) -> JudgeOutcome:
    """Execute Steps 4-11 of the Verification Judge protocol on one accepted 224x224 image."""
    gates: dict[str, bool] = {
        "locator_valid": False,
        "mask_stability_passed": False,
        "controls_valid": False,
        "edit_integrity_passed": False,
        "fill_stability_passed": False,
        "no_shortcut_flagged": False,
        "target_distinguishable_from_controls": False,
    }
    stability: dict[str, float] = {}
    shortcut_flags: dict[str, bool] = {
        "lung_mask_available": lung_mask is not None,
        "out_of_lung_attribution": False,
        "audited_shortcut_cue": bool(audited_shortcut_cue),
    }
    target_effects: dict[str, object] = {}
    control_effects: dict[str, object] = {"num_controls": 0}
    baseline_score = float(baseline_reader.score_raw)

    # 0. When a positive-prediction floor is configured (min_reader_score > 0) and the Reader
    # score is below it, a weak/unstable Grad-CAM++ map on a negative/no-finding scan should
    # yield NOT SUPPORTED (READER_SCORE_BELOW_THRESHOLD) rather than INDETERMINATE.
    if cfg.min_reader_score > 0.0 and baseline_score < cfg.min_reader_score and locator.is_weak_or_unstable:
        reasons = ["READER_SCORE_BELOW_THRESHOLD"]
        if all(float(v) < 0.5 for v in baseline_reader.unrelated_scores.values()):
            reasons.append("NO_FINDING_FLAGGED_BY_READER")
        jc = JudgeContract(
            target_effects={"primary_fill_method": cfg.primary_fill_method, "primary_score_drop": 0.0},
            control_effects=control_effects,
            stability_metrics=stability,
            shortcut_flags=shortcut_flags,
            gate_results=gates,
            verdict=Verdict.NOT_SUPPORTED,
            reason_codes=tuple(reasons),
        )
        return JudgeOutcome(jc, _build_evidence_record(record_id, jc, baseline_score))

    # 1. Locator quality gate
    if locator.is_weak_or_unstable:
        return _indeterminate_outcome(
            ("JUDGE_LOCATOR_WEAK_OR_UNSTABLE",), gates, stability, shortcut_flags,
            target_effects, control_effects, baseline_score, record_id,
        )
    gates["locator_valid"] = True

    # 2. Locator threshold & dilation stability check (resolves Step 2's stability_assessed=False)
    stability.update(_locator_stability_metrics(locator, locator_cfg, cfg.threshold_delta))
    if stability["mask_min_stability_iou"] < cfg.min_mask_threshold_iou:
        return _indeterminate_outcome(
            ("JUDGE_MASK_UNSTABLE",), gates, stability, shortcut_flags,
            target_effects, control_effects, baseline_score, record_id,
        )
    gates["mask_stability_passed"] = True

    # 3. Validate primary & comparison fill methods
    all_methods = [cfg.primary_fill_method, *[m for m in cfg.comparison_fill_methods if m != cfg.primary_fill_method]]
    if unet_eraser is not None and _UNET_METHOD_NAME not in all_methods:
        all_methods = [*all_methods, _UNET_METHOD_NAME]
    standard_methods = [m for m in all_methods if m != _UNET_METHOD_NAME]
    if any(m not in ERASER_METHODS for m in standard_methods):
        return _indeterminate_outcome(
            ("JUDGE_UNKNOWN_FILL_METHOD",), gates, stability, shortcut_flags,
            target_effects, control_effects, baseline_score, record_id,
        )

    # 4. Run target edits across primary + comparison fill methods (+ optional U-Net)
    target_edits: list[EraserContract] = []
    target_drops: dict[str, float] = {}
    unrelated_drifts: list[float] = []
    boundary_vals: list[float] = []
    try:
        for method in all_methods:
            # First create edit (pixel equality guaranteed inside erase() / unet_eraser.erase())
            if method == _UNET_METHOD_NAME:
                assert unet_eraser is not None
                prelim = unet_eraser.erase(image_224, locator.target_mask, seed=cfg.seed)
            else:
                prelim = erase(image_224, locator.target_mask, method, seed=cfg.seed)
            edited_rc, _, _ = reader.score(prelim.edited_image)
            drift = _max_unrelated_drift(baseline_reader.unrelated_scores, edited_rc.unrelated_scores)
            # Re-wrap or construct final EraserContract with measured unrelated drift
            if method == _UNET_METHOD_NAME:
                from dataclasses import replace as _dc_replace
                from veriray.contracts import IntegrityReport
                new_report = IntegrityReport(
                    outside_mask_exact_equality=prelim.integrity_report.outside_mask_exact_equality,
                    outside_mask_max_abs_diff=prelim.integrity_report.outside_mask_max_abs_diff,
                    boundary_discontinuity=prelim.integrity_report.boundary_discontinuity,
                    unrelated_output_max_drift=drift,
                    ssim_descriptive_proxy=prelim.integrity_report.ssim_descriptive_proxy,
                )
                final_edit = EraserContract(
                    edited_image=prelim.edited_image,
                    mask=prelim.mask,
                    fill_method=prelim.fill_method,
                    seed=prelim.seed,
                    integrity_report=new_report,
                    config=prelim.config,
                )
            else:
                final_edit = erase(
                    image_224,
                    locator.target_mask,
                    method,
                    seed=cfg.seed,
                    unrelated_output_max_drift=drift,
                )
            target_edits.append(final_edit)
            target_drops[method] = float(baseline_score - edited_rc.score_raw)
            unrelated_drifts.append(drift)
            boundary_vals.append(float(final_edit.integrity_report.boundary_discontinuity))
    except (EraseError, ReaderError) as exc:
        return _indeterminate_outcome(
            ("JUDGE_TARGET_EDIT_FAILED", exc.reason_code), gates, stability, shortcut_flags,
            target_effects, control_effects, baseline_score, record_id,
        )
    except Exception:  # noqa: BLE001 - fail closed
        return _indeterminate_outcome(
            ("JUDGE_TARGET_EDIT_UNEXPECTED_ERROR",), gates, stability, shortcut_flags,
            target_effects, control_effects, baseline_score, record_id,
        )

    primary_edit = target_edits[0]
    comp_edits = tuple(target_edits[1:])
    primary_drop = target_drops[cfg.primary_fill_method]
    median_target_drop = float(np.median(list(target_drops.values())))
    fill_spread = float(max(target_drops.values()) - min(target_drops.values())) if len(target_drops) > 1 else 0.0
    max_drift = float(max(unrelated_drifts))
    max_boundary = float(max(boundary_vals))

    stability["fill_method_spread"] = fill_spread
    stability["unrelated_output_max_drift"] = max_drift
    stability["primary_boundary_discontinuity"] = float(primary_edit.integrity_report.boundary_discontinuity)
    stability["max_boundary_discontinuity"] = max_boundary

    target_effects = {
        "primary_fill_method": cfg.primary_fill_method,
        "primary_score_drop": primary_drop,
        "median_score_drop": median_target_drop,
        "per_method_score_drops": target_drops,
    }

    # 5. Check integrity & fill-method stability gates before trusting effects
    if max_boundary > cfg.max_boundary_discontinuity:
        return _indeterminate_outcome(
            ("JUDGE_BOUNDARY_DISCONTINUITY_EXCEEDED",), gates, stability, shortcut_flags,
            target_effects, control_effects, baseline_score, record_id, primary_edit, comp_edits,
        )
    if max_drift > cfg.max_unrelated_drift:
        return _indeterminate_outcome(
            ("JUDGE_UNRELATED_LABEL_DRIFT_EXCEEDED",), gates, stability, shortcut_flags,
            target_effects, control_effects, baseline_score, record_id, primary_edit, comp_edits,
        )
    gates["edit_integrity_passed"] = True

    if fill_spread > cfg.max_fill_method_spread:
        return _indeterminate_outcome(
            ("JUDGE_FILL_METHOD_INSTABILITY",), gates, stability, shortcut_flags,
            target_effects, control_effects, baseline_score, record_id, primary_edit, comp_edits,
        )
    gates["fill_stability_passed"] = True

    # 6. Generate K matched control masks
    ctrl_cfg = ControlConfig(
        k=cfg.num_controls,
        min_controls=cfg.min_controls,
        max_target_overlap=cfg.max_target_overlap,
        area_tolerance=cfg.control_area_tolerance,
        seed=cfg.seed,
    )
    ctrl_res = generate_matched_controls(locator.target_mask, ctrl_cfg, anatomical_mask=lung_mask)
    if not ctrl_res.valid:
        return _indeterminate_outcome(
            ("JUDGE_INSUFFICIENT_CONTROLS", *ctrl_res.reason_codes), gates, stability, shortcut_flags,
            target_effects, control_effects, baseline_score, record_id,
            primary_edit, comp_edits, ctrl_res.masks,
        )
    gates["controls_valid"] = True

    # 7. Run primary fill method on each matched control
    control_drops: list[float] = []
    try:
        for i, cmask in enumerate(ctrl_res.masks):
            cedit = erase(image_224, cmask, cfg.primary_fill_method, seed=cfg.seed + i + 1)
            c_rc, _, _ = reader.score(cedit.edited_image)
            control_drops.append(float(baseline_score - c_rc.score_raw))
    except (EraseError, ReaderError) as exc:
        return _indeterminate_outcome(
            ("JUDGE_CONTROL_EDIT_FAILED", exc.reason_code), gates, stability, shortcut_flags,
            target_effects, control_effects, baseline_score, record_id,
            primary_edit, comp_edits, ctrl_res.masks,
        )
    except Exception:  # noqa: BLE001
        return _indeterminate_outcome(
            ("JUDGE_CONTROL_EDIT_UNEXPECTED_ERROR",), gates, stability, shortcut_flags,
            target_effects, control_effects, baseline_score, record_id,
            primary_edit, comp_edits, ctrl_res.masks,
        )

    c_arr = np.asarray(control_drops, dtype=np.float64)
    ctrl_median = float(np.median(c_arr))
    ctrl_iqr = float(np.percentile(c_arr, 75) - np.percentile(c_arr, 25))
    # Empirical rank: 1 + count of controls >= primary_drop. Never call this a p-value.
    n_ge = int(np.sum(c_arr >= primary_drop))
    empirical_rank = 1 + n_ge
    empirical_tail_fraction = empirical_rank / float(len(control_drops) + 1)
    margin = float(primary_drop - ctrl_median)

    target_effects["empirical_rank"] = empirical_rank
    target_effects["empirical_tail_fraction"] = empirical_tail_fraction
    target_effects["margin_over_control_median"] = margin
    control_effects = {
        "num_controls": len(control_drops),
        "score_drops": control_drops,
        "median_score_drop": ctrl_median,
        "iqr_score_drop": ctrl_iqr,
        "max_score_drop": float(np.max(c_arr)),
        "max_overlap_with_target": float(max(ctrl_res.overlaps_with_target)),
        "zone_matched_count": ctrl_res.zone_matched_count,
    }

    # 8. Shortcut checks (outside-lung attribution when lung_mask supplied + audited cue)
    if lung_mask is not None:
        lmask = np.asarray(lung_mask, dtype=bool)
        if lmask.shape != locator.target_mask.shape:
            return _indeterminate_outcome(
                ("JUDGE_BAD_LUNG_MASK",), gates, stability, shortcut_flags,
                target_effects, control_effects, baseline_score, record_id,
                primary_edit, comp_edits, ctrl_res.masks,
            )
        out_of_lung_frac = float(locator.target_mask[~lmask].sum()) / float(max(1, int(locator.target_mask.sum())))
        stability["out_of_lung_target_fraction"] = out_of_lung_frac
        if out_of_lung_frac > cfg.max_out_of_lung_fraction:
            shortcut_flags["out_of_lung_attribution"] = True

    if shortcut_flags["out_of_lung_attribution"] or shortcut_flags["audited_shortcut_cue"]:
        reasons = tuple(
            code
            for flag, code in (
                (shortcut_flags["out_of_lung_attribution"], "SHORTCUT_OUT_OF_LUNG_ATTRIBUTION"),
                (shortcut_flags["audited_shortcut_cue"], "SHORTCUT_AUDITED_CUE_FLAGGED"),
            )
            if flag
        )
        jc = JudgeContract(
            target_effects=target_effects,
            control_effects=control_effects,
            stability_metrics=stability,
            shortcut_flags=shortcut_flags,
            gate_results=gates,
            verdict=Verdict.SHORTCUT_RISK,
            reason_codes=reasons,
        )
        return JudgeOutcome(jc, _build_evidence_record(record_id, jc, baseline_score),
                            primary_edit, comp_edits, ctrl_res.masks)
    gates["no_shortcut_flagged"] = True

    # 9. Distinguish target drop from matched controls under frozen criteria
    distinguishable = (
        baseline_score >= cfg.min_reader_score
        and primary_drop >= cfg.min_target_drop
        and margin >= cfg.min_margin_over_controls
        and empirical_tail_fraction <= cfg.max_empirical_tail_fraction
    )
    gates["target_distinguishable_from_controls"] = bool(distinguishable)

    if distinguishable:
        verdict = Verdict.MODEL_SUPPORTED
        reasons = ("JUDGE_ALL_GATES_PASSED",)
    else:
        verdict = Verdict.NOT_SUPPORTED
        not_supp_reasons = []
        if baseline_score < cfg.min_reader_score:
            not_supp_reasons.append("READER_SCORE_BELOW_THRESHOLD")
            if all(float(v) < 0.5 for v in baseline_reader.unrelated_scores.values()):
                not_supp_reasons.append("NO_FINDING_FLAGGED_BY_READER")
        if primary_drop < cfg.min_target_drop:
            not_supp_reasons.append("TARGET_DROP_BELOW_THRESHOLD")
        if margin < cfg.min_margin_over_controls:
            not_supp_reasons.append("MARGIN_OVER_CONTROLS_BELOW_THRESHOLD")
        if empirical_tail_fraction > cfg.max_empirical_tail_fraction:
            not_supp_reasons.append("EMPIRICAL_RANK_NOT_DISTINCT_FROM_CONTROLS")
        reasons = tuple(not_supp_reasons) or ("TARGET_EVIDENCE_NOT_SUPPORTED",)

    jc = JudgeContract(
        target_effects=target_effects,
        control_effects=control_effects,
        stability_metrics=stability,
        shortcut_flags=shortcut_flags,
        gate_results=gates,
        verdict=verdict,
        reason_codes=reasons,
    )
    return JudgeOutcome(
        jc,
        _build_evidence_record(record_id, jc, baseline_score),
        primary_edit,
        comp_edits,
        ctrl_res.masks,
    )
