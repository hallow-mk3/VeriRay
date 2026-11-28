"""Typed dataclass module contracts for VeriRay v2.0 (Blueprint Sections 4, 4.1, 7.3, 8.4, 12.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping
import numpy as np

if TYPE_CHECKING:
    from veriray.manifest import RunManifest


SUPPORTED_FINDING: str = "pneumothorax"
# Restricted strictly to findings passing the frozen verifiable rule + border-shortcut gate
# on BOTH dev_val and locked_test (positives >= 100, Dice >= 0.50, border_probe <= 0.80):
SUPPORTED_FINDINGS: tuple[str, ...] = (
    "pneumothorax",
    "effusion",
    "pneumonia",
)
SYNTHETIC_COUNTERFACTUAL_LABEL: str = "SYNTHETIC COUNTERFACTUAL — NOT A REAL SCAN"
RESEARCH_DISCLAIMER: str = (
    "IMPORTANT: Research and competition prototype only. "
    "Not for diagnosis, triage, treatment or clinical use."
)

ALLOWED_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "source_dataset",
        "split",
        "view",
        "label_mapping",
        "annotation_available",
        "preprocessing_hash",
        "sample_id",
        "capture_conditions",
    }
)


class InputGateStatus(str, Enum):
    """Section 4 Input Gate failure/acceptance states."""

    ACCEPT = "ACCEPT"
    RETAKE = "RETAKE"
    REJECT = "REJECT"


class Verdict(str, Enum):
    """Section 8.4 allowed verdicts ONLY."""

    MODEL_SUPPORTED = "MODEL-SUPPORTED"
    NOT_SUPPORTED = "NOT SUPPORTED"
    SHORTCUT_RISK = "SHORTCUT RISK"
    INDETERMINATE = "INDETERMINATE"


VERDICT_USER_LANGUAGE: Mapping[Verdict, str] = MappingProxyType(
    {
        Verdict.MODEL_SUPPORTED: (
            "The model's prediction was supported by this test. "
            "This is not a diagnosis."
        ),
        Verdict.NOT_SUPPORTED: (
            "The model's prediction was not supported by this test. "
            "This does not mean disease is absent."
        ),
        Verdict.SHORTCUT_RISK: (
            "Possible reliance on a non-lung cue was detected. "
            "No diagnostic conclusion."
        ),
        Verdict.INDETERMINATE: (
            "The test could not produce a reliable result. Reason shown."
        ),
    }
)

# Section 8.4, 8.2, 12.1 & 18 prohibited terms/claims.
# Approved negated disclaimers are masked out prior to scanning so phrases like
# "Not for diagnosis, triage, treatment or clinical use" and "This is not a diagnosis"
# pass while affirmative diagnostic, triage, urgency, treatment, or PROVEN language fails.
_APPROVED_NEGATED_PHRASES: tuple[str, ...] = (
    "not for diagnosis, triage, treatment or clinical use",
    "this is not a diagnosis",
    "no diagnostic conclusion",
    "without claiming to diagnose disease",
    "describe model behaviour, not a diagnosis",
)

_FORBIDDEN_LANGUAGE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bPROVEN\b", "FORBIDDEN_VERDICT_PROVEN"),
    (r"\bUNPROVEN\b", "FORBIDDEN_VERDICT_UNPROVEN"),
    (r"\bDISEASE\s+CONFIRMED\b", "FORBIDDEN_VERDICT_DISEASE_CONFIRMED"),
    (r"\bCONFIRMED\s+DISEASE\b", "FORBIDDEN_VERDICT_DISEASE_CONFIRMED"),
    (r"\bNORMAL\b", "FORBIDDEN_VERDICT_NORMAL"),
    (r"\bCLEAR\b", "FORBIDDEN_VERDICT_CLEAR"),
    (r"\bRULED\s+OUT\b", "FORBIDDEN_VERDICT_RULED_OUT"),
    (r"\bTRIAGE\b", "FORBIDDEN_TRIAGE_LANGUAGE"),
    (r"\bURGENT\b|\bURGENCY\b|\bEMERGENCY\b", "FORBIDDEN_URGENCY_LANGUAGE"),
    (r"\bDIAGNOSIS\b|\bDIAGNOSE\b|\bDIAGNOSED\b|\bDIAGNOSTIC\b", "FORBIDDEN_DIAGNOSTIC_LANGUAGE"),
    (r"\bTREATMENT\b|\bPRESCRIBE\b|\bMEDICATION\b|\bDOSAGE\b", "FORBIDDEN_TREATMENT_LANGUAGE"),
    (r"\bp[\-\s]?value(?:s)?\b", "FORBIDDEN_FORMAL_P_VALUE_CLAIM"),
    (r"\btrue\s+anatomy\b|\bdisease[\-\s]free\s+reconstruction\b", "FORBIDDEN_REALISM_CLAIM"),
    (r"\bprobability\s+that\s+(?:a\s+|the\s+)?patient\s+has\b", "FORBIDDEN_DISEASE_PROBABILITY_CLAIM"),
)


def _freeze_mapping(mapping: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if mapping is None:
        return MappingProxyType({})
    return MappingProxyType(dict(mapping))


def validate_safe_language(text: str) -> None:
    """Raise ValueError if text contains forbidden verdict, diagnostic, triage, urgency, or p-value terms."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text field must be a non-empty string.")
    scrubbed = text.lower()
    for safe_phrase in _APPROVED_NEGATED_PHRASES:
        scrubbed = scrubbed.replace(safe_phrase, " ")

    violations: list[str] = []
    for pattern, code in _FORBIDDEN_LANGUAGE_PATTERNS:
        if re.search(pattern, scrubbed, flags=re.IGNORECASE):
            violations.append(code)
    if violations:
        raise ValueError(f"Unsafe or non-compliant language detected: {violations}")


def check_outside_mask_exact_equality(
    original_image: np.ndarray,
    edited_image: np.ndarray,
    mask: np.ndarray,
) -> tuple[bool, float]:
    """Assert/measure exact pixel equality outside the binary edit mask (Section 7.3)."""
    if original_image.shape != edited_image.shape:
        raise ValueError(
            f"Image shape mismatch: original {original_image.shape} vs edited {edited_image.shape}."
        )
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != original_image.shape[-2:] and mask_bool.shape != original_image.shape:
        raise ValueError(
            f"Mask shape {mask_bool.shape} incompatible with image shape {original_image.shape}."
        )
    if mask_bool.shape != original_image.shape:
        mask_bool = np.broadcast_to(mask_bool, original_image.shape)

    outside = ~mask_bool
    if not np.any(outside):
        return True, 0.0
    exact = bool(np.array_equal(edited_image[outside], original_image[outside]))
    max_diff = float(np.max(np.abs(edited_image[outside].astype(np.float64) - original_image[outside].astype(np.float64))))
    return exact, max_diff


@dataclass(frozen=True, slots=True)
class InputGateContract:
    """Section 4 Input Gate output: ACCEPT / RETAKE / REJECT with machine-readable reason; fail closed."""

    status: InputGateStatus
    reason_codes: tuple[str, ...]
    quality_flags: Mapping[str, bool] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.status, InputGateStatus):
            raise TypeError(f"status must be InputGateStatus, got {type(self.status).__name__}.")
        codes = tuple(str(c).strip() for c in self.reason_codes if str(c).strip())
        if not codes:
            raise ValueError("InputGateContract requires at least one machine-readable reason_code.")
        object.__setattr__(self, "reason_codes", codes)
        object.__setattr__(
            self,
            "quality_flags",
            MappingProxyType({str(k): bool(v) for k, v in dict(self.quality_flags).items()}),
        )

    @property
    def accepted(self) -> bool:
        return self.status is InputGateStatus.ACCEPT


@dataclass(frozen=True, slots=True)
class CaptureCleanerContract:
    """Section 4 Capture Cleaner (optional MVP): retains original and transformed separately."""

    original_image: np.ndarray
    transformed_image: np.ndarray | None
    transform_log: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.original_image, np.ndarray) or self.original_image.ndim < 2:
            raise ValueError("original_image must be at least a 2D numpy ndarray.")
        if self.transformed_image is not None:
            if not isinstance(self.transformed_image, np.ndarray) or self.transformed_image.ndim < 2:
                raise ValueError("transformed_image must be a numpy ndarray or None.")
            if self.transformed_image is self.original_image:
                raise ValueError("transformed_image must be retained separately; no silent in-place replacement.")
        object.__setattr__(self, "transform_log", _freeze_mapping(self.transform_log))


@dataclass(frozen=True, slots=True)
class InputContract:
    """Section 4.1 Input: image_tensor, image_metadata_allowlist, finding_id, model_id, preprocessing_id."""

    image_tensor: np.ndarray
    image_metadata_allowlist: Mapping[str, Any]
    finding_id: str
    model_id: str
    preprocessing_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.image_tensor, np.ndarray):
            raise TypeError("image_tensor must be a numpy.ndarray.")
        if self.image_tensor.ndim not in (2, 3, 4) or self.image_tensor.size == 0:
            raise ValueError(f"image_tensor must be 2D, 3D, or 4D non-empty array; got shape {self.image_tensor.shape}.")
        if not np.all(np.isfinite(self.image_tensor)):
            raise ValueError("image_tensor contains NaN or infinite values.")

        normalized_finding = self.finding_id.strip().lower()
        if normalized_finding not in SUPPORTED_FINDINGS:
            raise ValueError(
                f"VeriRay restricts finding_id to {SUPPORTED_FINDINGS}; got '{self.finding_id}'."
            )
        object.__setattr__(self, "finding_id", normalized_finding)

        if not self.model_id or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string.")
        if not self.preprocessing_id or not self.preprocessing_id.strip():
            raise ValueError("preprocessing_id must be a non-empty string.")

        meta_dict = dict(self.image_metadata_allowlist)
        disallowed = set(meta_dict.keys()) - ALLOWED_METADATA_KEYS
        if disallowed:
            raise ValueError(
                f"Disallowed metadata keys {sorted(disallowed)}; only allowlist keys {sorted(ALLOWED_METADATA_KEYS)} permitted."
            )
        object.__setattr__(self, "image_metadata_allowlist", _freeze_mapping(meta_dict))


@dataclass(frozen=True, slots=True)
class ReaderContract:
    """Section 4.1 Reader: score_raw, score_calibrated_optional, threshold_id (+ checkpoint/preprocessing IDs)."""

    score_raw: float
    score_calibrated_optional: float | None
    threshold_id: str
    checkpoint_id: str = ""
    preprocessing_id: str = ""
    unrelated_scores: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        raw = float(self.score_raw)
        if not np.isfinite(raw) or not (0.0 <= raw <= 1.0):
            raise ValueError(f"score_raw must be a finite model score in [0.0, 1.0]; got {self.score_raw}.")
        object.__setattr__(self, "score_raw", raw)

        if self.score_calibrated_optional is not None:
            cal = float(self.score_calibrated_optional)
            if not np.isfinite(cal) or not (0.0 <= cal <= 1.0):
                raise ValueError(
                    f"score_calibrated_optional must be in [0.0, 1.0] when provided; got {self.score_calibrated_optional}."
                )
            object.__setattr__(self, "score_calibrated_optional", cal)

        if not self.threshold_id or not self.threshold_id.strip():
            raise ValueError("threshold_id must be a non-empty string.")

        clean_unrelated: dict[str, float] = {}
        for k, v in dict(self.unrelated_scores).items():
            fv = float(v)
            if not np.isfinite(fv) or not (0.0 <= fv <= 1.0):
                raise ValueError(f"Unrelated score '{k}' must be finite in [0.0, 1.0]; got {v}.")
            clean_unrelated[str(k)] = fv
        object.__setattr__(self, "unrelated_scores", MappingProxyType(clean_unrelated))


@dataclass(frozen=True, slots=True)
class LocatorContract:
    """Section 4.1 Locator: heatmap, target_mask, localization_flags."""

    heatmap: np.ndarray
    target_mask: np.ndarray
    localization_flags: Mapping[str, bool]

    def __post_init__(self) -> None:
        if not isinstance(self.heatmap, np.ndarray) or self.heatmap.ndim != 2:
            raise ValueError("heatmap must be a 2D numpy.ndarray.")
        if not np.all(np.isfinite(self.heatmap)):
            raise ValueError("heatmap contains non-finite values.")
        hm_min, hm_max = float(np.min(self.heatmap)), float(np.max(self.heatmap))
        if hm_min < -1e-6 or hm_max > 1.0 + 1e-6:
            raise ValueError(f"heatmap must be normalized to [0.0, 1.0]; got range [{hm_min}, {hm_max}].")

        if not isinstance(self.target_mask, np.ndarray) or self.target_mask.shape != self.heatmap.shape:
            raise ValueError("target_mask must be a 2D numpy.ndarray matching heatmap.shape.")
        if self.target_mask.dtype != np.bool_:
            object.__setattr__(self, "target_mask", self.target_mask.astype(bool))

        flags = {str(k): bool(v) for k, v in dict(self.localization_flags).items()}
        object.__setattr__(self, "localization_flags", MappingProxyType(flags))

    @property
    def is_weak_or_unstable(self) -> bool:
        """Section 6: empty, diffuse, implausibly large, anatomically implausible, or unstable => INDETERMINATE."""
        failure_keys = (
            "empty",
            "diffuse",
            "implausibly_large",
            "anatomically_implausible",
            "unstable",
        )
        return (not bool(np.any(self.target_mask))) or any(
            self.localization_flags.get(k, False) for k in failure_keys
        )


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Section 7.3 Integrity checks for synthetic counterfactual edits."""

    outside_mask_exact_equality: bool
    outside_mask_max_abs_diff: float
    boundary_discontinuity: float
    unrelated_output_max_drift: float
    ssim_descriptive_proxy: float | None = None
    synthetic_label: str = SYNTHETIC_COUNTERFACTUAL_LABEL

    def __post_init__(self) -> None:
        if self.synthetic_label != SYNTHETIC_COUNTERFACTUAL_LABEL:
            raise ValueError(
                f"Every altered image must carry '{SYNTHETIC_COUNTERFACTUAL_LABEL}'; got '{self.synthetic_label}'."
            )
        if self.outside_mask_exact_equality and self.outside_mask_max_abs_diff != 0.0:
            raise ValueError(
                "outside_mask_exact_equality=True requires outside_mask_max_abs_diff == 0.0."
            )
        if not self.outside_mask_exact_equality and self.outside_mask_max_abs_diff <= 0.0:
            raise ValueError(
                "outside_mask_exact_equality=False requires positive outside_mask_max_abs_diff."
            )
        if not np.isfinite(self.boundary_discontinuity) or self.boundary_discontinuity < 0.0:
            raise ValueError("boundary_discontinuity must be finite and >= 0.0.")
        if not np.isfinite(self.unrelated_output_max_drift) or self.unrelated_output_max_drift < 0.0:
            raise ValueError("unrelated_output_max_drift must be finite and >= 0.0.")
        if self.ssim_descriptive_proxy is not None:
            s_val = float(self.ssim_descriptive_proxy)
            if not np.isfinite(s_val) or not (-1.0 <= s_val <= 1.0):
                raise ValueError("ssim_descriptive_proxy must lie in [-1.0, 1.0].")
            object.__setattr__(self, "ssim_descriptive_proxy", s_val)


@dataclass(frozen=True, slots=True)
class EraserContract:
    """Section 4.1 Eraser: edited_image, mask, fill_method, seed, integrity_report."""

    edited_image: np.ndarray
    mask: np.ndarray
    fill_method: str
    seed: int
    integrity_report: IntegrityReport
    config: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    synthetic_label: str = SYNTHETIC_COUNTERFACTUAL_LABEL

    def __post_init__(self) -> None:
        if not isinstance(self.edited_image, np.ndarray) or self.edited_image.ndim not in (2, 3):
            raise ValueError("edited_image must be a 2D or 3D numpy.ndarray.")
        if not np.all(np.isfinite(self.edited_image)):
            raise ValueError("edited_image contains non-finite values.")
        if not isinstance(self.mask, np.ndarray) or self.mask.ndim != 2:
            raise ValueError("mask must be a 2D numpy.ndarray.")
        if self.mask.shape != self.edited_image.shape[:2]:
            raise ValueError("mask shape must match spatial dimensions of edited_image.")
        if self.mask.dtype != np.bool_:
            object.__setattr__(self, "mask", self.mask.astype(bool))
        if not self.fill_method or not self.fill_method.strip():
            raise ValueError("fill_method must be a non-empty string.")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an int.")
        if not isinstance(self.integrity_report, IntegrityReport):
            raise TypeError("integrity_report must be an IntegrityReport instance.")
        if not self.integrity_report.outside_mask_exact_equality:
            raise ValueError(
                "EraserContract requires exact pixel equality outside the edit mask (Section 7.3)."
            )
        if self.synthetic_label != SYNTHETIC_COUNTERFACTUAL_LABEL:
            raise ValueError(
                f"synthetic_label must equal '{SYNTHETIC_COUNTERFACTUAL_LABEL}'."
            )
        object.__setattr__(self, "config", _freeze_mapping(self.config))


@dataclass(frozen=True, slots=True)
class JudgeContract:
    """Section 4.1 Judge: target_effects, control_effects, stability_metrics, shortcut_flags, gate_results."""

    target_effects: Mapping[str, Any]
    control_effects: Mapping[str, Any]
    stability_metrics: Mapping[str, float]
    shortcut_flags: Mapping[str, bool]
    gate_results: Mapping[str, bool]
    verdict: Verdict = Verdict.INDETERMINATE
    reason_codes: tuple[str, ...] = ("UNINITIALIZED",)

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, Verdict):
            raise TypeError(f"verdict must be Verdict, got {type(self.verdict).__name__}.")
        codes = tuple(str(c).strip() for c in self.reason_codes if str(c).strip())
        if not codes:
            raise ValueError("JudgeContract requires at least one machine-readable reason_code.")
        object.__setattr__(self, "reason_codes", codes)
        object.__setattr__(self, "target_effects", _freeze_mapping(self.target_effects))
        object.__setattr__(self, "control_effects", _freeze_mapping(self.control_effects))
        object.__setattr__(
            self,
            "stability_metrics",
            MappingProxyType({str(k): float(v) for k, v in dict(self.stability_metrics).items()}),
        )
        object.__setattr__(
            self,
            "shortcut_flags",
            MappingProxyType({str(k): bool(v) for k, v in dict(self.shortcut_flags).items()}),
        )
        gates = {str(k): bool(v) for k, v in dict(self.gate_results).items()}
        object.__setattr__(self, "gate_results", MappingProxyType(gates))

        # Section 8.4: MODEL-SUPPORTED requires all prespecified gates to pass!
        if self.verdict is Verdict.MODEL_SUPPORTED:
            if not gates or not all(gates.values()):
                raise ValueError(
                    "Verdict.MODEL_SUPPORTED requires non-empty gate_results with every gate True."
                )


@dataclass(frozen=True, slots=True)
class OutputContract:
    """Section 4.1 Output: verdict, reason_codes, evidence_summary, disclaimer, run_manifest."""

    verdict: Verdict
    reason_codes: tuple[str, ...]
    evidence_summary: Mapping[str, Any]
    disclaimer: str
    run_manifest: "RunManifest"
    user_facing_text: str = ""

    def __post_init__(self) -> None:
        from veriray.manifest import RunManifest

        if not isinstance(self.verdict, Verdict):
            raise TypeError(f"verdict must be Verdict, got {type(self.verdict).__name__}.")
        codes = tuple(str(c).strip() for c in self.reason_codes if str(c).strip())
        if not codes:
            raise ValueError("OutputContract requires at least one machine-readable reason_code.")
        object.__setattr__(self, "reason_codes", codes)

        if not isinstance(self.run_manifest, RunManifest):
            raise TypeError("run_manifest must be a RunManifest instance.")

        default_text = VERDICT_USER_LANGUAGE[self.verdict]
        final_user_text = self.user_facing_text.strip() if self.user_facing_text else default_text
        validate_safe_language(final_user_text)
        validate_safe_language(self.disclaimer)

        if "not for diagnosis" not in self.disclaimer.lower():
            raise ValueError("disclaimer must state clearly that VeriRay is not for diagnosis/clinical use.")

        object.__setattr__(self, "user_facing_text", final_user_text)
        object.__setattr__(self, "evidence_summary", _freeze_mapping(self.evidence_summary))

    @classmethod
    def fail_closed(
        cls,
        reason_codes: tuple[str, ...] | list[str],
        run_manifest: "RunManifest",
        failed_stage: str,
    ) -> "OutputContract":
        """Section 12.1 & 15.1 fail-closed factory: any gate/model/edit failure => INDETERMINATE."""
        codes = tuple(reason_codes) if reason_codes else ("UNSPECIFIED_FAILURE",)
        return cls(
            verdict=Verdict.INDETERMINATE,
            reason_codes=codes,
            evidence_summary={"failed_stage": str(failed_stage), "abstained": True},
            disclaimer=RESEARCH_DISCLAIMER,
            run_manifest=run_manifest,
            user_facing_text=VERDICT_USER_LANGUAGE[Verdict.INDETERMINATE],
        )


@dataclass(frozen=True, slots=True)
class VLMClaim:
    """26 Nov Blueprint §5: structured finding claim produced by a vision-language claimant."""

    finding: str
    status: str  # present | absent | uncertain
    region_description: str
    box_xyxy: tuple[float, float, float, float] | None = None
    raw_json_valid: bool = True

    def __post_init__(self) -> None:
        norm_finding = self.finding.strip().lower()
        if not norm_finding:
            raise ValueError("VLMClaim finding must be a non-empty string.")
        object.__setattr__(self, "finding", norm_finding)
        norm_status = self.status.strip().lower()
        if norm_status not in {"present", "absent", "uncertain"}:
            raise ValueError(f"VLMClaim status must be present/absent/uncertain; got '{self.status}'.")
        object.__setattr__(self, "status", norm_status)
        if not isinstance(self.region_description, str) or not self.region_description.strip():
            raise ValueError("VLMClaim region_description must be a non-empty string.")
        if self.box_xyxy is not None:
            if len(self.box_xyxy) != 4:
                raise ValueError("box_xyxy must have 4 coordinates (x1, y1, x2, y2).")
            coords = tuple(float(v) for v in self.box_xyxy)
            if not all(np.isfinite(c) and 0.0 <= c <= 1.0 for c in coords):
                raise ValueError("box_xyxy coordinates must be normalized and finite in [0.0, 1.0].")
            if coords[2] <= coords[0] or coords[3] <= coords[1]:
                raise ValueError("box_xyxy must satisfy x2 > x1 and y2 > y1.")
            object.__setattr__(self, "box_xyxy", coords)

    @property
    def is_verifiable_finding(self) -> bool:
        return self.finding in SUPPORTED_FINDINGS


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """26 Nov Blueprint §5: every number and flag produced by the Judge, each with an ID."""

    record_id: str
    finding_id: str
    verdict: Verdict
    reason_codes: tuple[str, ...]
    items: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.record_id or not self.record_id.strip():
            raise ValueError("EvidenceRecord record_id must be non-empty.")
        norm_finding = self.finding_id.strip().lower()
        if norm_finding not in SUPPORTED_FINDINGS:
            raise ValueError(f"EvidenceRecord finding_id must be in {SUPPORTED_FINDINGS}; got '{self.finding_id}'.")
        object.__setattr__(self, "finding_id", norm_finding)
        if not isinstance(self.verdict, Verdict):
            raise TypeError("verdict must be Verdict.")
        codes = tuple(str(c).strip() for c in self.reason_codes if str(c).strip())
        if not codes:
            raise ValueError("EvidenceRecord requires at least one reason_code.")
        object.__setattr__(self, "reason_codes", codes)
        raw_items = dict(self.items)
        if not raw_items or any(not str(k).strip() for k in raw_items):
            raise ValueError("EvidenceRecord items must map non-empty evidence IDs to values.")
        object.__setattr__(self, "items", _freeze_mapping(raw_items))


@dataclass(frozen=True, slots=True)
class ExplanationSentence:
    """26 Nov Blueprint §5: one generated sentence tagged with supporting evidence IDs."""

    text: str
    evidence_ids: tuple[str, ...]
    verified: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("ExplanationSentence text must be a non-empty string.")
        ids = tuple(str(eid).strip() for eid in self.evidence_ids if str(eid).strip())
        if self.verified:
            validate_safe_language(self.text)
            if not ids:
                raise ValueError("Verified ExplanationSentence must cite at least one evidence ID.")
        object.__setattr__(self, "evidence_ids", ids)
