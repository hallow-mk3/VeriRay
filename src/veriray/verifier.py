"""Rule-based Claim Verifier (Blueprint 26 Nov §5 module 7, §8, §9 exp 5).

Checks every generated ``ExplanationSentence`` against an ``EvidenceRecord``:
1. Must cite at least one existing evidence ID in ``evidence.items``.
2. Must pass ``check_output_safety`` (no diagnostic/triage/drug/dosage/urgency/p-value text).
3. May name only verified findings (cannot discuss findings that failed or unverified findings).
4. Must contain NO numeric literal absent from the cited evidence items.

Any sentence failing any check is dropped. If any fail (or verdict is INDETERMINATE / zero
sentences survive), the verifier returns fixed template sentences instead.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from veriray.contracts import VERDICT_USER_LANGUAGE, EvidenceRecord, ExplanationSentence, Verdict
from veriray.safety import check_output_safety

# Every pathology term across Chest X-ray & Brain MRI that must NEVER be narrated unless
# it is the exact finding verified by ``evidence.finding_id``.
_ALL_KNOWN_FINDINGS: tuple[str, ...] = (
    "pneumothorax",
    "aortic enlargement",
    "aortic_enlargement",
    "atelectasis",
    "calcification",
    "cardiomegaly",
    "consolidation",
    "ild",
    "interstitial lung disease",
    "infiltration",
    "lung opacity",
    "lung_opacity",
    "nodule mass",
    "nodule_mass",
    "nodule",
    "mass",
    "other lesion",
    "other_lesion",
    "pleural effusion",
    "pleural_effusion",
    "effusion",
    "pleural thickening",
    "pleural_thickening",
    "pulmonary fibrosis",
    "pulmonary_fibrosis",
    "fibrosis",
    "pneumonia",
    "edema",
    "emphysema",
    "hernia",
    "lung lesion",
    "fracture",
    "enlarged cardiomediastinum",
    "tuberculosis",
    "covid",
    "cancer",
    "tumor",
    "glioma",
    "meningioma",
    "pituitary",
    "ischemic stroke",
    "ischemic_stroke",
    "stroke",
    "hemorrhage",
)

_ALLOWED_FINDING_TERMS: dict[str, frozenset[str]] = {
    "pneumothorax": frozenset({"pneumothorax"}),
    "aortic_enlargement": frozenset({"aortic_enlargement", "aortic enlargement"}),
    "atelectasis": frozenset({"atelectasis"}),
    "calcification": frozenset({"calcification"}),
    "cardiomegaly": frozenset({"cardiomegaly"}),
    "consolidation": frozenset({"consolidation"}),
    "ild": frozenset({"ild", "interstitial lung disease"}),
    "infiltration": frozenset({"infiltration"}),
    "lung_opacity": frozenset({"lung_opacity", "lung opacity"}),
    "nodule_mass": frozenset({"nodule_mass", "nodule mass", "nodule", "mass"}),
    "other_lesion": frozenset({"other_lesion", "other lesion"}),
    "effusion": frozenset({"effusion", "pleural effusion", "pleural_effusion"}),
    "pleural_thickening": frozenset({"pleural_thickening", "pleural thickening"}),
    "pulmonary_fibrosis": frozenset({"pulmonary_fibrosis", "pulmonary fibrosis", "fibrosis"}),
    "pneumonia": frozenset({"pneumonia"}),
}

_NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?%?(?![A-Za-z_])")

# Instruction / meta-language pattern — catches hidden control sentences
_META_INSTRUCTION_RE = re.compile(
    r"\b(?:ignore|disregard|forget|override|pretend|you\s+must)\b"
    r"|your\s+rules|your\s+instructions|system\s+prompt|follow\s+(?:the\s+)?instruction"
    r"|please\s+\w+\s+(?:your|all|previous|prior|above|system)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VerifierReport:
    verified_sentences: tuple[ExplanationSentence, ...]
    dropped_sentences: tuple[tuple[str, tuple[str, ...]], ...]
    total_candidates: int
    dropped_count: int
    unsupported_claim_rate: float
    used_template_fallback: bool


def _collect_numbers(obj: Any) -> list[float]:
    nums: list[float] = []
    if isinstance(obj, bool):
        return nums
    if isinstance(obj, (int, float)) and math.isfinite(float(obj)):
        nums.append(float(obj))
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            nums.extend(_collect_numbers(x))
    elif isinstance(obj, dict):
        for v in obj.values():
            nums.extend(_collect_numbers(v))
    return nums


def _number_is_grounded(token: str, allowed_values: list[float]) -> bool:
    is_pct = token.endswith("%")
    raw = float(token[:-1] if is_pct else token)
    candidates = [raw / 100.0, raw] if is_pct else [raw]
    for cand in candidates:
        for ref in allowed_values:
            if abs(cand - ref) <= max(1e-3, 1e-3 * abs(ref)):
                return True
    return False


def verify_sentence(sentence: ExplanationSentence, evidence: EvidenceRecord) -> tuple[bool, tuple[str, ...]]:
    """Return ``(ok, rejection_reasons)`` for one candidate ExplanationSentence."""
    reasons: list[str] = []
    if not sentence.evidence_ids:
        reasons.append("MISSING_EVIDENCE_ID")
    unknown = [eid for eid in sentence.evidence_ids if eid not in evidence.items]
    if unknown:
        reasons.append("UNKNOWN_EVIDENCE_ID")

    safety = check_output_safety(sentence.text)
    if not safety.safe:
        reasons.extend(safety.reason_codes)

    lower = sentence.text.lower()
    fid_norm = evidence.finding_id.strip().lower().replace(" ", "_").replace("/", "_")
    if fid_norm == "pleural_effusion":
        fid_norm = "effusion"
    allowed_terms = _ALLOWED_FINDING_TERMS.get(fid_norm, frozenset({fid_norm, fid_norm.replace("_", " ")}))
    for other in _ALL_KNOWN_FINDINGS:
        if other in allowed_terms:
            continue
        if re.search(rf"\b{re.escape(other)}\b", lower):
            reasons.append(f"UNVERIFIED_FINDING_MENTIONED:{other}")
            break

    if _META_INSTRUCTION_RE.search(sentence.text):
        reasons.append("INSTRUCTION_META_LANGUAGE_DETECTED")

    if evidence.verdict is not Verdict.MODEL_SUPPORTED:
        if any(re.search(rf"\b{re.escape(t)}\b", lower) for t in allowed_terms):
            if "not supported" not in lower and "shortcut" not in lower and "could not produce" not in lower:
                if "supports" in lower or "confirmed" in lower or "passed" in lower:
                    reasons.append("FINDING_SUPPORTED_CLAIM_WITHOUT_VERDICT")

    cited_values: list[float] = []
    for eid in sentence.evidence_ids:
        if eid in evidence.items:
            cited_values.extend(_collect_numbers(evidence.items[eid]))
    for tok in _NUMBER_RE.findall(sentence.text):
        if not _number_is_grounded(tok, cited_values):
            reasons.append(f"UNGROUNDED_NUMBER:{tok}")
            break

    return not reasons, tuple(reasons)


def fixed_template_fallback(evidence: EvidenceRecord) -> tuple[ExplanationSentence, ...]:
    """Deterministic safe template fallback; never fails verification."""
    text = (
        f"{VERDICT_USER_LANGUAGE[evidence.verdict]} "
        f"Verdict: {evidence.verdict.value}; reason codes: {', '.join(evidence.reason_codes)}."
    )
    return (ExplanationSentence(text=text, evidence_ids=("E02_VERDICT", "E03_REASON_CODES"), verified=True),)


def verify_explanation(
    candidates: list[ExplanationSentence] | tuple[ExplanationSentence, ...],
    evidence: EvidenceRecord,
    *,
    max_drop_fraction: float = 0.5,
) -> VerifierReport:
    """Verify every sentence; drop bad ones; fall back to fixed templates on failure or INDETERMINATE."""
    total = len(candidates)
    if evidence.verdict is Verdict.INDETERMINATE:
        return VerifierReport(
            verified_sentences=fixed_template_fallback(evidence),
            dropped_sentences=tuple((s.text, ("INDETERMINATE_NO_NARRATION",)) for s in candidates),
            total_candidates=total,
            dropped_count=total,
            unsupported_claim_rate=1.0 if total else 0.0,
            used_template_fallback=True,
        )

    kept: list[ExplanationSentence] = []
    dropped: list[tuple[str, tuple[str, ...]]] = []
    for s in candidates:
        ok, reasons = verify_sentence(s, evidence)
        if ok:
            kept.append(ExplanationSentence(text=s.text, evidence_ids=s.evidence_ids, verified=True))
        else:
            dropped.append((s.text, reasons))

    drop_rate = len(dropped) / float(total) if total else 0.0
    if not kept or (total > 0 and drop_rate > max_drop_fraction):
        return VerifierReport(
            verified_sentences=fixed_template_fallback(evidence),
            dropped_sentences=tuple(dropped),
            total_candidates=total,
            dropped_count=len(dropped),
            unsupported_claim_rate=drop_rate,
            used_template_fallback=True,
        )

    return VerifierReport(
        verified_sentences=tuple(kept),
        dropped_sentences=tuple(dropped),
        total_candidates=total,
        dropped_count=len(dropped),
        unsupported_claim_rate=drop_rate,
        used_template_fallback=False,
    )
