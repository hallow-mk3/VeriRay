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

from veriray.contracts import SUPPORTED_FINDING, VERDICT_USER_LANGUAGE, EvidenceRecord, ExplanationSentence, Verdict
from veriray.safety import check_output_safety

# Other thoracic findings that must NEVER be explained unless verified (only pneumothorax is verified).
_OTHER_FINDINGS: tuple[str, ...] = (
    "atelectasis", "consolidation", "infiltration", "edema", "emphysema", "fibrosis",
    "effusion", "pneumonia", "pleural thickening", "pleural_thickening", "cardiomegaly",
    "nodule", "mass", "hernia", "lung lesion", "fracture", "lung opacity",
    "enlarged cardiomediastinum", "tuberculosis", "covid", "cancer", "tumor",
)

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
    for other in _OTHER_FINDINGS:
        if re.search(rf"\b{re.escape(other)}\b", lower):
            reasons.append(f"UNVERIFIED_FINDING_MENTIONED:{other}")
            break

    if _META_INSTRUCTION_RE.search(sentence.text):
        reasons.append("INSTRUCTION_META_LANGUAGE_DETECTED")

    if evidence.verdict is not Verdict.MODEL_SUPPORTED:
        if re.search(rf"\b{SUPPORTED_FINDING}\b", lower) and "not supported" not in lower and "shortcut" not in lower and "could not produce" not in lower:
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
