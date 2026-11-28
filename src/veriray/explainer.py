"""Grounded Explainer (Blueprint 26 Nov §5 module 6, §8).

Receives ONLY structured data: finding, verdict, reason codes, evidence record, requested language.
NEVER receives the image, filename, EXIF/metadata, or demographic attributes (age/sex) so that
wording is strictly invariant to age and sex (§8 Fairness).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from veriray.contracts import (
    RESEARCH_DISCLAIMER,
    VERDICT_USER_LANGUAGE,
    EvidenceRecord,
    ExplanationSentence,
    VLMClaim,
    Verdict,
)
from veriray.safety import sanitize_for_prompt
from veriray.verifier import VerifierReport, fixed_template_fallback, verify_explanation


def _canon_finding_key(name: str) -> str:
    norm = name.strip().lower().replace(" ", "_").replace("/", "_")
    return "effusion" if norm == "pleural_effusion" else norm


def _finding_prose(finding_id: str) -> str:
    return _canon_finding_key(finding_id).replace("_", " ")


@dataclass(frozen=True)
class ExplainerOutput:
    sentences: tuple[ExplanationSentence, ...]
    unverified_other_claims: tuple[str, ...]  # listed as "mentioned by the model, not verified"
    disclaimer: str
    verifier_report: VerifierReport
    language: str = "en"


def build_candidate_sentences(
    evidence: EvidenceRecord,
    *,
    language: str = "en",
) -> list[ExplanationSentence]:
    """Generate structured grounded sentences from ``EvidenceRecord`` only (English MVP)."""
    if language.lower() != "en":
        # §8: English first; extra languages cut unless native-reviewed.
        return list(fixed_template_fallback(evidence))

    verdict = evidence.verdict
    items = evidence.items
    finding_label = _finding_prose(evidence.finding_id)
    if verdict is Verdict.INDETERMINATE:
        return []  # §8: no partial or unverified narration on INDETERMINATE; verifier outputs fixed template.

    if verdict is Verdict.SHORTCUT_RISK:
        return [
            ExplanationSentence(
                text=f"{VERDICT_USER_LANGUAGE[Verdict.SHORTCUT_RISK]} Reason codes: {', '.join(evidence.reason_codes)}.",
                evidence_ids=("E02_VERDICT", "E03_REASON_CODES"),
            ),
            ExplanationSentence(
                text=(
                    f"The Reader model score for {finding_label} was {items['E01_MODEL_SCORE_RAW']:.4f}; "
                    "an explanation is withheld because a non-target cue flag was raised."
                ),
                evidence_ids=("E01_MODEL_SCORE_RAW", "E02_VERDICT"),
            ),
        ]

    if verdict is Verdict.NOT_SUPPORTED:
        sents = [
            ExplanationSentence(
                text=f"{VERDICT_USER_LANGUAGE[Verdict.NOT_SUPPORTED]} Reason codes: {', '.join(evidence.reason_codes)}.",
                evidence_ids=("E02_VERDICT", "E03_REASON_CODES"),
            ),
        ]
        if items.get("E04_TARGET_SCORE_DROP_PRIMARY") is not None and items.get("E07_CONTROL_SCORE_DROP_MEDIAN") is not None:
            sents.append(
                ExplanationSentence(
                    text=(
                        f"For {finding_label}, the uncalibrated model score was {items['E01_MODEL_SCORE_RAW']:.4f}, "
                        f"and the target edit changed the score by {items['E04_TARGET_SCORE_DROP_PRIMARY']:.4f} compared with "
                        f"a median change of {items['E07_CONTROL_SCORE_DROP_MEDIAN']:.4f} across {items['E06_CONTROL_COUNT']} matched controls."
                    ),
                    evidence_ids=("E01_MODEL_SCORE_RAW", "E04_TARGET_SCORE_DROP_PRIMARY", "E06_CONTROL_COUNT", "E07_CONTROL_SCORE_DROP_MEDIAN"),
                )
            )
        else:
            sents.append(
                ExplanationSentence(
                    text=(
                        f"For {finding_label}, the uncalibrated model score was {items['E01_MODEL_SCORE_RAW']:.4f}, "
                        "which is below the positive-prediction threshold."
                    ),
                    evidence_ids=("E01_MODEL_SCORE_RAW", "E03_REASON_CODES"),
                )
            )
        if "NO_FINDING_FLAGGED_BY_READER" in evidence.reason_codes:
            sents.append(
                ExplanationSentence(
                    text=(
                        "No monitored thoracic finding exceeded the Reader threshold in this scan "
                        "(this describes model scores only, not a clinical assessment)."
                    ),
                    evidence_ids=("E01_MODEL_SCORE_RAW", "E03_REASON_CODES"),
                )
            )
        return sents

    # Verdict.MODEL_SUPPORTED
    return [
        ExplanationSentence(
            text=f"{VERDICT_USER_LANGUAGE[Verdict.MODEL_SUPPORTED]}",
            evidence_ids=("E02_VERDICT",),
        ),
        ExplanationSentence(
            text=(
                f"The Reader model score for {finding_label} was {items['E01_MODEL_SCORE_RAW']:.4f}, "
                f"and editing the candidate region lowered the model score by {items['E04_TARGET_SCORE_DROP_PRIMARY']:.4f}."
            ),
            evidence_ids=("E01_MODEL_SCORE_RAW", "E04_TARGET_SCORE_DROP_PRIMARY"),
        ),
        ExplanationSentence(
            text=(
                f"Across {items['E06_CONTROL_COUNT']} matched control regions, the median score change was "
                f"{items['E07_CONTROL_SCORE_DROP_MEDIAN']:.4f}, giving an empirical tail fraction of "
                f"{items['E08_EMPIRICAL_TAIL_FRACTION']:.3f} (not a formal statistical test)."
            ),
            evidence_ids=("E06_CONTROL_COUNT", "E07_CONTROL_SCORE_DROP_MEDIAN", "E08_EMPIRICAL_TAIL_FRACTION"),
        ),
        ExplanationSentence(
            text=(
                f"Mask stability minimum IoU was {items['E11_MASK_MIN_STABILITY_IOU']:.3f} and unrelated-label "
                f"maximum drift was {items['E12_UNRELATED_LABEL_MAX_DRIFT']:.4f}."
            ),
            evidence_ids=("E11_MASK_MIN_STABILITY_IOU", "E12_UNRELATED_LABEL_MAX_DRIFT"),
        ),
    ]


def explain_verified(
    evidence: EvidenceRecord,
    *,
    vlm_claims: Iterable[VLMClaim] = (),
    language: str = "en",
) -> ExplainerOutput:
    """Produce a rule-verified explanation; non-target claims are marked unverified only."""
    target_key = _canon_finding_key(evidence.finding_id)
    # Never explain or pass unverified findings into the narrative; tag cleanly for display.
    other = tuple(
        f"{sanitize_for_prompt(c.finding)} ({c.status}): mentioned by the model, not verified"
        for c in vlm_claims
        if not c.is_verifiable_finding or _canon_finding_key(c.finding) != target_key
    )
    candidates = build_candidate_sentences(evidence, language=language)
    report = verify_explanation(candidates, evidence)
    return ExplainerOutput(
        sentences=report.verified_sentences,
        unverified_other_claims=other,
        disclaimer=RESEARCH_DISCLAIMER,
        verifier_report=report,
        language="en" if language.lower() != "en" else language.lower(),
    )
