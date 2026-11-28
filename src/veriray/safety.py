"""Safety guardrails, prompt-injection filtering, and grounded chat (Blueprint 26 Nov §8).

Implements the 5 guardrail layers:
1. Input filter for prompt-injection, diagnosis, dosage, treatment, and extraction attacks.
2. Isolation of untrusted strings (filenames, EXIF comments, burned-in image text never reach prompts).
3. Grounded generation / reviewed educational answers only.
4. Rule-based output verification (via :mod:`veriray.verifier`).
5. Fixed template fallback on any failure or refusal.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Iterable

from veriray.contracts import RESEARCH_DISCLAIMER, EvidenceRecord, Verdict, validate_safe_language

# ---------------------------------------------------------------------------
# Layer 1 normalisation helpers
# ---------------------------------------------------------------------------

_MAX_PROMPT_LEN: int = 500

# Confusable-to-ASCII skeleton map (Cyrillic, Ukrainian, Greek/Coptic, IPA/Latin lookalikes + fullwidth ASCII).
_CONFUSABLE_MAP: dict[int, str] = {
    # Cyrillic + Ukrainian/Extended Cyrillic homoglyphs
    0x0430: "a", 0x0432: "b", 0x0433: "r", 0x0435: "e", 0x0454: "e",
    0x0456: "i", 0x0457: "i", 0x0458: "j", 0x043A: "k", 0x043C: "m",
    0x043D: "h", 0x043E: "o", 0x0440: "p", 0x0441: "c", 0x0442: "t",
    0x0443: "y", 0x0445: "x", 0x0455: "s", 0x0501: "d", 0x051B: "q",
    0x051D: "w", 0x04BB: "h", 0x04CF: "l",
    0x0410: "A", 0x0412: "B", 0x0415: "E", 0x0404: "E", 0x0406: "I",
    0x0407: "I", 0x0408: "J", 0x041A: "K", 0x041C: "M", 0x041D: "H",
    0x041E: "O", 0x0420: "P", 0x0421: "C", 0x0422: "T", 0x0423: "Y",
    0x0425: "X", 0x0405: "S",
    # Greek & Coptic homoglyphs (NFKC does not fold cross-script Greek to Latin)
    0x03B1: "a", 0x03B2: "b", 0x03B5: "e", 0x03B7: "n", 0x03B9: "i",
    0x03BA: "k", 0x03BD: "v", 0x03BF: "o", 0x03C1: "p", 0x03C4: "t",
    0x03C5: "u", 0x03C7: "x", 0x03F5: "e", 0x03F2: "c",
    0x0391: "A", 0x0392: "B", 0x0395: "E", 0x0396: "Z", 0x0397: "H",
    0x0399: "I", 0x039A: "K", 0x039C: "M", 0x039D: "N", 0x039F: "O",
    0x03A1: "P", 0x03A4: "T", 0x03A5: "Y", 0x03A7: "X",
    # Common Latin/IPA small-cap & script lookalikes
    0x0251: "a", 0x0261: "g", 0x0269: "i", 0x026A: "i", 0x1D0F: "o",
    0x1D1C: "u", 0x1D20: "v", 0x0131: "i",
    # Fullwidth ASCII (U+FF01 … U+FF5E → ! … ~)
    **{0xFF01 + i: chr(0x21 + i) for i in range(94)},
}


def _normalize_prompt(text: str) -> str:
    """NFKC-normalize, strip Unicode format chars (Cf), map confusables, collapse whitespace, lowercase."""
    # 1. NFKC normalization (decomposes fullwidth, ligatures, etc.)
    text = unicodedata.normalize("NFKC", text)
    # 2. Strip Unicode format characters (category Cf: zero-width, soft hyphen, BOM, etc.)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    # 3. Map confusables to ASCII skeleton
    text = text.translate(_CONFUSABLE_MAP)
    # 4. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # 5. Lowercase
    return text.lower()


# Role-marker lines: "system:", "assistant:", "user:" at the start of a line.
_ROLE_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:system|assistant|user)\s*:", re.IGNORECASE
)

REFUSAL_MEDICAL_ADVICE = (
    "Request refused: VeriRay tests chest X-ray model behaviour for research and education only "
    "and cannot answer personal health, clinical care, or pharmacy questions. "
    f"{RESEARCH_DISCLAIMER}"
)
REFUSAL_INJECTION_OR_SCOPE = (
    "Request blocked by safety guardrails: I can only answer questions about this test's recorded "
    "evidence or general educational definitions of VeriRay's metrics. "
    f"{RESEARCH_DISCLAIMER}"
)

# Layer 1 & 4 patterns: clinical advice, drugs/dosages, direct diagnostic claims, and prompt injections.
_DRUG_OR_DOSAGE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:mg|mcg|ml|iu|tablet|tablets|capsule|capsules|pill|pills|dose|dosage|prescribe|prescription)\b",
        r"\b(?:aspirin|ibuprofen|paracetamol|acetaminophen|amoxicillin|azithromycin|prednisone|steroid|antibiotic|heparin|furosemide|morphine|insulin|albuterol|inhaler)\b",
        r"\b(?:alteplase|tenecteplase|tpa|warfarin|apixaban|rivaroxaban|clopidogrel|enoxaparin|levetiracetam|keppra|temozolomide|dexamethasone|mannitol|phenytoin|valproate|bevacizumab)\b",
        r"\b(?:isoniazid|rifampin|rifampicin|ethambutol|pyrazinamide|ceftriaxone|vancomycin|lisinopril|metoprolol|norepinephrine|epinephrine)\b",
        r"\b(?:take|administer|inject|start)\s+\d+\s*(?:mg|ml|times|pills?)\b",
    )
)

_CLINICAL_ACTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\byou\s+(?:have|definitely\s+have|likely\s+have|suffer\s+from|do\s+not\s+have)\b",
        r"\b(?:the\s+)?patient\s+(?:has|needs|requires|should\s+receive|is\s+(?:healthy|normal|disease[\s-]free))\b",
        r"\b(?:should\s+i|what\s+should\s+i|do\s+i\s+need|do\s+i\s+have)\b",
        r"\b(?:am\s+i|is\s+(?:the\s+)?patient)\s+(?:safe|okay|ok|fine|healthy|normal|out\s+of\s+danger|in\s+danger|going\s+to\s+(?:die|be\s+ok|survive))\b",
        r"\b(?:can\s+i|safe\s+to)\s+(?:safely\s+)?(?:go\s+home|be\s+discharged|skip\s+the\s+(?:doctor|hospital|er)|wait\s+until\s+tomorrow)\b",
        r"\b(?:chest\s+tube\s+(?:insertion|placement)|needle\s+decompression|thoracostomy|craniotomy|thrombectomy|thrombolysis|chemotherapy|radiotherapy|radiation\s+therapy|biopsy|intubat(?:e|ion)|surgery|hospitalize|go\s+to\s+(?:the\s+)?er)\b",
        r"\b(?:treat|cure|medicine|medication|therapy|prognosis|clinical\s+management)\b",
    )
)

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore(?:[\s_]+(?:all|any|previous|prior|above|system))*(?:[\s_]+(?:instructions?|rules|constraints))",
        r"disregard(?:[\s_]+your)?[\s_]+(?:rules|instructions|guardrails|safety)",
        r"(?:reveal|print|dump|show|output|repeat)[\s_]+(?:your[\s_]+)?(?:system[\s_]+prompt|hidden[\s_]+prompt|instructions|api[\s_]+key|secrets?)",
        r"\b(?:jailbreak|developer[\s_]+(?:mode|bypass)|dan[\s_]+mode|bypass[\s_]+safety|system[\s_]+override|unrestricted)\b",
        r"override[\s_]+(?:(?:the|all)[\s_]+)?(?:verdict|templates)",
        r"(?:output|print|verdict[\s_]+(?:as|is))[\s_]+proven\b",
        r"follow[\s_]+(?:the[\s_]+)?(?:instruction|it[\s_]+as[\s_]+a[\s_]+command)",
        r"(?:pretend|act[\s_]+as[\s_]+if)[\s_]+you[\s_]+are[\s_]+(?:a[\s_]+doctor|not[\s_]+bound|unrestricted)",
        r"(?:extract|dump|show)[\s_]+(?:patient[\s_]+name|patient[\s_]+id|mrn|ssn|exif|metadata|personal[\s_]+data)",
        r"<\s*script\b|;\s*drop\s+table\b|\{\{.*?\}\}",
        # Multilingual injection attempts (Spanish, French, German, Hindi transliterated/Devanagari)
        r"ignora\s+las\s+instrucciones|dame\s+un\s+diagn[oó]stico|receta\s+medicamento|dosis\s+de\s+medicamento",
        r"ignorez\s+les\s+instructions|donne[\-\s]moi\s+un\s+diagnostic|prescrivez",
        r"ignoriere\s+(?:alle\s+)?anweisungen|gib\s+mir\s+eine\s+diagnose|medikamente",
        r"पिछले\s+निर्देशों\s+को\s+अनदेखा|निदान\s+बताओ|दवा\s+बताओ|खुराक\s+लिखो",
    )
)

# Reviewed educational answers (fixed templates only; no dynamic medical advice).
_EDUCATION_TOPICS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("what is pneumothorax", "pneumothorax mean", "define pneumothorax", "explain pneumothorax"),
        "Educational overview: Pneumothorax is the medical term for air in the pleural space between the lung and chest wall. "
        "VeriRay only tests whether a classifier's model score for that label depends on a proposed image region; this is not a diagnosis. "
        f"{RESEARCH_DISCLAIMER}",
    ),
    (
        ("what is veriy", "what is veriray", "what does veriray do", "how does veriray work"),
        "VeriRay stress-tests a chest X-ray classifier's pneumothorax prediction by editing the proposed image region, "
        "comparing the score drop against matched control regions, and checking stability and integrity gates before showing a verdict. "
        f"{RESEARCH_DISCLAIMER}",
    ),
    (
        ("what are matched controls", "what does matched controls mean", "controls mean", "why controls"),
        "Matched controls are up to 19 non-overlapping regions of similar area and shape to the candidate region. "
        "If editing the candidate region drops the model score much more than editing matched controls, the prediction is more specific to that region—though controls are not guaranteed negatives and empirical ranks do not claim formal statistical significance. "
        f"{RESEARCH_DISCLAIMER}",
    ),
    (
        ("indeterminate mean", "shortcut risk mean", "not supported mean", "model-supported mean", "verdict mean"),
        "VeriRay uses four verdicts only: MODEL-SUPPORTED (all gates passed and target effect exceeded matched controls), "
        "NOT SUPPORTED (adequate test, effect did not meet criteria; disease absence is not implied), "
        "SHORTCUT RISK (out-of-lung or audited cue reliance flagged), and INDETERMINATE (a quality, localization, edit, or stability gate failed). "
        f"{RESEARCH_DISCLAIMER}",
    ),
)


@dataclass(frozen=True)
class SafetyCheckResult:
    safe: bool
    category: str
    reason_codes: tuple[str, ...]


def check_output_safety(text: str) -> SafetyCheckResult:
    """Strict output filter: calls ``validate_safe_language`` plus drug, dosage, and clinical-action checks."""
    reasons: list[str] = []
    try:
        validate_safe_language(text)
    except ValueError as exc:
        reasons.append(f"LANGUAGE_GUARD:{exc}")
    for pat in _DRUG_OR_DOSAGE_PATTERNS:
        if pat.search(text):
            reasons.append("FORBIDDEN_DRUG_OR_DOSAGE_TEXT")
            break
    for pat in _CLINICAL_ACTION_PATTERNS:
        if pat.search(text):
            reasons.append("FORBIDDEN_CLINICAL_ACTION_TEXT")
            break
    return SafetyCheckResult(not reasons, "output_ok" if not reasons else "output_blocked", tuple(reasons))


def classify_user_prompt(prompt: str) -> SafetyCheckResult:
    """Layer 1 input filter — normalizes then checks for injections, PII, diagnosis/dosage/treatment.

    Hardening sequence (applied before pattern matching):
      1. Hard length cap (500 characters raw) → PROMPT_TOO_LONG
      2. NFKC normalization + Unicode format-char (Cf) stripping
      3. Confusable-to-ASCII mapping (Cyrillic homoglyphs, fullwidth)
      4. Whitespace collapse + lowercase
      5. Role-marker line detection (system:, assistant:, user:)
      6. All injection / drug / clinical-action patterns on the normalized text
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return SafetyCheckResult(False, "empty", ("EMPTY_PROMPT",))
    # Hard length cap (applied to raw prompt before any normalization)
    if len(prompt) > _MAX_PROMPT_LEN:
        return SafetyCheckResult(False, "injection", ("PROMPT_TOO_LONG",))
    # Normalize for pattern matching
    normalized = _normalize_prompt(prompt)
    reasons: list[str] = []
    # Role-marker injection check (on both raw and normalized to catch multi-line variants)
    if _ROLE_MARKER_RE.search(prompt) or _ROLE_MARKER_RE.search(normalized):
        reasons.append("PROMPT_INJECTION_OR_EXTRACTION")
    # Injection patterns on normalized text
    if not reasons:
        for pat in _INJECTION_PATTERNS:
            if pat.search(normalized):
                reasons.append("PROMPT_INJECTION_OR_EXTRACTION")
                break
    for pat in _DRUG_OR_DOSAGE_PATTERNS:
        if pat.search(normalized):
            reasons.append("MEDICAL_DRUG_OR_DOSAGE_REQUEST")
            break
    for pat in _CLINICAL_ACTION_PATTERNS:
        if pat.search(normalized):
            reasons.append("MEDICAL_ADVICE_OR_DIAGNOSIS_REQUEST")
            break
    if any(w in normalized for w in ("diagnos", "triage", "prescription", "cure me", "what disease do i")):
        if "what does" not in normalized and "not a diagnosis" not in normalized:
            reasons.append("MEDICAL_DIAGNOSIS_REQUEST")
    if reasons:
        cat = "injection" if "PROMPT_INJECTION_OR_EXTRACTION" in reasons else "medical_refusal"
        return SafetyCheckResult(False, cat, tuple(dict.fromkeys(reasons)))
    return SafetyCheckResult(True, "allowed", ("PROMPT_OK",))


def sanitize_for_prompt(text: str) -> str:
    """Strip control chars and untrusted markup so filenames/EXIF/OCR never act as instructions."""
    cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", str(text))
    return re.sub(r"\s+", " ", cleaned).strip()[:200]


@dataclass(frozen=True)
class ChatResponse:
    allowed: bool
    answer: str
    cited_evidence_ids: tuple[str, ...]
    refusal_reason_codes: tuple[str, ...] = ()


def answer_chat_question(
    question: str,
    evidence: EvidenceRecord | None = None,
    *,
    unverified_findings: Iterable[str] = (),
) -> ChatResponse:
    """Grounded chat: answers only from EvidenceRecord + reviewed educational templates; refuses all else."""
    check = classify_user_prompt(question)
    if not check.safe:
        msg = REFUSAL_INJECTION_OR_SCOPE if check.category == "injection" else REFUSAL_MEDICAL_ADVICE
        return ChatResponse(False, msg, (), check.reason_codes)

    q = question.lower()
    # 1. Reviewed educational definitions first only when clearly definitional ("what is/are/does ... mean")
    is_definitional = any(p in q for p in ("what is ", "what are ", "what does ", "how does ", "define ", "explain ", " mean"))
    if is_definitional:
        for triggers, answer in _EDUCATION_TOPICS:
            if any(t in q for t in triggers):
                return ChatResponse(True, answer, ())

    # 2. Evidence-grounded questions about the current run
    if evidence is not None:
        items = evidence.items
        if any(k in q for k in ("score", "drop", "control", "rank", "verdict", "why", "gate", "shortcut", "reason", "evidence", "stability")):
            if evidence.verdict is Verdict.INDETERMINATE:
                ans = (
                    f"The verdict is INDETERMINATE [E02_VERDICT] with reason codes {', '.join(evidence.reason_codes)} [E03_REASON_CODES]. "
                    f"No further explanation is generated when a gate fails. {RESEARCH_DISCLAIMER}"
                )
                return ChatResponse(True, ans, ("E02_VERDICT", "E03_REASON_CODES"))

            parts: list[str] = []
            cites: list[str] = []
            finding_disp = evidence.finding_id.replace("_", " ")
            if any(k in q for k in ("verdict", "why", "evidence")):
                parts.append(f"The verdict for {finding_disp} is {evidence.verdict.value} [E02_VERDICT].")
                cites.append("E02_VERDICT")
            if any(k in q for k in ("score", "drop", "why", "evidence")):
                parts.append(f"The uncalibrated Reader model score was {items.get('E01_MODEL_SCORE_RAW')} [E01_MODEL_SCORE_RAW].")
                cites.append("E01_MODEL_SCORE_RAW")
                if items.get("E04_TARGET_SCORE_DROP_PRIMARY") is not None:
                    parts.append(f"The target-region edit changed the model score by {items['E04_TARGET_SCORE_DROP_PRIMARY']:.4f} [E04_TARGET_SCORE_DROP_PRIMARY].")
                    cites.append("E04_TARGET_SCORE_DROP_PRIMARY")
            if any(k in q for k in ("control", "rank", "why", "evidence")) and items.get("E07_CONTROL_SCORE_DROP_MEDIAN") is not None:
                parts.append(
                    f"Across {items.get('E06_CONTROL_COUNT')} matched controls [E06_CONTROL_COUNT], median score drop was "
                    f"{items['E07_CONTROL_SCORE_DROP_MEDIAN']:.4f} [E07_CONTROL_SCORE_DROP_MEDIAN] "
                    f"(empirical tail fraction {items.get('E08_EMPIRICAL_TAIL_FRACTION')}, not formal statistical significance) [E08_EMPIRICAL_TAIL_FRACTION]."
                )
                cites.extend(["E06_CONTROL_COUNT", "E07_CONTROL_SCORE_DROP_MEDIAN", "E08_EMPIRICAL_TAIL_FRACTION"])
            if any(k in q for k in ("shortcut", "gate", "stability")) and items.get("E11_MASK_MIN_STABILITY_IOU") is not None:
                parts.append(
                    f"Minimum mask stability IoU was {items['E11_MASK_MIN_STABILITY_IOU']:.3f} [E11_MASK_MIN_STABILITY_IOU] "
                    f"and unrelated-label max drift was {items.get('E12_UNRELATED_LABEL_MAX_DRIFT', 0.0):.4f} [E12_UNRELATED_LABEL_MAX_DRIFT]."
                )
                cites.extend(["E11_MASK_MIN_STABILITY_IOU", "E12_UNRELATED_LABEL_MAX_DRIFT"])
            if unverified_findings and "other" in q:
                names = ", ".join(sanitize_for_prompt(f) for f in unverified_findings)
                parts.append(f"Other claimant mentions ({names}) are unverified and are not explained.")
            if parts:
                parts.append(RESEARCH_DISCLAIMER)
                ans = " ".join(parts)
                if check_output_safety(ans).safe:
                    return ChatResponse(True, ans, tuple(dict.fromkeys(cites)))

    # 3. Fallback to reviewed educational topics even without "what is" phrasing
    for triggers, answer in _EDUCATION_TOPICS:
        if any(t in q for t in triggers):
            return ChatResponse(True, answer, ())

    return ChatResponse(False, REFUSAL_INJECTION_OR_SCOPE, (), ("OUT_OF_GROUNDED_SCOPE",))
