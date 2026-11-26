"""Experiment 13: Controlled phone-photo capture robustness pilot (Blueprint §9, v2 §11).

Keeps original scan and phone-photo captures paired by ``source_image_id``, measuring:
- Input Gate acceptance vs. RETAKE/REJECT abstention counts
- Verdict-flip counts between original scan and phone photograph
- Reader score shift

Strictly described as a controlled capture engineering pilot, NEVER as evidence of
clinical performance in Indian hospitals.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np


def evaluate_capture_pilot(paired_cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize paired (original_scan, phone_photo) outcomes grouped by source_image_id."""
    cases = list(paired_cases)
    n = len(cases)
    if n == 0:
        return {"n_pairs": 0}

    gate_rejected_orig = 0
    gate_rejected_phone = 0
    verdict_flips = 0
    score_deltas: list[float] = []
    source_ids: set[str] = set()

    for c in cases:
        source_ids.add(str(c.get("source_image_id") or id(c)))
        if not bool(c.get("orig_gate_accepted", True)):
            gate_rejected_orig += 1
        if not bool(c.get("phone_gate_accepted", True)):
            gate_rejected_phone += 1
        if str(c.get("orig_verdict")) != str(c.get("phone_verdict")):
            verdict_flips += 1
        if c.get("orig_score") is not None and c.get("phone_score") is not None:
            score_deltas.append(abs(float(c["phone_score"]) - float(c["orig_score"])))

    return {
        "n_pairs": n,
        "unique_source_images": len(source_ids),
        "orig_gate_abstained": gate_rejected_orig,
        "phone_gate_abstained": gate_rejected_phone,
        "phone_gate_abstention_rate": gate_rejected_phone / float(n),
        "verdict_flips": verdict_flips,
        "verdict_flip_rate": verdict_flips / float(n),
        "mean_abs_score_delta": float(np.mean(score_deltas)) if score_deltas else None,
        "limitation": "Controlled capture engineering pilot only; NOT evidence of clinical performance in Indian hospitals.",
    }
