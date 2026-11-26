"""Experiment 7: Hand-reviewed shortcut / chest-tube audit metrics (Blueprint 26 Nov §9)."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def compute_shortcut_audit(cases: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute flag rate and false-accept rate across manually reviewed shortcut-cue cases.

    Each item has ``cue_category`` (e.g. ``"chest_tube"``), ``has_shortcut_cue`` (bool),
    ``verdict`` (str), and optional ``y_true`` (0/1).
    """
    by_cue: dict[str, dict[str, int]] = {}
    total_cue = 0
    flagged_cue = 0
    false_accepted = 0

    for item in cases:
        if not bool(item.get("has_shortcut_cue")):
            continue
        cat = str(item.get("cue_category") or "unspecified_cue")
        st = by_cue.setdefault(cat, {"total": 0, "flagged_shortcut_or_indeterminate": 0, "accepted_model_supported": 0})
        st["total"] += 1
        total_cue += 1
        verdict = str(item.get("verdict"))
        if verdict in {"SHORTCUT RISK", "INDETERMINATE"}:
            st["flagged_shortcut_or_indeterminate"] += 1
            flagged_cue += 1
        elif verdict == "MODEL-SUPPORTED":
            st["accepted_model_supported"] += 1
            false_accepted += 1

    for st in by_cue.values():
        n = max(1, st["total"])
        st["flag_rate"] = st["flagged_shortcut_or_indeterminate"] / float(n)
        st["false_accept_rate"] = st["accepted_model_supported"] / float(n)

    return {
        "total_reviewed_cue_cases": total_cue,
        "overall_flag_rate": (flagged_cue / float(total_cue)) if total_cue else None,
        "overall_false_accept_rate": (false_accepted / float(total_cue)) if total_cue else None,
        "by_cue_category": by_cue,
        "limitation": "Measures only the manually reviewed cue categories; never claims VeriRay detects all shortcuts.",
    }
