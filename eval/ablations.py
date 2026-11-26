"""Ablations A-G and Fill-Method Sensitivity (Blueprint 26 Nov §9 Exp 2 & 3).

Configurations:
  A. Reader only
  B. Reader + Grad-CAM display (same selective score criterion as A; isolates display effect)
  C. Reader + single target edit
  D. C + random controls (approximated by control median without zone/area matching if logged)
  E. D + matched controls
  F. E + stability & integrity gates (Full Judge)
  G. Alternative fill methods (telea vs gaussian_blur vs mean_fill) + verdict flip rate
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from eval.risk_coverage import evaluate_selective_point, reader_threshold_at_coverage


def run_ablations(
    records: list[Mapping[str, Any]],
    *,
    min_target_drop: float = 0.03,
    min_margin: float = 0.02,
    n_boot: int = 300,
    seed: int = 20261126,
) -> dict[str, Any]:
    """Compute Configurations A-F from per-case Judge records."""
    n = len(records)
    if n == 0:
        return {"n": 0, "configurations": []}

    scores = [float(r["score"]) for r in records]
    # Full Judge F acceptance mask
    mask_f = [bool(r.get("judge_accepted")) for r in records]
    cov_f = sum(mask_f) / float(n)
    _, mask_a = reader_threshold_at_coverage(scores, cov_f if cov_f > 0 else 0.5)
    mask_b = list(mask_a)  # Grad-CAM display does not alter automated score selection
    mask_c = [float(r.get("target_drop", 0.0)) >= min_target_drop for r in records]
    mask_d = [
        bool(c_ok and (float(r.get("target_drop", 0.0)) - float(r.get("random_control_median_drop", r.get("control_median_drop", 0.0)))) >= min_margin)
        for c_ok, r in zip(mask_c, records)
    ]
    mask_e = [
        bool(c_ok and (float(r.get("target_drop", 0.0)) - float(r.get("control_median_drop", 0.0))) >= min_margin)
        for c_ok, r in zip(mask_c, records)
    ]

    configs = [
        ("A. Reader only (matched coverage)", mask_a),
        ("B. Reader + Grad-CAM display", mask_b),
        ("C. Reader + single target edit", mask_c),
        ("D. C + uncalibrated/random controls", mask_d),
        ("E. D + matched controls", mask_e),
        ("F. E + stability & integrity gates (Full Judge)", mask_f),
    ]
    table = []
    for name, m in configs:
        pt = evaluate_selective_point(records, accepted_mask=m, n_boot=n_boot, seed=seed)
        table.append({"configuration": name, **pt})
    return {"n": n, "configurations": table}


def compute_fill_sensitivity(
    cases: Iterable[Mapping[str, Mapping[str, float]]],
    *,
    methods: tuple[str, ...] = ("telea", "gaussian_blur", "mean_fill"),
    min_target_drop: float = 0.03,
) -> dict[str, Any]:
    """Experiment 3 (Config G): measure verdict flip rate across fill methods."""
    total = 0
    flips = 0
    per_method_accepts = {m: 0 for m in methods}
    for item in cases:
        drops = item.get("per_method_score_drops") or {}
        if not all(m in drops for m in methods):
            continue
        total += 1
        decisions = {m: float(drops[m]) >= min_target_drop for m in methods}
        for m, ok in decisions.items():
            if ok:
                per_method_accepts[m] += 1
        if len(set(decisions.values())) > 1:
            flips += 1

    return {
        "n": total,
        "methods": list(methods),
        "accept_counts": per_method_accepts,
        "flip_count": flips,
        "verdict_flip_rate": (flips / float(total)) if total else None,
    }
