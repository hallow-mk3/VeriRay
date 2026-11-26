"""Risk-coverage curves, matched-coverage comparison, and patient-level bootstrap CIs (Blueprint §9.1-9.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def patient_bootstrap_ci(
    records: list[Mapping[str, Any]],
    metric_fn,
    *,
    n_boot: int = 500,
    seed: int = 20261126,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    """Cluster bootstrap over ``patient_id`` so derived/multiple scans stay grouped together."""
    if not records:
        return None, None
    by_patient: dict[str, list[Mapping[str, Any]]] = {}
    for r in records:
        pid = str(r.get("patient_id") or r.get("sample_id") or id(r))
        by_patient.setdefault(pid, []).append(r)
    pids = list(by_patient.keys())
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(pids), size=len(pids))
        sample: list[Mapping[str, Any]] = []
        for idx in chosen:
            sample.extend(by_patient[pids[int(idx)]])
        v = metric_fn(sample)
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return None, None
    lo = float(np.percentile(vals, 100 * (alpha / 2.0)))
    hi = float(np.percentile(vals, 100 * (1.0 - alpha / 2.0)))
    return lo, hi


def _accepted_precision(recs: Iterable[Mapping[str, Any]], accepted_flag: str = "accepted") -> float | None:
    acc = [r for r in recs if bool(r.get(accepted_flag))]
    if not acc:
        return None
    return sum(int(r["y_true"]) for r in acc) / float(len(acc))


def evaluate_selective_point(
    records: list[Mapping[str, Any]],
    *,
    accepted_mask: list[bool],
    n_boot: int = 500,
    seed: int = 20261126,
    min_reliable_accepted: int = 10,
) -> dict[str, Any]:
    """Compute coverage, accepted count, precision, error (risk), and patient-level bootstrap 95% CI."""
    n = len(records)
    if n == 0 or len(accepted_mask) != n:
        return {"n": 0, "accepted_count": 0, "coverage": 0.0, "precision": None, "risk": None}

    flagged = [{**r, "_acc": bool(a)} for r, a in zip(records, accepted_mask)]
    acc = [r for r in flagged if r["_acc"]]
    k = len(acc)
    cov = k / float(n)
    prec = (sum(int(r["y_true"]) for r in acc) / float(k)) if k > 0 else None
    risk = (1.0 - prec) if prec is not None else None
    ci_lo, ci_hi = patient_bootstrap_ci(
        flagged, lambda s: _accepted_precision(s, "_acc"), n_boot=n_boot, seed=seed,
    )
    return {
        "n": n,
        "accepted_count": k,
        "abstention_count": n - k,
        "coverage": cov,
        "abstention_rate": 1.0 - cov,
        "precision": prec,
        "risk": risk,
        "precision_ci95": [ci_lo, ci_hi],
        "low_coverage_warning": k < min_reliable_accepted,
    }


def reader_threshold_at_coverage(scores: list[float], target_coverage: float) -> tuple[float, list[bool]]:
    """Select the top ``target_coverage`` fraction by Reader score for matched-coverage comparison."""
    n = len(scores)
    if n == 0 or target_coverage <= 0.0:
        return float("inf"), [False] * n
    k = max(1, min(n, int(round(target_coverage * n))))
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    mask = [False] * n
    for idx in order[:k]:
        mask[int(idx)] = True
    thresh = float(scores[int(order[k - 1])])
    return thresh, mask


def compare_reader_vs_judge(
    records: list[Mapping[str, Any]],
    *,
    target_coverages: tuple[float, ...] = (0.25, 0.50, 0.75),
    n_boot: int = 500,
    seed: int = 20261126,
) -> dict[str, Any]:
    """Compare Reader-only and Judge at the Judge's operating coverage and predeclared coverages.

    Each record needs ``patient_id``, ``y_true`` (0/1), ``score`` (float), ``judge_accepted`` (bool).
    Optional ``judge_rank_score`` orders Judge-accepted predictions when sweeping lower coverages.
    """
    n = len(records)
    if n == 0:
        return {"n": 0, "operating_point": {}, "matched_coverages": []}

    scores = [float(r["score"]) for r in records]
    judge_mask = [bool(r["judge_accepted"]) for r in records]
    judge_op = evaluate_selective_point(records, accepted_mask=judge_mask, n_boot=n_boot, seed=seed)

    # Matched comparison at the Judge's exact operating coverage
    _, reader_at_judge_cov = reader_threshold_at_coverage(scores, judge_op["coverage"])
    reader_matched = evaluate_selective_point(records, accepted_mask=reader_at_judge_cov, n_boot=n_boot, seed=seed)

    delta_prec = (
        None
        if (judge_op["precision"] is None or reader_matched["precision"] is None)
        else float(judge_op["precision"] - reader_matched["precision"])
    )

    matched_rows: list[dict[str, Any]] = []
    # Sweep target coverages (for Judge: if target <= judge_op coverage, rank accepted cases by score/margin)
    for cov in target_coverages:
        _, r_mask = reader_threshold_at_coverage(scores, cov)
        r_pt = evaluate_selective_point(records, accepted_mask=r_mask, n_boot=n_boot, seed=seed)

        if cov <= judge_op["coverage"] + 1e-9 and judge_op["accepted_count"] > 0:
            # Rank only within Judge-accepted cases; rejected cases remain abstained
            j_scores = [
                float(r.get("judge_rank_score", r["score"])) if j_ok else -1e9
                for r, j_ok in zip(records, judge_mask)
            ]
            k_target = max(1, min(judge_op["accepted_count"], int(round(cov * n))))
            order = np.argsort(-np.asarray(j_scores, dtype=float), kind="mergesort")
            j_mask = [False] * n
            for idx in order[:k_target]:
                if judge_mask[int(idx)]:
                    j_mask[int(idx)] = True
            j_pt = evaluate_selective_point(records, accepted_mask=j_mask, n_boot=n_boot, seed=seed)
            j_pt["attainable"] = True
        else:
            j_pt = {"attainable": False, "coverage": judge_op["coverage"], "accepted_count": judge_op["accepted_count"]}

        matched_rows.append({"target_coverage": float(cov), "reader_only": r_pt, "full_judge": j_pt})

    return {
        "n": n,
        "operating_point": {
            "full_judge": judge_op,
            "reader_at_same_coverage": reader_matched,
            "precision_delta_judge_minus_reader": delta_prec,
        },
        "matched_coverages": matched_rows,
    }


def plot_risk_coverage(
    records: list[Mapping[str, Any]],
    output_path: Path | str,
    *,
    title: str = "Reader-only vs. Full Judge (Risk-Coverage)",
) -> Path:
    """Save risk-coverage plot (error rate vs coverage) with accepted counts annotated."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scores = [float(r["score"]) for r in records]
    cov_grid = np.linspace(0.10, 1.0, 10)
    r_covs, r_risks = [], []
    for c in cov_grid:
        _, m = reader_threshold_at_coverage(scores, float(c))
        pt = evaluate_selective_point(records, accepted_mask=m, n_boot=50)
        if pt["risk"] is not None:
            r_covs.append(pt["coverage"])
            r_risks.append(pt["risk"])

    j_mask = [bool(r["judge_accepted"]) for r in records]
    j_pt = evaluate_selective_point(records, accepted_mask=j_mask, n_boot=100)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(r_covs, r_risks, marker="o", label="Reader only")
    if j_pt["risk"] is not None:
        ax.scatter(
            [j_pt["coverage"]],
            [j_pt["risk"]],
            s=90,
            color="tab:green",
            zorder=5,
            label=f"Full Judge (n={j_pt['accepted_count']}/{len(records)})",
        )
    ax.set_xlabel("Coverage (accepted fraction)")
    ax.set_ylabel("Risk (1 - precision among accepted)")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
