"""Reader, localization, integrity, shortcut, latency, and subgroup metrics (Blueprint §9)."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def compute_reader_metrics(
    y_true: Iterable[int],
    scores: Iterable[float],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute AUROC, AUPRC, Brier score, and thresholded confusion metrics."""
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    n = int(len(y))
    if n == 0 or len(s) != n:
        return {"n": 0, "auroc": None, "auprc": None, "brier": None}

    has_both = len(np.unique(y)) == 2
    auroc = float(roc_auc_score(y, s)) if has_both else None
    auprc = float(average_precision_score(y, s)) if has_both else None
    brier = float(brier_score_loss(y, np.clip(s, 0.0, 1.0)))

    pred = (s >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    return {
        "n": n,
        "positives": int(np.sum(y == 1)),
        "threshold": float(threshold),
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
        "sensitivity": tp / float(tp + fn) if (tp + fn) > 0 else None,
        "specificity": tn / float(tn + fp) if (tn + fp) > 0 else None,
        "ppv": tp / float(tp + fp) if (tp + fp) > 0 else None,
        "npv": tn / float(tn + fn) if (tn + fn) > 0 else None,
    }


def compute_localization_metrics(
    cases: Iterable[Mapping[str, Any]],
    *,
    iou_success_threshold: float = 0.10,
) -> dict[str, Any]:
    """Experiment 6: IoU, pointing-game accuracy, and localization success on eligible annotated cases.

    Each case dict has ``heatmap`` (2D), ``mask`` (2D bool), and ``gt_boxes_xyxy`` (list of (x1,y1,x2,y2)
    in the same pixel coordinates, or empty if unannotated). Only cases with >=1 GT box are eligible.
    """
    ious: list[float] = []
    pointing_hits = 0
    success_hits = 0
    eligible = 0
    total_seen = 0

    for item in cases:
        total_seen += 1
        boxes = item.get("gt_boxes_xyxy") or []
        if not boxes:
            continue
        eligible += 1
        hm = np.asarray(item["heatmap"], dtype=float)
        mask = np.asarray(item["mask"], dtype=bool)
        h, w = mask.shape

        gt_mask = np.zeros((h, w), dtype=bool)
        for x1, y1, x2, y2 in boxes:
            ix1, iy1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
            ix2, iy2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
            if ix2 > ix1 and iy2 > iy1:
                gt_mask[iy1:iy2, ix1:ix2] = True

        inter = int(np.logical_and(mask, gt_mask).sum())
        union = int(np.logical_or(mask, gt_mask).sum())
        iou = 0.0 if union == 0 else inter / float(union)
        ious.append(iou)
        if iou >= iou_success_threshold:
            success_hits += 1

        peak_y, peak_x = np.unravel_index(int(np.argmax(hm)), hm.shape)
        if gt_mask[peak_y, peak_x]:
            pointing_hits += 1

    return {
        "total_cases_seen": total_seen,
        "eligible_annotated_cases": eligible,
        "iou_success_threshold": float(iou_success_threshold),
        "mean_iou": float(np.mean(ious)) if ious else None,
        "median_iou": float(np.median(ious)) if ious else None,
        "pointing_game_accuracy": pointing_hits / float(eligible) if eligible else None,
        "localization_success_rate": success_hits / float(eligible) if eligible else None,
        "pointing_hits": pointing_hits,
        "success_hits": success_hits,
    }


def summarize_latencies(records: Iterable[Mapping[str, Mapping[str, float]]]) -> dict[str, dict[str, float]]:
    """Experiment 10: per-stage and end-to-end p50/p95 latency (ms)."""
    by_stage: dict[str, list[float]] = {}
    totals: list[float] = []
    for rec in records:
        lat = rec.get("latencies_ms") or {}
        if not lat:
            continue
        total = 0.0
        for stage, ms in lat.items():
            val = float(ms)
            by_stage.setdefault(stage, []).append(val)
            total += val
        totals.append(total)

    out: dict[str, dict[str, float]] = {}
    for stage, vals in sorted(by_stage.items()):
        arr = np.asarray(vals, dtype=float)
        out[stage] = {"count": float(len(arr)), "p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95))}
    if totals:
        arr = np.asarray(totals, dtype=float)
        out["end_to_end"] = {"count": float(len(arr)), "p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95))}
    return out


def compute_subgroup_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    group_key: str,
    verdict_key: str = "verdict",
    accepted_verdict: str = "MODEL-SUPPORTED",
) -> dict[str, Any]:
    """Experiment 12: Reader AUROC and acceptance rate by sex or age band (never claims zero bias)."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in records:
        g = str(r.get(group_key) or "UNKNOWN")
        groups.setdefault(g, []).append(r)

    table: dict[str, dict[str, Any]] = {}
    for g, items in sorted(groups.items()):
        y = [int(x["y_true"]) for x in items]
        s = [float(x["score"]) for x in items]
        rm = compute_reader_metrics(y, s)
        acc = [x for x in items if str(x.get(verdict_key)) == accepted_verdict]
        prec = (sum(int(x["y_true"]) for x in acc) / float(len(acc))) if acc else None
        table[g] = {
            "n": len(items),
            "accepted_count": len(acc),
            "acceptance_rate": len(acc) / float(len(items)),
            "accepted_precision": prec,
            "reader_auroc": rm["auroc"],
        }

    return {
        "group_key": group_key,
        "groups": table,
        "limitations": "Measured subgroup gaps on available metadata only; no race data; does not claim zero bias.",
    }
