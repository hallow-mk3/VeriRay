#!/usr/bin/env python
"""Validation-only Judge threshold tuning and config freezing (Blueprint §7, §10, §11).

Sweeps candidate ``(min_target_drop, min_margin_over_controls, max_empirical_tail_fraction)``
triples on validation records ONLY and freezes the winning configuration to
``configs/frozen.yaml`` before the locked test set is ever opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eval.risk_coverage import evaluate_selective_point  # noqa: E402
from veriray.config import configs_from_yaml, freeze_config  # noqa: E402
from veriray.judge import JudgeConfig  # noqa: E402


def select_best_judge_thresholds(
    val_records: list[Mapping[str, Any]],
    base_cfg: JudgeConfig = JudgeConfig(),
    *,
    drop_grid: Iterable[float] = (0.02, 0.03, 0.05, 0.08),
    margin_grid: Iterable[float] = (0.01, 0.02, 0.04),
    tail_grid: Iterable[float] = (0.10, 0.15, 0.20),
    min_accepted: int = 5,
) -> tuple[JudgeConfig, dict[str, Any]]:
    """Choose thresholds maximizing validation precision among accepted cases (with min_accepted floor)."""
    best_cfg = base_cfg
    best_score = (-1.0, -1)
    best_stats: dict[str, Any] = {}

    for d in drop_grid:
        for m in margin_grid:
            for t in tail_grid:
                acc = [
                    bool(
                        float(r.get("target_drop", 0.0)) >= d
                        and (float(r.get("target_drop", 0.0)) - float(r.get("control_median_drop", 0.0))) >= m
                        and float(r.get("empirical_tail_fraction", 1.0)) <= t
                        and not bool(r.get("shortcut_or_integrity_failed", False))
                    )
                    for r in val_records
                ]
                pt = evaluate_selective_point(val_records, accepted_mask=acc, n_boot=50, min_reliable_accepted=min_accepted)
                if pt["accepted_count"] < min_accepted or pt["precision"] is None:
                    continue
                cand_score = (float(pt["precision"]), int(pt["accepted_count"]))
                if cand_score > best_score:
                    best_score = cand_score
                    best_cfg = JudgeConfig(
                        **{
                            **base_cfg.__dict__,
                            "min_target_drop": float(d),
                            "min_margin_over_controls": float(m),
                            "max_empirical_tail_fraction": float(t),
                        }
                    )
                    best_stats = {"precision": pt["precision"], "coverage": pt["coverage"], "accepted_count": pt["accepted_count"]}

    return best_cfg, best_stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-records-json", type=Path, required=True, help="JSON list of validation run records")
    ap.add_argument("--base-config", type=Path, default=ROOT / "configs" / "default.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "configs" / "frozen.yaml")
    args = ap.parse_args()

    loc_cfg, base_judge, _ = configs_from_yaml(args.base_config)
    val_records = json.loads(args.val_records_json.read_text(encoding="utf-8"))
    tuned_judge, stats = select_best_judge_thresholds(val_records, base_judge)
    out_path, digest = freeze_config(loc_cfg, tuned_judge, out_path=args.out, fitted_split="val", metadata=stats)
    print(f"Froze config to {out_path} (sha256={digest[:12]}...) val_stats={stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
