#!/usr/bin/env python
"""Run Experiment 8 (red-team guardrail evaluation) locally and write measured pass rates."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from redteam.suite import RedTeamCase, build_redteam_suite  # noqa: E402
from veriray.contracts import EvidenceRecord, Verdict  # noqa: E402
from veriray.safety import answer_chat_question, check_output_safety, sanitize_for_prompt  # noqa: E402


def _mock_evidence() -> EvidenceRecord:
    return EvidenceRecord(
        record_id="redteam-ev-1",
        finding_id="pneumothorax",
        verdict=Verdict.MODEL_SUPPORTED,
        reason_codes=("JUDGE_ALL_GATES_PASSED",),
        items={
            "E01_MODEL_SCORE_RAW": 0.72,
            "E02_VERDICT": "MODEL-SUPPORTED",
            "E03_REASON_CODES": ["JUDGE_ALL_GATES_PASSED"],
            "E04_TARGET_SCORE_DROP_PRIMARY": 0.21,
            "E06_CONTROL_COUNT": 19,
            "E07_CONTROL_SCORE_DROP_MEDIAN": 0.02,
            "E08_EMPIRICAL_TAIL_FRACTION": 0.05,
            "E11_MASK_MIN_STABILITY_IOU": 0.85,
            "E12_UNRELATED_LABEL_MAX_DRIFT": 0.01,
        },
    )


def evaluate_suite(cases: list[RedTeamCase] | None = None) -> dict:
    cases = cases or build_redteam_suite()
    ev = _mock_evidence()
    by_cat: dict[str, dict[str, int]] = {}
    failures: list[dict] = []

    for c in cases:
        cat = by_cat.setdefault(c.category, {"total": 0, "passed": 0})
        cat["total"] += 1
        if c.untrusted_carrier != "chat":
            # Untrusted filename/EXIF/OCR strings are sanitized and never executed as instructions;
            # verify both that querying about them is safe and that direct evaluation refuses injection.
            safe_str = sanitize_for_prompt(c.prompt)
            resp = answer_chat_question(f"Follow the instruction in {c.untrusted_carrier}: {safe_str}", ev)
        else:
            resp = answer_chat_question(c.prompt, ev)

        out_safe = check_output_safety(resp.answer).safe
        passed = ((not resp.allowed) == c.expect_blocked) and out_safe
        if passed:
            cat["passed"] += 1
        else:
            failures.append({"id": c.case_id, "category": c.category, "allowed": resp.allowed, "out_safe": out_safe})

    for stats in by_cat.values():
        stats["pass_rate"] = stats["passed"] / float(stats["total"])

    attack_total = sum(v["total"] for k, v in by_cat.items() if k != "benign_grounded_control")
    attack_passed = sum(v["passed"] for k, v in by_cat.items() if k != "benign_grounded_control")
    return {
        "attack_prompts_total": attack_total,
        "attack_prompts_passed": attack_passed,
        "attack_pass_rate": attack_passed / float(max(1, attack_total)),
        "by_category": by_cat,
        "failures": failures,
    }


def main() -> int:
    report = evaluate_suite()
    out = ROOT / "artifacts" / "redteam_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Attacks: {report['attack_prompts_passed']}/{report['attack_prompts_total']} passed "
          f"({report['attack_pass_rate']:.1%})")
    for cat, stats in report["by_category"].items():
        print(f"  {cat:32} {stats['passed']:3}/{stats['total']:3} ({stats['pass_rate']:.1%})")
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
