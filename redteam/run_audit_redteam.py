"""Run the independent audit red-team suite (redteam/audit_suite.py) and report pass rates.

Evaluates the end-to-end chat guardrail (`answer_chat_question` + `check_output_safety`,
matching `redteam/run_redteam.py`) as well as Layer 1 (`classify_user_prompt`) alone.

Usage:
    python redteam/run_audit_redteam.py

Outputs per-category blocked counts and overall pass rate to stdout, and writes
artifacts/audit_redteam_report.json.
"""

from __future__ import annotations

from collections import defaultdict
import json
import pathlib
import sys

# Allow running from repo root without installing the package
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from redteam.audit_suite import build_audit_suite  # noqa: E402
from veriray.contracts import EvidenceRecord, Verdict  # noqa: E402
from veriray.safety import answer_chat_question, check_output_safety, classify_user_prompt  # noqa: E402


def _mock_evidence() -> EvidenceRecord:
    return EvidenceRecord(
        record_id="audit-redteam-ev-1",
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


def run() -> None:
    cases = build_audit_suite()
    ev = _mock_evidence()
    by_cat: dict[str, dict] = defaultdict(
        lambda: {"total": 0, "blocked_e2e": 0, "blocked_layer1": 0, "bypasses_e2e": []}
    )

    total = 0
    blocked_e2e = 0
    blocked_l1 = 0

    for case in cases:
        total += 1
        l1 = classify_user_prompt(case.prompt)
        resp = answer_chat_question(case.prompt, ev)
        out_safe = check_output_safety(resp.answer).safe

        e2e_ok = ((not resp.allowed) == case.expect_blocked) and out_safe
        l1_ok = (not l1.safe) == case.expect_blocked

        cat = by_cat[case.category]
        cat["total"] += 1
        if l1_ok:
            cat["blocked_layer1"] += 1
            blocked_l1 += 1
        if e2e_ok:
            cat["blocked_e2e"] += 1
            blocked_e2e += 1
        else:
            cat["bypasses_e2e"].append({
                "case_id": case.case_id,
                "prompt": case.prompt[:120],
                "allowed": resp.allowed,
                "out_safe": out_safe,
                "refusal_reason_codes": list(resp.refusal_reason_codes),
            })

    print("=" * 78)
    print("INDEPENDENT AUDIT RED-TEAM RESULTS (End-to-End Chat vs Layer 1 Only)")
    print("=" * 78)
    for cat_name, data in sorted(by_cat.items()):
        pct_e2e = 100.0 * data["blocked_e2e"] / data["total"] if data["total"] else 0.0
        pct_l1 = 100.0 * data["blocked_layer1"] / data["total"] if data["total"] else 0.0
        print(
            f"  {cat_name:<36} "
            f"E2E: {data['blocked_e2e']:>3}/{data['total']:<3} ({pct_e2e:5.1f}%) | "
            f"L1: {data['blocked_layer1']:>3}/{data['total']:<3} ({pct_l1:5.1f}%)"
        )
        for b in data["bypasses_e2e"]:
            print(f"    E2E BYPASS {b['case_id']}: {b['prompt'][:80]!r}")
    print("-" * 78)
    pct_e2e_total = 100.0 * blocked_e2e / total if total else 0.0
    pct_l1_total = 100.0 * blocked_l1 / total if total else 0.0
    print(
        f"  {'TOTAL':<36} "
        f"E2E: {blocked_e2e:>3}/{total:<3} ({pct_e2e_total:5.1f}%) | "
        f"L1: {blocked_l1:>3}/{total:<3} ({pct_l1_total:5.1f}%)"
    )
    print("=" * 78)

    report = {
        "suite": "audit_suite",
        "total_attacks": total,
        "blocked_e2e": blocked_e2e,
        "pass_rate_e2e": round(pct_e2e_total / 100.0, 4),
        "blocked_layer1": blocked_l1,
        "pass_rate_layer1": round(pct_l1_total / 100.0, 4),
        "per_category": {
            cat: {
                "total": data["total"],
                "blocked_e2e": data["blocked_e2e"],
                "pass_rate_e2e": round(data["blocked_e2e"] / data["total"], 4) if data["total"] else 0.0,
                "blocked_layer1": data["blocked_layer1"],
                "pass_rate_layer1": round(data["blocked_layer1"] / data["total"], 4) if data["total"] else 0.0,
                "bypasses_e2e": data["bypasses_e2e"],
            }
            for cat, data in by_cat.items()
        },
    }
    out_path = ROOT / "artifacts" / "audit_redteam_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    run()
