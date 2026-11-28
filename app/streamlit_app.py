"""VeriRay local verification app: Gate -> Reader -> Locator -> Eraser + Matched Controls -> Judge.

Runs 100% offline on this machine; uploads stay in memory (clear-session button removes state).
Run from the repository root:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path
import re
import sys

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veriray.brain_mri import BrainMRIError, BrainMRIReader, decode_brain_upload  # noqa: E402
from veriray.chest_hd import CHEST_HD_DISPLAY_NAMES  # noqa: E402
from veriray.config import configs_from_yaml  # noqa: E402
from veriray.contracts import SUPPORTED_FINDINGS, SYNTHETIC_COUNTERFACTUAL_LABEL, EvidenceRecord, Verdict  # noqa: E402
from veriray.controls import ControlConfig, generate_matched_controls  # noqa: E402
from veriray.eraser import EraseError, erase, render_labelled  # noqa: E402
from veriray.judge import JudgeConfig  # noqa: E402
from veriray.locator import LocatorConfig  # noqa: E402
from veriray.modality import guess_modality  # noqa: E402
from veriray.pipeline import BEHAVIOUR_NOTICE, SCORE_LABEL, overlay_heatmap, run_verification  # noqa: E402
from veriray.reader import Reader, ReaderError, summarize_reader_findings  # noqa: E402
from veriray.safety import answer_chat_question, sanitize_for_prompt  # noqa: E402
from veriray.unet_eraser import UNetEraser  # noqa: E402
from veriray.vlm import MedGemmaRunner, VLMConfig, VLMError, box_mask_iou, run_grounded_vlm_explainer  # noqa: E402

st.set_page_config(page_title="VeriRay — Verification Judge", layout="wide")

# ── Claude-inspired Warm Parchment + Botanical Sage Green Design System (100% Local CSS) ──
st.markdown(
    """
<style>
:root {
    --vr-bg: #FAF9F5;
    --vr-surface: #F2F0E8;
    --vr-card: #FFFFFF;
    --vr-border: #E5E2D9;
    --vr-border-strong: #D2CEC2;
    --vr-ink: #1F1E1D;
    --vr-muted: #66645E;
    --vr-sage: #5E8C6A;
    --vr-sage-deep: #3E6B4B;
    --vr-sage-soft: #E8F0EA;
    --vr-sage-border: #BFD3C4;
    --vr-amber-soft: #F6EFE2;
    --vr-amber-border: #D6B679;
    --vr-amber-ink: #6B4E21;
    --vr-clay-soft: #F6E9E6;
    --vr-clay-border: #CFA098;
    --vr-clay-ink: #6E2D24;
    --vr-slate-soft: #EFECE6;
    --vr-slate-border: #C4BFB4;
    --vr-slate-ink: #4A4740;
    --vr-serif: Georgia, 'Iowan Old Style', Charter, 'Palatino Linotype', 'Book Antiqua', serif;
    --vr-sans: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
    --vr-mono: 'Cascadia Code', Consolas, 'SFMono-Regular', Menlo, monospace;
}

.stApp {
    background-color: var(--vr-bg);
    color: var(--vr-ink);
    font-family: var(--vr-sans);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3.5rem;
    max-width: 1260px;
}

/* Editorial serif typography for headings & titles (Claude aesthetic) */
h1, h2, h3, .vr-serif-title {
    font-family: var(--vr-serif) !important;
    color: var(--vr-ink) !important;
    font-weight: 500 !important;
    letter-spacing: -0.02em !important;
}

h1 { font-size: 2.15rem !important; line-height: 1.2 !important; }
h2 { font-size: 1.45rem !important; margin-top: 1.4rem !important; }
h3 { font-size: 1.18rem !important; margin-top: 1rem !important; }

/* Metric cards styled like warm parchment editorial tiles */
div[data-testid="stMetric"] {
    background: var(--vr-card);
    border: 1px solid var(--vr-border);
    border-radius: 12px;
    padding: 0.9rem 1.05rem;
    box-shadow: 0 1px 2px rgba(31, 30, 29, 0.03);
}

div[data-testid="stMetricLabel"] {
    color: var(--vr-muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

div[data-testid="stMetricValue"] {
    font-family: var(--vr-serif) !important;
    color: var(--vr-ink) !important;
    font-size: 1.55rem !important;
}

/* Buttons in warm sage & stone */
div.stButton > button {
    border-radius: 9px;
    border: 1px solid var(--vr-border-strong);
    background-color: var(--vr-surface);
    color: var(--vr-ink);
    font-weight: 500;
    transition: all 0.15s ease;
}
div.stButton > button:hover {
    border-color: var(--vr-sage);
    background-color: var(--vr-sage-soft);
    color: var(--vr-sage-deep);
}

/* File uploader & expanders */
section[data-testid="stFileUploaderDropzone"] {
    background-color: var(--vr-card) !important;
    border: 1.5px dashed var(--vr-sage-border) !important;
    border-radius: 14px !important;
}

details[data-testid="stExpander"] {
    background-color: var(--vr-card);
    border: 1px solid var(--vr-border) !important;
    border-radius: 12px !important;
}

/* Custom top header & notice ribbon */
.vr-header-wrap {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.75rem;
    border-bottom: 1px solid var(--vr-border);
    padding-bottom: 0.85rem;
    margin-bottom: 0.9rem;
}

.vr-brand {
    font-family: var(--vr-serif);
    font-size: 2.05rem;
    font-weight: 500;
    letter-spacing: -0.025em;
    color: var(--vr-ink);
    margin: 0;
}

.vr-brand-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--vr-sage);
    margin-right: 10px;
    vertical-align: middle;
}

.vr-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
}

.vr-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.22rem 0.68rem;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 500;
    background: var(--vr-sage-soft);
    color: var(--vr-sage-deep);
    border: 1px solid var(--vr-sage-border);
}

.vr-pill-neutral {
    background: var(--vr-surface);
    color: var(--vr-muted);
    border: 1px solid var(--vr-border);
    font-family: var(--vr-mono);
    font-size: 0.73rem;
}

.vr-notice {
    background: var(--vr-surface);
    border: 1px solid var(--vr-border);
    border-left: 4px solid var(--vr-sage);
    border-radius: 10px;
    padding: 0.68rem 1rem;
    font-size: 0.88rem;
    color: var(--vr-ink);
    margin-bottom: 1.1rem;
}

/* Verdict cards */
.vr-verdict-card {
    border-radius: 14px;
    padding: 1.15rem 1.35rem;
    margin: 0.75rem 0 1.15rem 0;
    border: 1px solid var(--vr-border);
}
.vr-verdict-supported {
    background: var(--vr-sage-soft);
    border-color: var(--vr-sage);
    color: #24422D;
}
.vr-verdict-not-supported {
    background: var(--vr-amber-soft);
    border-color: var(--vr-amber-border);
    color: var(--vr-amber-ink);
}
.vr-verdict-shortcut {
    background: var(--vr-clay-soft);
    border-color: var(--vr-clay-border);
    color: var(--vr-clay-ink);
}
.vr-verdict-indeterminate {
    background: var(--vr-slate-soft);
    border-color: var(--vr-slate-border);
    color: var(--vr-slate-ink);
}
.vr-verdict-heading {
    font-family: var(--vr-serif);
    font-size: 1.35rem;
    font-weight: 600;
    margin-bottom: 0.25rem;
}
.vr-verdict-sub {
    font-size: 0.95rem;
    line-height: 1.5;
    margin-bottom: 0.5rem;
}
.vr-code-chip {
    display: inline-block;
    font-family: var(--vr-mono);
    font-size: 0.74rem;
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid rgba(31, 30, 29, 0.14);
    border-radius: 6px;
    padding: 0.12rem 0.48rem;
    margin-right: 0.35rem;
    margin-top: 0.2rem;
}

/* 7-Gate checklist grid */
.vr-gate-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
    gap: 0.55rem;
    margin: 0.6rem 0 1rem 0;
}
.vr-gate-item {
    background: var(--vr-card);
    border: 1px solid var(--vr-border);
    border-radius: 10px;
    padding: 0.55rem 0.8rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.83rem;
}
.vr-gate-pass {
    font-family: var(--vr-mono);
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--vr-sage-deep);
    background: var(--vr-sage-soft);
    border: 1px solid var(--vr-sage-border);
    padding: 0.1rem 0.45rem;
    border-radius: 999px;
}
.vr-gate-fail {
    font-family: var(--vr-mono);
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--vr-clay-ink);
    background: var(--vr-clay-soft);
    border: 1px solid var(--vr-clay-border);
    padding: 0.1rem 0.45rem;
    border-radius: 999px;
}

/* Editorial explanation card (Claude response feel) */
.vr-prose-card {
    background: var(--vr-card);
    border: 1px solid var(--vr-border);
    border-radius: 14px;
    padding: 1.15rem 1.35rem;
    margin: 0.5rem 0 1rem 0;
}
.vr-prose-line {
    font-family: var(--vr-serif);
    font-size: 1.02rem;
    line-height: 1.65;
    color: var(--vr-ink);
    margin-bottom: 0.6rem;
}
.vr-prose-line:last-child {
    margin-bottom: 0;
}
.vr-cite {
    font-family: var(--vr-mono);
    font-size: 0.72rem;
    background: var(--vr-sage-soft);
    color: var(--vr-sage-deep);
    border: 1px solid var(--vr-sage-border);
    border-radius: 5px;
    padding: 0.08rem 0.4rem;
    margin-left: 0.3rem;
    vertical-align: baseline;
}
</style>
""",
    unsafe_allow_html=True,
)

_GATE_LABELS = {
    "locator_valid": "Spatial locator quality",
    "mask_stability_passed": "Threshold & dilation stability",
    "controls_valid": "Matched controls (K>=8)",
    "edit_integrity_passed": "Outside-mask pixel & drift integrity",
    "fill_stability_passed": "Multi-fill stability (Telea/Blur/Mean)",
    "no_shortcut_flagged": "Thoracic ROI / shortcut audit",
    "target_distinguishable_from_controls": "Target drop > matched controls",
}


def _render_verdict_card(verdict: Verdict, subtitle: str, reason_codes: tuple[str, ...] | list[str]) -> None:
    cls_map = {
        Verdict.MODEL_SUPPORTED: "vr-verdict-supported",
        Verdict.NOT_SUPPORTED: "vr-verdict-not-supported",
        Verdict.SHORTCUT_RISK: "vr-verdict-shortcut",
        Verdict.INDETERMINATE: "vr-verdict-indeterminate",
    }
    chips = "".join(f'<span class="vr-code-chip">{html.escape(str(c))}</span>' for c in reason_codes)
    st.markdown(
        f"""
        <div class="vr-verdict-card {cls_map[verdict]}">
            <div class="vr-verdict-heading">{html.escape(verdict.value)}</div>
            <div class="vr-verdict-sub">{html.escape(subtitle)}</div>
            <div>{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_gate_grid(gate_results: dict[str, bool]) -> None:
    if not gate_results:
        return
    cards = []
    for key, passed in gate_results.items():
        label = html.escape(_GATE_LABELS.get(key, key.replace("_", " ").capitalize()))
        badge = '<span class="vr-gate-pass">PASS</span>' if passed else '<span class="vr-gate-fail">FAIL</span>'
        cards.append(f'<div class="vr-gate-item"><span>{label}</span>{badge}</div>')
    st.markdown(f'<div class="vr-gate-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _render_explanation_card(sentences: list[tuple[str, list[str] | tuple[str, ...]]]) -> None:
    lines = []
    for txt, eids in sentences:
        clean_txt = re.sub(r"\s*\[E\d+_[A-Z0-9_,\s]+\]", "", str(txt)).strip()
        cites = "".join(f'<span class="vr-cite">{html.escape(str(e))}</span>' for e in eids)
        lines.append(f'<div class="vr-prose-line">{html.escape(clean_txt)} {cites}</div>')
    st.markdown(f'<div class="vr-prose-card">{"".join(lines)}</div>', unsafe_allow_html=True)


# ── Header Bar ──────────────────────────────────────────────────────────────────────────────
hdr_left, hdr_right = st.columns([6, 1])
with hdr_left:
    st.markdown(
        """
        <div class="vr-header-wrap">
            <div>
                <p class="vr-brand"><span class="vr-brand-dot"></span>VeriRay</p>
                <span style="color:#66645E; font-size:0.92rem;">Counterfactual &amp; Matched-Control Verification Judge</span>
            </div>
            <div class="vr-pill-row">
                <span class="vr-pill">100% Offline</span>
                <span class="vr-pill">In-Memory Only</span>
                <span class="vr-pill">SHA-256 Verified</span>
                <span class="vr-pill-neutral">512px ConvNeXt-FPN</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hdr_right:
    if st.button("Clear session", width="stretch"):
        st.session_state.clear()
        st.rerun()

st.markdown(
    f'<div class="vr-notice"><strong>Non-Diagnostic Research Guardrail:</strong> {html.escape(BEHAVIOUR_NOTICE)}</div>',
    unsafe_allow_html=True,
)

MODE_CHEST = "Chest X-ray Verification Judge"
MODE_BRAIN = "Brain MRI (5-class Tumor & Stroke + FPN/Grad-CAM++)"
MODE_DEMO = "Precomputed 3-Minute Stage Cases"
MODE_EVAL = "Locked-Test Evidence Pack & Audits"

mode = st.radio(
    "Workspace",
    [MODE_CHEST, MODE_BRAIN, MODE_DEMO, MODE_EVAL],
    horizontal=True,
    label_visibility="collapsed",
)

# ═════════════════════════════════════════════════════════════════════════════════════════════
# MODE 1: BRAIN MRI (5-CLASS TUMOR & STROKE + FPN/GRAD-CAM++ + COUNTERFACTUAL)
# ═════════════════════════════════════════════════════════════════════════════════════════════
if mode == MODE_BRAIN:
    st.subheader("Brain MRI — Within-Source Model Scores, FPN/Grad-CAM++ & Counterfactual Check")
    st.caption(
        "Supports 2D slices (PNG/JPG/BMP) and 3D NIfTI volumes (.nii / .nii.gz) decoded strictly in memory "
        "with canonical skull-peel and radial-shell intensity standardization."
    )
    try:
        bm_reader = BrainMRIReader.load(ROOT / "models")
    except BrainMRIError as exc:
        st.error(f"INDETERMINATE — {exc.reason_code}: {exc}")
        st.stop()

    mri_file = st.file_uploader(
        "Upload a public, de-identified Brain MRI slice or 3D NIfTI volume (PNG/JPG/BMP/NII/NII.GZ)",
        type=["png", "jpg", "jpeg", "bmp", "nii", "gz"],
    )
    if mri_file is None:
        st.info("Upload a Brain MRI slice or 3D NIfTI volume to inspect model scores, candidate mask, and Telea counterfactual drop.")
        st.stop()

    try:
        arr = decode_brain_upload(mri_file.getvalue(), mri_file.name)
        out = bm_reader.predict_and_cam(arr)
    except BrainMRIError as exc:
        st.error(f"INDETERMINATE — {exc.reason_code}: {exc}")
        st.stop()
    except Exception:  # noqa: BLE001 - fail closed on corrupt image
        st.error("INDETERMINATE — CORRUPT_OR_UNREADABLE_IMAGE")
        st.stop()

    pred_cls = out["predicted_class"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Top model class (not a diagnosis)", pred_cls.replace("_", " "))
    m2.metric("Top class model score", f"{out['scores'][pred_cls]:.4f}")
    m3.metric("Supervised FPN lesion peak", f"{out.get('supervised_lesion_peak', 0.0):.4f}")

    st.dataframe(
        [{"class": k.replace("_", " "), "model_score": round(float(v), 4)} for k, v in sorted(out["scores"].items(), key=lambda kv: kv[1], reverse=True)],
        width="stretch",
        hide_index=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.image(out["display_image"].astype("uint8"), caption="Canonicalized Brain MRI (256px)", clamp=True)
    c2.image(overlay_heatmap(out["display_image"], out["heatmap"]), caption="FPN + Grad-CAM++ candidate region")
    c3.image((out["target_mask"] * 255).astype("uint8"), caption="Candidate binary mask")

    if out["target_mask"].any() and not out["target_mask"].all():
        cf = erase(out["display_image"], out["target_mask"], "telea", seed=0)
        cf_out = bm_reader.predict_and_cam(cf.edited_image)
        target_drop = float(out["scores"][pred_cls] - cf_out["scores"][pred_cls])
        st.subheader(f"Counterfactual Perturbation ({SYNTHETIC_COUNTERFACTUAL_LABEL})")
        cf_col1, cf_col2 = st.columns([1, 1])
        with cf_col1:
            st.image(render_labelled(cf.edited_image, scale=2), caption=f"Telea counterfactual — {SYNTHETIC_COUNTERFACTUAL_LABEL}")
        with cf_col2:
            k1, k2 = st.columns(2)
            k1.metric(f"Baseline `{pred_cls}` score", f"{out['scores'][pred_cls]:.4f}")
            k2.metric("Target edit score drop", f"{target_drop:+.4f}")
            ctrl_res = generate_matched_controls(out["target_mask"], ControlConfig(k=9, min_controls=3, seed=20261126))
            if ctrl_res.valid:
                c_drops = []
                for ci, cmask in enumerate(ctrl_res.masks):
                    ccf = erase(out["display_image"], cmask, "telea", seed=ci + 1)
                    ccf_out = bm_reader.predict_and_cam(ccf.edited_image)
                    c_drops.append(float(out["scores"][pred_cls] - ccf_out["scores"][pred_cls]))
                c_med = float(np.median(c_drops))
                emp_rank = 1 + int(np.sum(np.asarray(c_drops) >= target_drop))
                k3, k4 = st.columns(2)
                k3.metric(f"Control median drop (K={len(c_drops)})", f"{c_med:+.4f}")
                k4.metric("Empirical rank (not a p-value)", f"{emp_rank} / {len(c_drops) + 1}")
    st.caption(f"Checkpoint: `{out['checkpoint_id']}` · {BEHAVIOUR_NOTICE}")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════════════════════
# MODE 2: PRECOMPUTED 3-MINUTE STAGE DEMO CASES
# ═════════════════════════════════════════════════════════════════════════════════════════════
if mode == MODE_DEMO:
    demo_dir = ROOT / "artifacts" / "demo_cases"
    idx_path = demo_dir / "index.json"
    if not idx_path.is_file():
        st.info("Precomputed demo pack not generated yet. Run: `python scripts/generate_demo_pack.py`")
        st.stop()
    index = json.loads(idx_path.read_text(encoding="utf-8"))
    choice = st.selectbox("Select precomputed stage case", range(len(index)), format_func=lambda i: index[i]["title"])
    cdir = demo_dir / index[choice]["case_id"]
    bundle = json.loads((cdir / "bundle.json").read_text(encoding="utf-8"))
    st.caption(bundle["precomputed_notice"])
    v_enum = Verdict(bundle["verdict"])
    _render_verdict_card(v_enum, bundle["title"], bundle["reason_codes"])

    c1, c2, c3 = st.columns(3)
    c1.image(str(cdir / "original.png"), caption="Original view")
    c2.image(str(cdir / "overlay.png"), caption="Candidate region overlay")
    c3.image(str(cdir / "mask.png"), caption="Candidate mask")
    if bundle["has_counterfactual"]:
        st.subheader(f"Counterfactual ({bundle['synthetic_label']})")
        st.image(str(cdir / "counterfactual_labelled.png"), caption=bundle["synthetic_label"], width=440)

    st.subheader("Verified explanation")
    _render_explanation_card([(s["text"], s["evidence_ids"]) for s in bundle["verified_explanation"]])

    with st.expander("Structured EvidenceRecord JSON"):
        st.json(bundle["evidence_items"])

    demo_q = st.text_input("Ask the evidence-grounded chat about this precomputed case:")
    if demo_q:
        ev = EvidenceRecord(
            record_id=bundle["case_id"],
            finding_id="pneumothorax",
            verdict=v_enum,
            reason_codes=tuple(bundle["reason_codes"]),
            items=bundle["evidence_items"],
        )
        ans = answer_chat_question(demo_q, ev)
        (st.success if ans.allowed else st.warning)(ans.answer)
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════════════════════
# MODE 3: LOCKED-TEST EVIDENCE PACK & METHODOLOGICAL AUDITS
# ═════════════════════════════════════════════════════════════════════════════════════════════
if mode == MODE_EVAL:
    pack_path = ROOT / "artifacts" / "eval_test_locked" / "test_evidence_pack.json"
    if not pack_path.is_file():
        st.info("Locked test evidence pack not found. Run `python scripts/run_locked_test_and_exp5.py` first.")
        st.stop()
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    rc_cmp = pack.get("risk_coverage_comparison", {}).get("operating_point", {})
    fj = rc_cmp.get("full_judge", {})
    ro = rc_cmp.get("reader_at_same_coverage", {})
    loc_rep = pack.get("localization_reporting_half", {})
    exp5 = pack.get("experiment_5_claim_verifier", {})

    st.subheader("Locked-Test Headline Scorecard (Untouched During Training & Tuning)")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric(
        "Full Judge Precision",
        f"{fj.get('precision', 0.0):.2%}",
        f"{fj.get('accepted_count', 0)}/{fj.get('n', 0)} accepted ({fj.get('coverage', 0.0):.2%} cov)",
    )
    h2.metric(
        "Reader-Only @ Matched Cov",
        f"{ro.get('precision', 0.0):.2%}",
        f"{rc_cmp.get('precision_delta_judge_minus_reader', 0.0):+.2%} Judge delta",
    )
    h3.metric(
        "NIH Expert Pointing Game",
        f"{(loc_rep.get('pointing_game_accuracy') or 0.0):.2%}",
        f"{loc_rep.get('pointing_hits', 0)}/{loc_rep.get('eligible_annotated_cases', 0)} boxes · IoU {(loc_rep.get('mean_iou') or 0.0):.3f}",
    )
    h4.metric(
        "Claim Verifier Unsupported Rate",
        f"{(exp5.get('unsupported_rate_after_verifier') or 0.0):.1%}",
        f"down from {(exp5.get('unsupported_rate_before_verifier') or 0.0):.1%} raw VLM",
    )

    tab_judge, tab_chest, tab_brain, tab_safety = st.tabs(
        [
            "1. Full Judge & Ablations (n=445)",
            "2. 15→3 Finding Gate & External NIH (n=25,596)",
            "3. Brain MRI Within-Source Locked Test (n=1,378)",
            "4. Red-Team, Boundary & Latency Audits",
        ]
    )

    with tab_judge:
        st.caption(pack.get("note", "NIH ChestX-ray14 locked test evidence pack"))
        rm = pack.get("reader_metrics", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Reader AUROC", f"{rm.get('auroc', 0.0):.4f}")
        c2.metric("Reader AUPRC", f"{rm.get('auprc', 0.0):.4f}")
        c3.metric("Brier Score", f"{rm.get('brier', 0.0):.4f}")
        c4.metric("Sensitivity @ 0.5", f"{rm.get('sensitivity', 0.0):.2%}")
        c5.metric("Specificity @ 0.5", f"{rm.get('specificity', 0.0):.2%}")

        cal_path = ROOT / "models" / "platt_calibrator.json"
        if cal_path.is_file():
            cal_d = json.loads(cal_path.read_text(encoding="utf-8"))
            st.caption(
                f"Platt calibrator (`{cal_d.get('calibrator_id')}`): fitted on `{cal_d.get('fitted_split')}` "
                f"(n={cal_d.get('num_samples')}), coef={cal_d.get('coef', 0.0):.4f}, intercept={cal_d.get('intercept', 0.0):.4f}"
            )

        col_plot, col_verdicts = st.columns([3, 2])
        with col_plot:
            rc_img = ROOT / "artifacts" / "eval_test_locked" / "risk_coverage.png"
            if rc_img.is_file():
                st.image(str(rc_img), caption="Locked test selective risk-coverage curve")
        with col_verdicts:
            st.markdown("#### Verdict distribution (`n = 445`)")
            vc = pack.get("verdict_counts", {})
            st.dataframe(
                [{"verdict": k, "cases": v, "share": f"{(v / max(rm.get('n', 445), 1)):.2%}"} for k, v in vc.items()],
                width="stretch",
                hide_index=True,
            )
            fs = pack.get("fill_sensitivity", {})
            if fs:
                st.markdown("#### Multi-fill sensitivity")
                st.write(
                    f"Tested `{fs.get('n', 0)}` cases across `{', '.join(fs.get('methods', []))}` · "
                    f"Verdict flip rate: **{fs.get('verdict_flip_rate', 0.0):.2%}** (`{fs.get('flip_count', 0)}` flips, gated by Judge)."
                )

        st.markdown("#### Protocol Ablations A–F")
        abl = pack.get("ablations", {})
        if abl.get("configurations"):
            st.dataframe(abl["configurations"], width="stretch", hide_index=True)

        c_sex, c_age = st.columns(2)
        with c_sex:
            st.markdown("#### Subgroup audit — Sex")
            sex_tbl = pack.get("subgroups_sex", {}).get("groups", pack.get("subgroups_sex", {}))
            if isinstance(sex_tbl, dict):
                st.dataframe(
                    [{"group": k, **v} for k, v in sex_tbl.items() if isinstance(v, dict)],
                    width="stretch",
                    hide_index=True,
                )
        with c_age:
            st.markdown("#### Subgroup audit — Age band")
            age_tbl = pack.get("subgroups_age_band", {}).get("groups", pack.get("subgroups_age_band", {}))
            if isinstance(age_tbl, dict):
                st.dataframe(
                    [{"group": k, **v} for k, v in age_tbl.items() if isinstance(v, dict)],
                    width="stretch",
                    hide_index=True,
                )

    with tab_chest:
        chest_lt_path = ROOT / "artifacts" / "eval_test_locked" / "chest_hd_locked_test.json"
        nih_lt_path = ROOT / "artifacts" / "eval_test_locked" / "external_nih_test.json"
        if chest_lt_path.is_file():
            clt = json.loads(chest_lt_path.read_text(encoding="utf-8"))
            nih_lt = json.loads(nih_lt_path.read_text(encoding="utf-8")).get("metrics", {}) if nih_lt_path.is_file() else {}
            rule = clt.get("frozen_admission_rule", {})
            st.markdown(
                f"#### Frozen Verifiable Admission Gate (`n = {clt.get('test_scans_count', 0):,}` locked test scans · "
                f"Macro AUROC `{clt.get('macro_locked_test_auroc_15_findings', 0.0):.4f}`)"
            )
            st.caption(
                f"Admission rule (required on BOTH dev_val and locked_test): positives >= {rule.get('min_positives_per_split')}, "
                f"box/mask Dice >= {rule.get('min_pos_box_or_mask_dice')}, border-probe AUROC <= {rule.get('max_border_probe_auroc')}, "
                "and zero cross-source confound. Tuberculosis was dropped completely after border-only probes exposed a 97.5% framing confound."
            )
            rows_gate = []
            for fid, info in clt.get("findings", {}).items():
                lt = info.get("locked_test", {})
                nih_m = nih_lt.get(fid, {})
                admitted = info.get("passed_both_val_and_locked_test", False)
                reasons_fail = info.get("dev_val_fail_reasons", []) or info.get("locked_test_fail_reasons", [])
                rows_gate.append(
                    {
                        "status": "ADMITTED" if admitted else "EXCLUDED",
                        "finding": fid,
                        "supervision": lt.get("supervision_type"),
                        "test_pos": lt.get("test_positives"),
                        "within_source_test_auroc": lt.get("auroc"),
                        "box_or_mask_dice": lt.get("pos_box_or_mask_dice"),
                        "border_probe_auroc": lt.get("border_probe_test_auroc"),
                        "nih_external_auroc": nih_m.get("auroc"),
                        "nih_pointing_game": nih_m.get("pointing_game_hit_rate"),
                        "gate_note": "Passed dev_val + locked_test" if admitted else "; ".join(reasons_fail),
                    }
                )
            rows_gate.sort(key=lambda r: (0 if r["status"] == "ADMITTED" else 1, -(r["within_source_test_auroc"] or 0.0)))
            st.dataframe(rows_gate, width="stretch", hide_index=True)

    with tab_brain:
        brain_lt_path = ROOT / "artifacts" / "eval_test_locked" / "brain_mri_locked_test.json"
        brain_aud_path = ROOT / "data" / "manifests" / "brain_mri_split_audit.json"
        if brain_lt_path.is_file():
            blt = json.loads(brain_lt_path.read_text(encoding="utf-8"))
            wi = blt.get("headline_within_isles_stroke_vs_normal", {})
            wb = blt.get("within_brats_glioma_vs_normal", {})
            w2 = blt.get("within_2d_4class", {})
            st.markdown(f"#### Within-Source Brain MRI Locked Test (`n = {blt.get('test_slices_count', 0):,}` slices)")
            st.caption(blt.get("note", ""))
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Within-ISLES Stroke AUROC", f"{wi.get('auroc_p_stroke', 0.0):.4f}", f"Mask Dice {wi.get('stroke_mask_dice', 0.0):.4f}")
            b2.metric("Within-BraTS Glioma AUROC", f"{wb.get('auroc_p_glioma', 0.0):.4f}", f"Mask Dice {wb.get('glioma_mask_dice', 0.0):.4f}")
            b3.metric("Within-2D 4-Class Accuracy", f"{w2.get('accuracy_4cls', 0.0):.2%}", f"n={w2.get('n_slices', 0)} slices")
            b4.metric("Overall Lesion Pixel Dice", f"{blt.get('overall_lesion_segmentation_dice_locked_test', 0.0):.4f}")

            st.markdown("#### Figshare 2D Exact Pixel-Mask Tumor Dice (Locked Test)")
            t_dices = w2.get("tumor_mask_dice_by_class", {})
            if t_dices:
                st.dataframe([{"tumor_class": k, "pixel_mask_dice": v} for k, v in t_dices.items()], width="stretch", hide_index=True)

            if brain_aud_path.is_file():
                baud = json.loads(brain_aud_path.read_text(encoding="utf-8"))
                p8 = baud.get("border_probe_after_canonical_8pct_outer_frame", {})
                p14 = baud.get("parenchymal_rim_probe_after_canonical_14pct", {})
                st.caption(
                    f"De-confounding audit (`{baud.get('dropped_synthetic_tr_aug_copies', 0)}` synthetic `Tr-aug` copies purged): "
                    f"3-way outer-frame border probe dropped from `{baud.get('border_probe_before_canonical_8pct', {}).get('val_acc', 0.0):.2%}` "
                    f"to `{p8.get('val_acc', 0.0):.2%}` (chance `33.33%`). Cortical 14% strip retains `{p14.get('val_acc', 0.0):.2%}` "
                    "MRI sequence contrast (DWI vs T1ce), which is why within-source metrics are reported as headline."
                )

    with tab_safety:
        rt_p = ROOT / "artifacts" / "redteam_report.json"
        art_p = ROOT / "artifacts" / "audit_redteam_report.json"
        bnd_p = ROOT / "artifacts" / "boundary_report.json"
        lat_p = ROOT / "artifacts" / "latency_and_api_report.json"
        s1, s2, s3, s4 = st.columns(4)
        if rt_p.is_file():
            rtd = json.loads(rt_p.read_text(encoding="utf-8"))
            s1.metric("Primary Red-Team Suite", f"{rtd.get('attack_prompts_passed', 0)}/{rtd.get('attack_prompts_total', 0)}", f"{rtd.get('attack_pass_rate', 0.0):.1%} blocked")
        if art_p.is_file():
            artd = json.loads(art_p.read_text(encoding="utf-8"))
            s2.metric("Independent Audit Red-Team (E2E)", f"{artd.get('blocked_e2e', 0)}/{artd.get('total_attacks', 0)}", f"Layer-1 alone: {artd.get('pass_rate_layer1', 0.0):.1%}")
        if bnd_p.is_file():
            bndd = json.loads(bnd_p.read_text(encoding="utf-8"))
            s3.metric("Boundary Fail-Closed Suite", f"{bndd.get('total_boundary_cases', 0)}/{bndd.get('total_boundary_cases', 0)}", f"{bndd.get('fail_closed_rate', 0.0):.1%} fail-closed")
        if lat_p.is_file():
            latd = json.loads(lat_p.read_text(encoding="utf-8"))
            e2e_lat = latd.get("stage_latencies", {}).get("end_to_end", {})
            s4.metric("Median End-to-End Latency", f"{e2e_lat.get('p50_ms', 0.0):.0f} ms", f"p95: {e2e_lat.get('p95_ms', 0.0):.0f} ms")

        if art_p.is_file():
            artd = json.loads(art_p.read_text(encoding="utf-8"))
            st.markdown("#### Independent Audit Red-Team Breakdown (130 Adversarial Attacks)")
            st.dataframe(
                [{"category": k, **v} for k, v in artd.get("per_category", {}).items()],
                width="stretch",
                hide_index=True,
            )

    st.stop()

# ═════════════════════════════════════════════════════════════════════════════════════════════
# MODE 4 (DEFAULT): LIVE CHEST X-RAY VERIFICATION JUDGE
# ═════════════════════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading verified local Reader from models/ ...")
def _load_reader(finding_id: str = "pneumothorax"):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return Reader.load_for_finding(ROOT / "models", finding_id, device=device), None
    except ReaderError as exc:
        return None, exc


ctrl_col1, ctrl_col2 = st.columns([2, 3])
with ctrl_col1:
    _avail_findings = list(SUPPORTED_FINDINGS)
    target_finding = st.selectbox(
        "Verified thoracic finding (`SUPPORTED_FINDINGS`)",
        _avail_findings,
        index=0,
        format_func=lambda f: f"{CHEST_HD_DISPLAY_NAMES.get(f, f.capitalize())} ({f})",
        help="Restricted to the 3 findings that passed positives>=100, Dice>=0.50, and border-probe<=0.80 on both dev_val and locked_test.",
    )
with ctrl_col2:
    st.write("")
    use_medgemma = st.checkbox(
        "Enable MedGemma 4B Claimant & Grounded Explainer (4-bit local GPU, ~3.4 GB VRAM)",
        value=False,
    )

reader, load_err = _load_reader(target_finding)
if load_err is not None:
    st.error(f"INDETERMINATE — {load_err.reason_code}: {load_err.message}")
    st.stop()

uploads = st.file_uploader(
    "Upload public, de-identified chest X-rays (PNG/JPEG/BMP/TIFF — single scan or batch queue)",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)
if not uploads:
    st.info("Select a verified thoracic finding above and drop one or more de-identified chest X-rays to run the 7-gate Verification Judge.")
    st.stop()

frozen_cfg_path = ROOT / "configs" / "frozen.yaml"
if frozen_cfg_path.is_file():
    loc_cfg, judge_cfg, _ = configs_from_yaml(frozen_cfg_path)
else:
    loc_cfg, judge_cfg = LocatorConfig(), JudgeConfig()

queue_results = []
status_bar = st.progress(0, text="Running Verification Judge...")
for idx, upl in enumerate(uploads):
    status_bar.progress((idx + 1) / len(uploads), text=f"Verifying {idx + 1}/{len(uploads)}: {upl.name}")

    # ── Modality mismatch guard (bounded against image decompression bombs) ──
    raw_upl_bytes = upl.getvalue()
    if len(raw_upl_bytes) <= 50 * 1024 * 1024:
        try:
            import io as _io
            from PIL import Image as _PILImage

            with _PILImage.open(_io.BytesIO(raw_upl_bytes)) as _im:
                _w, _h = _im.size
                if 0 < _w <= 8192 and 0 < _h <= 8192 and (_w * _h) <= 8192 * 8192:
                    _im.load()
                    _raw = np.asarray(_im.convert("L"), dtype=np.float32)
                else:
                    _raw = None
            if _raw is not None:
                _guess = guess_modality(_raw)
                if _guess.is_brain_mri_suspected and _guess.confidence >= 0.60:
                    st.warning(
                        f"**Modality mismatch detected for `{upl.name}`** — this scan resembles a **Brain MRI**, not a chest X-ray.  \n"
                        f"Switch Workspace above to **{MODE_BRAIN}** and re-upload "
                        f"*(heuristic confidence: {_guess.confidence:.0%}; not a clinical assessment)*."
                    )
                    with st.expander("Modality heuristic signals"):
                        st.json({k: round(v, 4) for k, v in _guess.signals.items()})
                        st.caption(_guess.reason)
                    continue
        except Exception:  # noqa: BLE001
            pass

    r_item = run_verification(
        raw_upl_bytes,
        filename_hint=upl.name,
        reader=reader,
        models_dir=ROOT / "models",
        manifest_dir=None,
        locator_cfg=loc_cfg,
        judge_cfg=judge_cfg,
        finding_id=target_finding,
    )
    queue_results.append((upl.name, r_item))
status_bar.empty()

if not queue_results:
    st.info(f"All uploaded file(s) were flagged as Brain MRI scans. Switch Workspace above to **{MODE_BRAIN}**.")
    st.stop()

if len(queue_results) > 1:
    st.subheader(f"Batch Queue Summary ({len(queue_results)} scans)")
    st.dataframe(
        [
            {
                "case": name,
                "verdict": item.verdict.value,
                "score_raw": round(item.reader.score_raw, 4) if item.reader else None,
                "reasons": ", ".join(item.reason_codes),
                "total_ms": round(sum(item.latencies_ms.values()), 1),
            }
            for name, item in queue_results
        ],
        width="stretch",
        hide_index=True,
    )
    selected_idx = st.selectbox(
        "Inspect scan",
        range(len(queue_results)),
        format_func=lambda i: f"{queue_results[i][0]} — {queue_results[i][1].verdict.value}",
    )
else:
    selected_idx = 0

selected_name, res = queue_results[selected_idx]

if use_medgemma and res.display_image is not None:
    with st.spinner("Running MedGemma 4B (4-bit local GPU)..."):
        try:
            vlm = MedGemmaRunner.load(VLMConfig(model_dir=ROOT / "models" / "medgemma", quantization="4bit"))
            vlm_claims = vlm.claim_on_image(res.display_image)
            vlm_expl_rep = (
                run_grounded_vlm_explainer(res.judge_outcome.evidence_record, vlm.generate_text)
                if res.judge_outcome is not None
                else None
            )
            vlm_peak = vlm.peak_vram_mb
            vlm.unload()
            st.subheader("MedGemma 4B Claimant (Local 4-Bit GPU)")
            for c in vlm_claims:
                iou_txt = ""
                if c.box_xyxy is not None and res.locator is not None:
                    iou_txt = f" · IoU with Locator mask: {box_mask_iou(res.locator.target_mask, c.box_xyxy):.3f}"
                c_norm = c.finding.strip().lower().replace(" ", "_").replace("/", "_")
                if c_norm == "pleural_effusion":
                    c_norm = "effusion"
                tag = "verified target finding" if (c.is_verifiable_finding and c_norm == target_finding) else "mentioned by the model, not verified"
                s_finding = sanitize_for_prompt(c.finding)
                s_status = sanitize_for_prompt(c.status)
                s_region = sanitize_for_prompt(c.region_description)
                st.write(f"- **{s_finding}** (`{s_status}`, {s_region}) — *{tag}*{iou_txt}")
            if vlm_expl_rep is not None:
                st.caption(
                    f"MedGemma grounded explainer: {len(vlm_expl_rep.verified_sentences)} verified sentences, "
                    f"{vlm_expl_rep.dropped_count} dropped by Claim Verifier, peak VRAM {vlm_peak:.0f} MB"
                )
                _render_explanation_card([(s.text, s.evidence_ids) for s in vlm_expl_rep.verified_sentences])
        except VLMError as exc:
            st.warning(f"MedGemma unavailable ({exc.reason_code}): {exc}")

user_text = res.output.user_facing_text if res.output is not None else "The test could not produce a reliable result."
_render_verdict_card(res.verdict, user_text, res.reason_codes)

if res.judge_outcome is not None:
    st.markdown("#### 7-Gate Verification Protocol")
    _render_gate_grid(dict(res.judge_outcome.contract.gate_results))

if res.reader is not None and res.locator is not None and res.display_image is not None:
    target_disp = CHEST_HD_DISPLAY_NAMES.get(target_finding, target_finding.replace("_", " ").capitalize())
    disp_px = int(res.display_image.shape[0])
    score_lbl = SCORE_LABEL if target_finding == "pneumothorax" else f"Model score (uncalibrated {target_disp} output)"

    c1, c2, c3 = st.columns(3)
    c1.image(res.display_image.astype("uint8"), caption=f"Native {disp_px}px model-space radiograph", clamp=True)
    c2.image(overlay_heatmap(res.display_image, res.locator.heatmap), caption="FPN + Grad-CAM++ candidate region")
    c3.image((res.locator.target_mask * 255).astype("uint8"), caption="Binary candidate mask")

    st.subheader("Reader Scores & Gate Provenance")
    m1, m2, m3 = st.columns(3)
    m1.metric(score_lbl, f"{res.reader.score_raw:.4f}")
    if res.reader.score_calibrated_optional is not None:
        m2.metric("Platt-calibrated score (val-fitted)", f"{res.reader.score_calibrated_optional:.4f}")
    else:
        m2.metric("Calibration state", "Uncalibrated raw score")
    m3.metric("Input Gate", res.gate.status.value if res.gate is not None else "N/A")

    target_thresh = judge_cfg.min_reader_score if judge_cfg.min_reader_score > 0 else 0.5
    screening = summarize_reader_findings(res.reader, target_threshold=target_thresh, other_threshold=0.5)
    if screening.all_scores_low:
        st.info(
            "Reader screening summary: No monitored thoracic finding exceeded the model-score threshold. "
            "This describes uncalibrated model scores only, not a clinical assessment."
        )
    elif not screening.target_flagged and screening.flagged_other_findings:
        flagged_txt = ", ".join(f"{lbl} ({val:.3f})" for lbl, val in screening.flagged_other_findings)
        st.warning(
            f"{target_disp} score is below threshold, while other unverified Reader channel(s) scored >= 0.50: {flagged_txt}"
        )

    with st.expander("Monitored uncoupled Reader channels & checkpoint provenance"):
        rows_unrel = [{"channel": target_disp + " (target)", "score": round(res.reader.score_raw, 4)}]
        for lbl, val in sorted(res.reader.unrelated_scores.items(), key=lambda kv: kv[1], reverse=True):
            rows_unrel.append({"channel": lbl, "score": round(float(val), 4)})
        st.dataframe(rows_unrel, width="stretch", hide_index=True)
        st.caption(
            f"Checkpoint: `{res.reader.checkpoint_id}` · Preprocessing: `{res.reader.preprocessing_id}` · "
            f"Threshold ID: `{res.reader.threshold_id}`"
        )

if res.judge_outcome is not None and res.judge_outcome.primary_target_edit is not None and res.display_image is not None:
    edit = res.judge_outcome.primary_target_edit
    st.subheader(f"Counterfactual Perturbation ({SYNTHETIC_COUNTERFACTUAL_LABEL})")
    view_choice = st.radio(
        "Comparison layout",
        ["Side-by-side", "Interactive toggle"],
        horizontal=True,
        label_visibility="collapsed",
    )
    labelled_edit = render_labelled(edit.edited_image, scale=1 if res.display_image.shape[0] >= 512 else 2)
    if view_choice == "Side-by-side":
        col_a, col_b = st.columns(2)
        col_a.image(res.display_image.astype("uint8"), caption=f"Original ({int(res.display_image.shape[0])}px)", clamp=True)
        col_b.image(labelled_edit, caption=f"{edit.fill_method} — {SYNTHETIC_COUNTERFACTUAL_LABEL}")
    else:
        show_cf = st.toggle("Show synthetic counterfactual", value=True)
        if show_cf:
            st.image(labelled_edit, caption=f"{edit.fill_method} — {SYNTHETIC_COUNTERFACTUAL_LABEL}", width=520)
        else:
            st.image(res.display_image.astype("uint8"), caption=f"Original ({int(res.display_image.shape[0])}px)", clamp=True, width=520)

    if (ROOT / "models" / "UNET_CHECKPOINT.json").is_file() and res.locator is not None:
        with st.expander("Optional Gated-Convolution U-Net counterfactual comparison"):
            try:
                unet = UNetEraser.load(ROOT / "models")
                unet_edit = unet.erase(res.display_image, res.locator.target_mask, seed=0)
                unet_rc, _, _ = reader.score(unet_edit.edited_image)
                unet_drop = float(res.reader.score_raw - unet_rc.score_raw) if res.reader is not None else 0.0
                st.image(render_labelled(unet_edit.edited_image, scale=1), caption=f"gated_unet — {SYNTHETIC_COUNTERFACTUAL_LABEL}", width=440)
                st.caption(
                    f"Gated U-Net score drop: {unet_drop:.4f} · boundary discontinuity: {unet_edit.integrity_report.boundary_discontinuity:.4f}"
                )
            except EraseError as exc:
                st.caption(f"Gated U-Net unavailable ({exc.reason_code})")

    jc = res.judge_outcome.contract
    te = dict(jc.target_effects)
    ce = dict(jc.control_effects)
    if "score_drops" in ce:
        drops = ce["score_drops"]
        st.subheader("Target Region vs. Anatomically Matched Controls")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric(f"Target Drop ({te.get('primary_fill_method')})", f"{te.get('primary_score_drop', 0.0):+.4f}")
        e2.metric(f"Control Median (K={ce.get('num_controls')})", f"{ce.get('median_score_drop', 0.0):+.4f}")
        e3.metric("Margin Over Controls", f"{te.get('margin_over_control_median', 0.0):+.4f}")
        e4.metric("Empirical Rank (not p-value)", f"{te.get('empirical_rank')} / {int(ce.get('num_controls', 0)) + 1}")
        st.bar_chart(
            {
                "Target edit drop": [te.get("primary_score_drop", 0.0)],
                "Matched control median": [ce.get("median_score_drop", 0.0)],
                "Matched control max": [float(np.max(drops)) if drops else 0.0],
            },
            color="#5E8C6A",
        )

if res.explanation is not None:
    st.subheader("Rule-Verified Grounded Explanation")
    _render_explanation_card([(s.text, s.evidence_ids) for s in res.explanation.sentences])
    if res.explanation.unverified_other_claims:
        st.caption("Mentioned by claimant model, not verified: " + "; ".join(res.explanation.unverified_other_claims))

st.subheader("Evidence-Grounded Chat (5-Layer Guardrailed)")
chat_q = st.text_input("Ask about this scan's verdict, score drop, matched controls, or VeriRay definitions:")
if chat_q:
    ev_rec = res.judge_outcome.evidence_record if res.judge_outcome is not None else None
    chat_resp = answer_chat_question(chat_q, ev_rec)
    if chat_resp.allowed:
        st.success(chat_resp.answer)
    else:
        st.warning(chat_resp.answer)

with st.expander("Audit Trail — Structured EvidenceRecord & RunManifest"):
    if res.judge_outcome is not None:
        st.json(dict(res.judge_outcome.evidence_record.items))
    if res.manifest is not None:
        st.caption(
            f"Run ID: `{res.manifest.run_id}` · offline_only={res.manifest.offline_only} · "
            f"latencies (ms): { {k: round(v, 1) for k, v in res.latencies_ms.items()} }"
        )
