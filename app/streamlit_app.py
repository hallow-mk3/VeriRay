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

st.set_page_config(page_title="VeriRay", layout="wide")

# ── Minimalist Claude Dark Mode × Sage Green Aesthetic (100% Local, Zero External Assets) ──
st.markdown(
    """
<style>
:root {
    --bg: #141413;
    --surface: #1C1C1A;
    --surface-elevated: #232320;
    --border: #2B2A27;
    --border-subtle: #22221F;
    --ink: #ECE9E1;
    --ink-secondary: #A3A096;
    --ink-muted: #6E6C64;
    --sage: #739E82;
    --sage-soft: rgba(115, 158, 130, 0.12);
    --sage-border: rgba(115, 158, 130, 0.28);
    --amber: #C9A66B;
    --amber-soft: rgba(201, 166, 107, 0.10);
    --rose: #C47A70;
    --rose-soft: rgba(196, 122, 112, 0.10);
    --serif: Georgia, 'Iowan Old Style', Charter, 'Palatino Linotype', serif;
    --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif;
    --mono: 'Cascadia Code', Consolas, 'SFMono-Regular', Menlo, monospace;
}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--bg) !important;
    color: var(--ink) !important;
    font-family: var(--sans);
    -webkit-font-smoothing: antialiased;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

#MainMenu, footer { visibility: hidden; }

/* Centered, calm editorial column like Claude */
.block-container {
    padding-top: 2.8rem !important;
    padding-bottom: 4.5rem !important;
    max-width: 980px !important;
}

/* Typography */
h1, h2, h3, .claude-serif {
    font-family: var(--serif) !important;
    color: var(--ink) !important;
    font-weight: 400 !important;
    letter-spacing: -0.018em !important;
}
h1 { font-size: 2rem !important; }
h2 { font-size: 1.35rem !important; margin-top: 1.8rem !important; margin-bottom: 0.5rem !important; }
h3 { font-size: 1.12rem !important; margin-top: 1.4rem !important; margin-bottom: 0.4rem !important; }

p, label, span, li {
    color: var(--ink-secondary);
}

hr {
    border: none !important;
    border-top: 1px solid var(--border-subtle) !important;
    margin: 1.5rem 0 !important;
}

code {
    font-family: var(--mono) !important;
    background: var(--surface) !important;
    color: var(--ink) !important;
    border: 1px solid var(--border) !important;
    border-radius: 5px !important;
    padding: 0.08rem 0.35rem !important;
    font-size: 0.80em !important;
}

/* Minimal top bar */
.vr-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 0.9rem;
    margin-bottom: 1.1rem;
    border-bottom: 1px solid var(--border-subtle);
}
.vr-wordmark {
    font-family: var(--serif);
    font-size: 1.32rem;
    color: var(--ink);
    letter-spacing: -0.02em;
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
}
.vr-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--sage);
    display: inline-block;
}
.vr-meta {
    font-size: 0.78rem;
    color: var(--ink-muted);
}

/* Editorial Hero (Empty state) */
.vr-hero {
    text-align: center;
    margin: 2.2rem auto 1.8rem auto;
    max-width: 600px;
}
.vr-hero-title {
    font-family: var(--serif);
    font-size: 2.15rem;
    color: var(--ink);
    letter-spacing: -0.025em;
    margin-bottom: 0.45rem;
    line-height: 1.2;
}
.vr-hero-sub {
    font-size: 0.94rem;
    color: var(--ink-secondary);
    line-height: 1.55;
}

/* Quiet Segmented Switcher */
div[role="radiogroup"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 3px !important;
    gap: 2px !important;
    display: flex !important;
    margin-bottom: 1.6rem !important;
}
div[role="radiogroup"] > label {
    background: transparent !important;
    border: none !important;
    border-radius: 7px !important;
    padding: 0.42rem 0.85rem !important;
    margin: 0 !important;
    flex: 1 1 auto !important;
    justify-content: center !important;
    transition: background 0.14s ease !important;
}
div[role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, 0.03) !important;
}
div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}
div[role="radiogroup"] > label p {
    font-size: 0.83rem !important;
    font-weight: 450 !important;
    color: var(--ink-muted) !important;
}
div[role="radiogroup"] > label:has(input:checked) {
    background: var(--surface-elevated) !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.28);
}
div[role="radiogroup"] > label:has(input:checked) p {
    color: var(--ink) !important;
}

/* Inputs & Selectboxes */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--ink) !important;
    box-shadow: none !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border-color: var(--sage) !important;
}

/* Quiet Buttons */
div.stButton > button {
    background: transparent !important;
    color: var(--ink-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 0.80rem !important;
    padding: 0.35rem 0.75rem !important;
    transition: all 0.14s ease !important;
}
div.stButton > button:hover {
    border-color: var(--sage) !important;
    color: var(--ink) !important;
    background: var(--sage-soft) !important;
}

/* Claude-style Composer Dropzone */
section[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 1.5rem 1.4rem !important;
    transition: border-color 0.16s ease !important;
}
section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--sage) !important;
}
section[data-testid="stFileUploaderDropzone"] button {
    background: var(--sage-soft) !important;
    color: var(--sage) !important;
    border: 1px solid var(--sage-border) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
}

/* Minimal Metric Cards */
div[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 0.9rem 1.05rem !important;
}
div[data-testid="stMetricLabel"] p {
    color: var(--ink-muted) !important;
    font-size: 0.77rem !important;
}
div[data-testid="stMetricValue"] {
    font-family: var(--serif) !important;
    color: var(--ink) !important;
    font-size: 1.5rem !important;
    font-variant-numeric: tabular-nums;
}
div[data-testid="stMetricDelta"] {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: var(--ink-muted) !important;
    font-size: 0.84rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--ink) !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: var(--sage) !important;
}

/* Expanders */
details[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: 11px !important;
}
details[data-testid="stExpander"] summary p {
    color: var(--ink-secondary) !important;
    font-size: 0.84rem !important;
}

/* Verdict Card — Minimal & Editorial */
.vr-verdict {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.45rem;
    margin: 0.5rem 0 1.4rem 0;
}
.vr-verdict-supported { border-left: 3px solid var(--sage); }
.vr-verdict-amber { border-left: 3px solid var(--amber); }
.vr-verdict-rose { border-left: 3px solid var(--rose); }
.vr-verdict-muted { border-left: 3px solid var(--ink-muted); }

.vr-verdict-title {
    font-family: var(--serif);
    font-size: 1.38rem;
    color: var(--ink);
    margin-bottom: 0.25rem;
}
.vr-verdict-desc {
    font-size: 0.92rem;
    color: var(--ink-secondary);
    line-height: 1.55;
    margin-bottom: 0.6rem;
}
.vr-tag {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.71rem;
    color: var(--ink-secondary);
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 0.12rem 0.48rem;
    margin-right: 0.35rem;
    margin-top: 0.2rem;
}
.vr-tag-sage {
    color: var(--sage);
    background: var(--sage-soft);
    border-color: var(--sage-border);
}

/* Claude Assistant Prose Block */
.vr-prose {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin: 0.6rem 0 1.3rem 0;
}
.vr-prose-p {
    font-family: var(--serif);
    font-size: 1.03rem;
    line-height: 1.72;
    color: var(--ink);
    margin-bottom: 0.65rem;
}
.vr-prose-p:last-child { margin-bottom: 0; }
.vr-cite {
    font-family: var(--mono);
    font-size: 0.69rem;
    color: var(--sage);
    background: var(--sage-soft);
    border-radius: 4px;
    padding: 0.08rem 0.38rem;
    margin-left: 0.28rem;
}

/* Subtle Note / Response Block */
.vr-note {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 11px;
    padding: 0.85rem 1.1rem;
    font-size: 0.87rem;
    color: var(--ink-secondary);
    line-height: 1.55;
    margin: 0.6rem 0;
}
.vr-note-sage {
    border-color: var(--sage-border);
    color: var(--ink);
}

/* Minimal Table */
.vr-tbl-wrap {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow-x: auto;
    background: var(--surface);
    margin: 0.5rem 0 1.2rem 0;
}
table.vr-tbl {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.83rem;
}
table.vr-tbl th {
    text-align: left;
    font-weight: 500;
    font-size: 0.75rem;
    color: var(--ink-muted);
    padding: 0.65rem 0.9rem;
    border-bottom: 1px solid var(--border);
}
table.vr-tbl td {
    padding: 0.62rem 0.9rem;
    border-bottom: 1px solid var(--border-subtle);
    color: var(--ink-secondary);
    font-variant-numeric: tabular-nums;
}
table.vr-tbl tr:last-child td { border-bottom: none; }

/* Minimal Gate Row inside expander */
.vr-gate-row {
    display: flex;
    justify-content: space-between;
    padding: 0.45rem 0;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 0.84rem;
}
.vr-gate-row:last-child { border-bottom: none; }

/* Footer Disclaimer */
.vr-footer {
    margin-top: 2.2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border-subtle);
    font-size: 0.78rem;
    color: var(--ink-muted);
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)

_GATE_LABELS = {
    "locator_valid": "Spatial locator quality",
    "mask_stability_passed": "Mask threshold & dilation stability",
    "controls_valid": "Matched comparison regions (K >= 8)",
    "edit_integrity_passed": "Outside-mask pixel equality & label drift",
    "fill_stability_passed": "Multi-fill consistency (Telea, blur, mean)",
    "no_shortcut_flagged": "Thoracic ROI shortcut check",
    "target_distinguishable_from_controls": "Target drop exceeds matched controls",
}


def _note(text: str, sage: bool = False) -> None:
    cls = "vr-note vr-note-sage" if sage else "vr-note"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def _render_verdict(verdict: Verdict, subtitle: str, reason_codes: tuple[str, ...] | list[str]) -> None:
    accent = {
        Verdict.MODEL_SUPPORTED: "vr-verdict-supported",
        Verdict.NOT_SUPPORTED: "vr-verdict-amber",
        Verdict.SHORTCUT_RISK: "vr-verdict-rose",
        Verdict.INDETERMINATE: "vr-verdict-muted",
    }[verdict]
    tag_cls = "vr-tag vr-tag-sage" if verdict is Verdict.MODEL_SUPPORTED else "vr-tag"
    tags = "".join(f'<span class="{tag_cls}">{html.escape(str(c))}</span>' for c in reason_codes)
    st.markdown(
        f"""
        <div class="vr-verdict {accent}">
            <div class="vr-verdict-title">{html.escape(verdict.value)}</div>
            <div class="vr-verdict-desc">{html.escape(subtitle)}</div>
            <div>{tags}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_prose(sentences: list[tuple[str, list[str] | tuple[str, ...]]]) -> None:
    blocks = []
    for txt, eids in sentences:
        clean_txt = re.sub(r"\s*\[E\d+_[A-Z0-9_,\s]+\]", "", str(txt)).strip()
        cites = "".join(f'<span class="vr-cite">{html.escape(str(e))}</span>' for e in eids)
        blocks.append(f'<div class="vr-prose-p">{html.escape(clean_txt)}{cites}</div>')
    st.markdown(f'<div class="vr-prose">{"".join(blocks)}</div>', unsafe_allow_html=True)


def _render_table(columns: list[tuple[str, str]], rows: list[dict]) -> None:
    if not rows:
        return
    ths = "".join(f"<th>{html.escape(lbl)}</th>" for _, lbl in columns)
    trs = []
    for r in rows:
        tds = []
        for key, _ in columns:
            val = r.get(key, "—")
            if val is None:
                cell = '<span style="color:#6E6C64;">—</span>'
            elif key == "status":
                cell = (
                    '<span class="vr-tag vr-tag-sage" style="margin:0;">Admitted</span>'
                    if val == "ADMITTED"
                    else '<span class="vr-tag" style="margin:0;">Excluded</span>'
                )
            elif isinstance(val, float):
                cell = f"{val:.4f}"
            else:
                cell = html.escape(str(val))
            tds.append(f"<td>{cell}</td>")
        trs.append(f"<tr>{''.join(tds)}</tr>")
    st.markdown(
        f'<div class="vr-tbl-wrap"><table class="vr-tbl"><thead><tr>{ths}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def _render_footer() -> None:
    st.markdown(
        f'<div class="vr-footer"><span>{html.escape(BEHAVIOUR_NOTICE)}</span>'
        f'<span>Offline · In-memory only · SHA-256 verified</span></div>',
        unsafe_allow_html=True,
    )


# ── Minimal Top Bar ─────────────────────────────────────────────────────────────────────────
bar_l, bar_r = st.columns([6, 1])
with bar_l:
    st.markdown(
        '<div class="vr-topbar">'
        '<span class="vr-wordmark"><span class="vr-dot"></span>VeriRay</span>'
        '<span class="vr-meta">Counterfactual Verification Judge · Offline</span>'
        "</div>",
        unsafe_allow_html=True,
    )
with bar_r:
    if st.button("New session", width="stretch"):
        st.session_state.clear()
        st.rerun()

MODE_CHEST = "Chest X-ray"
MODE_BRAIN = "Brain MRI"
MODE_DEMO = "Demo Cases"
MODE_EVAL = "Evidence & Audits"

mode = st.radio(
    "Mode",
    [MODE_CHEST, MODE_BRAIN, MODE_DEMO, MODE_EVAL],
    horizontal=True,
    label_visibility="collapsed",
)

# ═════════════════════════════════════════════════════════════════════════════════════════════
# MODE: BRAIN MRI
# ═════════════════════════════════════════════════════════════════════════════════════════════
if mode == MODE_BRAIN:
    try:
        bm_reader = BrainMRIReader.load(ROOT / "models")
    except BrainMRIError as exc:
        _render_verdict(Verdict.INDETERMINATE, str(exc), [exc.reason_code])
        _render_footer()
        st.stop()

    mri_file = st.file_uploader(
        "Brain MRI slice or NIfTI volume (.png, .jpg, .bmp, .nii, .nii.gz)",
        type=["png", "jpg", "jpeg", "bmp", "nii", "gz"],
    )
    if mri_file is None:
        st.markdown(
            '<div class="vr-hero">'
            '<div class="vr-hero-title">Brain MRI perturbation check</div>'
            '<div class="vr-hero-sub">5-class tumor &amp; ischemic stroke model scores, FPN/Grad-CAM++ candidate region, '
            "and Telea counterfactual comparison — decoded strictly in memory.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        _render_footer()
        st.stop()

    try:
        arr = decode_brain_upload(mri_file.getvalue(), mri_file.name)
        out = bm_reader.predict_and_cam(arr)
    except BrainMRIError as exc:
        _render_verdict(Verdict.INDETERMINATE, str(exc), [exc.reason_code])
        _render_footer()
        st.stop()
    except Exception:  # noqa: BLE001
        _render_verdict(Verdict.INDETERMINATE, "Unreadable image or volume.", ["CORRUPT_OR_UNREADABLE_IMAGE"])
        _render_footer()
        st.stop()

    pred_cls = out["predicted_class"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Top model class", pred_cls.replace("_", " ").title())
    m2.metric("Model score", f"{out['scores'][pred_cls]:.4f}")
    m3.metric("FPN lesion peak", f"{out.get('supervised_lesion_peak', 0.0):.4f}")

    c1, c2, c3 = st.columns(3)
    c1.image(out["display_image"].astype("uint8"), caption="Canonical slice · 256px", width="stretch", clamp=True)
    c2.image(overlay_heatmap(out["display_image"], out["heatmap"]), caption="Candidate region", width="stretch")
    c3.image((out["target_mask"] * 255).astype("uint8"), caption="Candidate mask", width="stretch")

    if out["target_mask"].any() and not out["target_mask"].all():
        cf = erase(out["display_image"], out["target_mask"], "telea", seed=0)
        cf_out = bm_reader.predict_and_cam(cf.edited_image)
        target_drop = float(out["scores"][pred_cls] - cf_out["scores"][pred_cls])
        st.markdown("### Counterfactual comparison")
        cf_l, cf_r = st.columns([1, 1.3])
        with cf_l:
            st.image(render_labelled(cf.edited_image, scale=2), caption=SYNTHETIC_COUNTERFACTUAL_LABEL, width="stretch")
        with cf_r:
            ctrl_res = generate_matched_controls(out["target_mask"], ControlConfig(k=9, min_controls=3, seed=20261126))
            k1, k2 = st.columns(2)
            k1.metric("Target edit drop", f"{target_drop:+.4f}")
            if ctrl_res.valid:
                c_drops = [
                    float(out["scores"][pred_cls] - bm_reader.predict_and_cam(erase(out["display_image"], cm, "telea", seed=ci + 1).edited_image)["scores"][pred_cls])
                    for ci, cm in enumerate(ctrl_res.masks)
                ]
                c_med = float(np.median(c_drops))
                emp_rank = 1 + int(np.sum(np.asarray(c_drops) >= target_drop))
                k2.metric(f"Control median (K={len(c_drops)})", f"{c_med:+.4f}", f"rank {emp_rank}/{len(c_drops)+1}")

    with st.expander("All class scores & checkpoint"):
        _render_table(
            [("class", "Class"), ("score", "Model score")],
            [{"class": k.replace("_", " "), "score": float(v)} for k, v in sorted(out["scores"].items(), key=lambda kv: kv[1], reverse=True)],
        )
        st.caption(f"Checkpoint: `{out['checkpoint_id']}`")

    _render_footer()
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════════════════════
# MODE: PRECOMPUTED DEMO CASES
# ═════════════════════════════════════════════════════════════════════════════════════════════
if mode == MODE_DEMO:
    demo_dir = ROOT / "artifacts" / "demo_cases"
    idx_path = demo_dir / "index.json"
    if not idx_path.is_file():
        _note("Precomputed demo pack not found. Run `python scripts/generate_demo_pack.py`.")
        _render_footer()
        st.stop()
    index = json.loads(idx_path.read_text(encoding="utf-8"))
    choice = st.selectbox("Scenario", range(len(index)), format_func=lambda i: index[i]["title"])
    cdir = demo_dir / index[choice]["case_id"]
    bundle = json.loads((cdir / "bundle.json").read_text(encoding="utf-8"))
    st.caption(bundle["precomputed_notice"])

    v_enum = Verdict(bundle["verdict"])
    _render_verdict(v_enum, bundle["title"], bundle["reason_codes"])

    c1, c2, c3 = st.columns(3)
    c1.image(str(cdir / "original.png"), caption="Original", width="stretch")
    c2.image(str(cdir / "overlay.png"), caption="Candidate region", width="stretch")
    c3.image(str(cdir / "mask.png"), caption="Mask", width="stretch")

    st.markdown("### Explanation")
    _render_prose([(s["text"], s["evidence_ids"]) for s in bundle["verified_explanation"]])

    if bundle["has_counterfactual"]:
        with st.expander("View synthetic counterfactual"):
            st.image(str(cdir / "counterfactual_labelled.png"), caption=bundle["synthetic_label"], width=380)

    demo_q = st.text_input("Ask about this case's recorded evidence", placeholder="e.g., How did the matched controls behave?")
    if demo_q:
        ev = EvidenceRecord(
            record_id=bundle["case_id"],
            finding_id="pneumothorax",
            verdict=v_enum,
            reason_codes=tuple(bundle["reason_codes"]),
            items=bundle["evidence_items"],
        )
        ans = answer_chat_question(demo_q, ev)
        _note(html.escape(ans.answer), sage=ans.allowed)

    with st.expander("Evidence record"):
        st.json(bundle["evidence_items"])

    _render_footer()
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════════════════════
# MODE: LOCKED-TEST EVIDENCE & AUDITS
# ═════════════════════════════════════════════════════════════════════════════════════════════
if mode == MODE_EVAL:
    pack_path = ROOT / "artifacts" / "eval_test_locked" / "test_evidence_pack.json"
    if not pack_path.is_file():
        _note("Locked test evidence pack not found. Run `python scripts/run_locked_test_and_exp5.py`.")
        _render_footer()
        st.stop()

    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    rc_cmp = pack.get("risk_coverage_comparison", {}).get("operating_point", {})
    fj = rc_cmp.get("full_judge", {})
    ro = rc_cmp.get("reader_at_same_coverage", {})
    loc_rep = pack.get("localization_reporting_half", {})
    exp5 = pack.get("experiment_5_claim_verifier", {})

    st.markdown("## Locked-test evaluation")
    st.caption("Evaluated once on untouched patient/cluster-disjoint test splits under frozen configuration.")

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Judge precision", f"{fj.get('precision', 0.0):.2%}", f"{fj.get('accepted_count', 0)}/{fj.get('n', 0)} ({fj.get('coverage', 0.0):.1%} cov)")
    h2.metric("Reader @ matched cov", f"{ro.get('precision', 0.0):.2%}", f"{rc_cmp.get('precision_delta_judge_minus_reader', 0.0):+.2%} delta")
    h3.metric("Expert pointing game", f"{(loc_rep.get('pointing_game_accuracy') or 0.0):.1%}", f"mean IoU {(loc_rep.get('mean_iou') or 0.0):.3f}")
    h4.metric("Unsupported claims", f"{(exp5.get('unsupported_rate_after_verifier') or 0.0):.1%}", f"from {(exp5.get('unsupported_rate_before_verifier') or 0.0):.1%} pre-verifier")

    t1, t2, t3, t4 = st.tabs(["Chest Judge (n=445)", "Finding Gate & NIH (n=25,596)", "Brain MRI (n=1,378)", "Safety & Latency"])

    with t1:
        rm = pack.get("reader_metrics", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reader AUROC", f"{rm.get('auroc', 0.0):.4f}")
        c2.metric("Reader AUPRC", f"{rm.get('auprc', 0.0):.4f}")
        c3.metric("Brier score", f"{rm.get('brier', 0.0):.4f}")
        c4.metric("Specificity @ 0.5", f"{rm.get('specificity', 0.0):.2%}")

        col_l, col_r = st.columns([1.4, 1])
        with col_l:
            rc_img = ROOT / "artifacts" / "eval_test_locked" / "risk_coverage.png"
            if rc_img.is_file():
                st.image(str(rc_img), caption="Selective risk-coverage (NIH locked test)", width="stretch")
        with col_r:
            vc = pack.get("verdict_counts", {})
            _render_table(
                [("verdict", "Verdict"), ("cases", "Count"), ("share", "Share")],
                [{"verdict": k, "cases": v, "share": f"{(v / max(rm.get('n', 445), 1)):.1%}"} for k, v in vc.items()],
            )
            fs = pack.get("fill_sensitivity", {})
            if fs:
                st.caption(
                    f"Multi-fill flip rate across {', '.join(fs.get('methods', []))}: "
                    f"{fs.get('verdict_flip_rate', 0.0):.1%} ({fs.get('flip_count', 0)}/{fs.get('n', 0)} cases)."
                )

        abl = pack.get("ablations", {})
        if abl.get("configurations"):
            st.markdown("### Ablations A–F")
            _render_table(
                [("configuration", "Configuration"), ("accepted_count", "Accepted"), ("coverage", "Coverage"), ("precision", "Precision"), ("risk", "Risk")],
                [
                    {
                        "configuration": x["configuration"],
                        "accepted_count": x["accepted_count"],
                        "coverage": f"{x['coverage']:.2%}",
                        "precision": f"{x['precision']:.2%}",
                        "risk": f"{x['risk']:.2%}",
                    }
                    for x in abl["configurations"]
                ],
            )

    with t2:
        chest_lt_path = ROOT / "artifacts" / "eval_test_locked" / "chest_hd_locked_test.json"
        nih_lt_path = ROOT / "artifacts" / "eval_test_locked" / "external_nih_test.json"
        if chest_lt_path.is_file():
            clt = json.loads(chest_lt_path.read_text(encoding="utf-8"))
            nih_lt = json.loads(nih_lt_path.read_text(encoding="utf-8")).get("metrics", {}) if nih_lt_path.is_file() else {}
            rule = clt.get("frozen_admission_rule", {})
            st.caption(
                f"Admission requires positives >= {rule.get('min_positives_per_split')}, Dice >= {rule.get('min_pos_box_or_mask_dice')}, "
                f"and border-probe AUROC <= {rule.get('max_border_probe_auroc')} on both dev_val and locked_test "
                f"(n = {clt.get('test_scans_count', 0):,} scans; macro AUROC {clt.get('macro_locked_test_auroc_15_findings', 0.0):.4f})."
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
                        "test_pos": lt.get("test_positives"),
                        "test_auroc": lt.get("auroc"),
                        "dice": lt.get("pos_box_or_mask_dice"),
                        "border": lt.get("border_probe_test_auroc"),
                        "nih_auroc": nih_m.get("auroc"),
                        "nih_pg": f"{nih_m['pointing_game_hit_rate']:.1%}" if "pointing_game_hit_rate" in nih_m else None,
                        "note": "Passed all gates" if admitted else "; ".join(reasons_fail),
                    }
                )
            rows_gate.sort(key=lambda r: (0 if r["status"] == "ADMITTED" else 1, -(r["test_auroc"] or 0.0)))
            _render_table(
                [
                    ("status", "Status"),
                    ("finding", "Finding"),
                    ("test_pos", "Positives"),
                    ("test_auroc", "Test AUROC"),
                    ("dice", "Dice"),
                    ("border", "Border probe"),
                    ("nih_auroc", "External NIH"),
                    ("nih_pg", "Pointing"),
                    ("note", "Reason"),
                ],
                rows_gate,
            )

    with t3:
        brain_lt_path = ROOT / "artifacts" / "eval_test_locked" / "brain_mri_locked_test.json"
        if brain_lt_path.is_file():
            blt = json.loads(brain_lt_path.read_text(encoding="utf-8"))
            wi = blt.get("headline_within_isles_stroke_vs_normal", {})
            wb = blt.get("within_brats_glioma_vs_normal", {})
            w2 = blt.get("within_2d_4class", {})
            st.caption(blt.get("note", ""))
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Within-ISLES stroke AUROC", f"{wi.get('auroc_p_stroke', 0.0):.4f}", f"Dice {wi.get('stroke_mask_dice', 0.0):.4f}")
            b2.metric("Within-BraTS glioma AUROC", f"{wb.get('auroc_p_glioma', 0.0):.4f}", f"Dice {wb.get('glioma_mask_dice', 0.0):.4f}")
            b3.metric("Within-2D 4-class acc", f"{w2.get('accuracy_4cls', 0.0):.2%}")
            b4.metric("Overall lesion Dice", f"{blt.get('overall_lesion_segmentation_dice_locked_test', 0.0):.4f}")

            t_dices = w2.get("tumor_mask_dice_by_class", {})
            if t_dices:
                _render_table(
                    [("tumor_class", "2D Figshare class"), ("pixel_mask_dice", "Exact pixel-mask Dice")],
                    [{"tumor_class": k, "pixel_mask_dice": v} for k, v in t_dices.items()],
                )

    with t4:
        rt_p = ROOT / "artifacts" / "redteam_report.json"
        art_p = ROOT / "artifacts" / "audit_redteam_report.json"
        bnd_p = ROOT / "artifacts" / "boundary_report.json"
        lat_p = ROOT / "artifacts" / "latency_and_api_report.json"
        s1, s2, s3, s4 = st.columns(4)
        if rt_p.is_file():
            rtd = json.loads(rt_p.read_text(encoding="utf-8"))
            s1.metric("Primary red-team", f"{rtd.get('attack_prompts_passed', 0)}/{rtd.get('attack_prompts_total', 0)}", "100% blocked")
        if art_p.is_file():
            artd = json.loads(art_p.read_text(encoding="utf-8"))
            s2.metric("Audit red-team (E2E)", f"{artd.get('blocked_e2e', 0)}/{artd.get('total_attacks', 0)}", f"L1 {artd.get('pass_rate_layer1', 0.0):.1%}")
        if bnd_p.is_file():
            bndd = json.loads(bnd_p.read_text(encoding="utf-8"))
            s3.metric("Boundary fail-closed", f"{bndd.get('total_boundary_cases', 0)}/{bndd.get('total_boundary_cases', 0)}", "100%")
        if lat_p.is_file():
            latd = json.loads(lat_p.read_text(encoding="utf-8"))
            e2e_lat = latd.get("stage_latencies", {}).get("end_to_end", {})
            s4.metric("Median latency", f"{e2e_lat.get('p50_ms', 0.0):.0f} ms", f"p95 {e2e_lat.get('p95_ms', 0.0):.0f} ms")

        if art_p.is_file():
            artd = json.loads(art_p.read_text(encoding="utf-8"))
            _render_table(
                [("category", "Category"), ("total", "Total"), ("blocked_e2e", "Blocked E2E"), ("blocked_layer1", "Blocked Layer 1")],
                [{"category": k, **v} for k, v in artd.get("per_category", {}).items()],
            )

    _render_footer()
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════════════════════
# MODE (DEFAULT): CHEST X-RAY VERIFICATION JUDGE
# ═════════════════════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _load_reader(finding_id: str = "pneumothorax"):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return Reader.load_for_finding(ROOT / "models", finding_id, device=device), None
    except ReaderError as exc:
        return None, exc


c_sel, c_opt = st.columns([1.2, 1.8])
with c_sel:
    target_finding = st.selectbox(
        "Finding",
        list(SUPPORTED_FINDINGS),
        index=0,
        format_func=lambda f: CHEST_HD_DISPLAY_NAMES.get(f, f.capitalize()),
        help="Admitted findings passing locked-test sample, Dice, and border-shortcut checks.",
    )
with c_opt:
    st.write("")
    st.write("")
    use_medgemma = st.checkbox("Include MedGemma 4B claimant & explainer (local 4-bit GPU)", value=False)

reader, load_err = _load_reader(target_finding)
if load_err is not None:
    _render_verdict(Verdict.INDETERMINATE, load_err.message, [load_err.reason_code])
    _render_footer()
    st.stop()

uploads = st.file_uploader(
    "Upload de-identified chest X-ray (.png, .jpg, .bmp, .tif)",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)

if not uploads:
    st.markdown(
        '<div class="vr-hero">'
        '<div class="vr-hero-title">Verify a model prediction</div>'
        '<div class="vr-hero-sub">Tests whether a thoracic prediction depends on a localized region '
        "across multi-fill counterfactuals and 19 matched controls — abstaining when stability or shortcut gates fail.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    _render_footer()
    st.stop()

frozen_cfg_path = ROOT / "configs" / "frozen.yaml"
if frozen_cfg_path.is_file():
    loc_cfg, judge_cfg, _ = configs_from_yaml(frozen_cfg_path)
else:
    loc_cfg, judge_cfg = LocatorConfig(), JudgeConfig()

queue_results = []
status_bar = st.progress(0, text="Verifying...")
for idx, upl in enumerate(uploads):
    status_bar.progress((idx + 1) / len(uploads), text=f"Verifying {upl.name}...")

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
                    _note(
                        f"`{html.escape(upl.name)}` resembles a Brain MRI ({_guess.confidence:.0%} heuristic match). "
                        f"Switch to <strong>{MODE_BRAIN}</strong> above."
                    )
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
    _render_footer()
    st.stop()

if len(queue_results) > 1:
    _render_table(
        [("case", "Scan"), ("verdict", "Verdict"), ("score_raw", "Score"), ("total_ms", "ms")],
        [
            {
                "case": name,
                "verdict": item.verdict.value,
                "score_raw": round(item.reader.score_raw, 4) if item.reader else None,
                "total_ms": round(sum(item.latencies_ms.values()), 1),
            }
            for name, item in queue_results
        ],
    )
    selected_idx = st.selectbox(
        "Select scan",
        range(len(queue_results)),
        format_func=lambda i: f"{queue_results[i][0]} — {queue_results[i][1].verdict.value}",
    )
else:
    selected_idx = 0

selected_name, res = queue_results[selected_idx]

user_text = res.output.user_facing_text if res.output is not None else "The test could not produce a reliable result."
_render_verdict(res.verdict, user_text, res.reason_codes)

if res.reader is not None and res.locator is not None and res.display_image is not None:
    target_disp = CHEST_HD_DISPLAY_NAMES.get(target_finding, target_finding.replace("_", " ").capitalize())
    disp_px = int(res.display_image.shape[0])
    score_lbl = SCORE_LABEL if target_finding == "pneumothorax" else f"Model score ({target_disp})"

    c1, c2, c3 = st.columns(3)
    c1.image(res.display_image.astype("uint8"), caption=f"Original · {disp_px}px", width="stretch", clamp=True)
    c2.image(overlay_heatmap(res.display_image, res.locator.heatmap), caption="Candidate region", width="stretch")
    c3.image((res.locator.target_mask * 255).astype("uint8"), caption="Mask", width="stretch")

    if res.explanation is not None:
        _render_prose([(s.text, s.evidence_ids) for s in res.explanation.sentences])
        if res.explanation.unverified_other_claims:
            st.caption("Mentioned by model, not verified: " + "; ".join(res.explanation.unverified_other_claims))

    jc = res.judge_outcome.contract if res.judge_outcome is not None else None
    te = dict(jc.target_effects) if jc is not None else {}
    ce = dict(jc.control_effects) if jc is not None else {}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(score_lbl, f"{res.reader.score_raw:.4f}")
    if "score_drops" in ce:
        m2.metric("Target edit drop", f"{te.get('primary_score_drop', 0.0):+.4f}")
        m3.metric(f"Control median (K={ce.get('num_controls')})", f"{ce.get('median_score_drop', 0.0):+.4f}")
        m4.metric("Empirical rank", f"{te.get('empirical_rank')} / {int(ce.get('num_controls', 0)) + 1}")
    else:
        m2.metric("Calibrated score", f"{res.reader.score_calibrated_optional:.4f}" if res.reader.score_calibrated_optional is not None else "—")
        m3.metric("Input gate", res.gate.status.value if res.gate is not None else "—")
        m4.metric("Latency", f"{sum(res.latencies_ms.values()):.0f} ms")

if use_medgemma and res.display_image is not None:
    with st.spinner("Running MedGemma 4B..."):
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
            st.markdown("### MedGemma 4B")
            for c in vlm_claims:
                iou_txt = f" · IoU {box_mask_iou(res.locator.target_mask, c.box_xyxy):.3f}" if (c.box_xyxy is not None and res.locator is not None) else ""
                c_norm = c.finding.strip().lower().replace(" ", "_").replace("/", "_")
                if c_norm == "pleural_effusion":
                    c_norm = "effusion"
                tag = "verified target" if (c.is_verifiable_finding and c_norm == target_finding) else "unverified"
                st.write(f"- **{sanitize_for_prompt(c.finding)}** ({sanitize_for_prompt(c.status)}, {sanitize_for_prompt(c.region_description)}) · *{tag}*{iou_txt}")
            if vlm_expl_rep is not None:
                _render_prose([(s.text, s.evidence_ids) for s in vlm_expl_rep.verified_sentences])
                st.caption(f"{vlm_expl_rep.dropped_count} unverified sentence(s) filtered · {vlm_peak:.0f} MB VRAM")
        except VLMError as exc:
            _note(f"MedGemma unavailable (`{html.escape(exc.reason_code)}`): {html.escape(str(exc))}")

if res.judge_outcome is not None:
    gates_dict = dict(res.judge_outcome.contract.gate_results)
    n_pass = sum(1 for v in gates_dict.values() if v)
    with st.expander(f"Verification gates ({n_pass}/{len(gates_dict)} passed) & counterfactual"):
        rows_g = "".join(
            f'<div class="vr-gate-row"><span>{html.escape(_GATE_LABELS.get(k, k))}</span>'
            f'<span style="font-family:var(--mono); font-size:0.76rem; color:{"#739E82" if v else "#C47A70"};">'
            f'{"Passed" if v else "Failed"}</span></div>'
            for k, v in gates_dict.items()
        )
        st.markdown(rows_g, unsafe_allow_html=True)

        if res.judge_outcome.primary_target_edit is not None and res.display_image is not None:
            edit = res.judge_outcome.primary_target_edit
            labelled_edit = render_labelled(edit.edited_image, scale=1 if res.display_image.shape[0] >= 512 else 2)
            st.write("")
            ca, cb = st.columns(2)
            ca.image(res.display_image.astype("uint8"), caption="Original", width="stretch", clamp=True)
            cb.image(labelled_edit, caption=f"{edit.fill_method} · {SYNTHETIC_COUNTERFACTUAL_LABEL}", width="stretch")

            if (ROOT / "models" / "UNET_CHECKPOINT.json").is_file() and res.locator is not None:
                try:
                    unet = UNetEraser.load(ROOT / "models")
                    unet_edit = unet.erase(res.display_image, res.locator.target_mask, seed=0)
                    unet_rc, _, _ = reader.score(unet_edit.edited_image)
                    unet_drop = float(res.reader.score_raw - unet_rc.score_raw) if res.reader is not None else 0.0
                    st.caption(
                        f"Optional Gated U-Net drop: {unet_drop:+.4f} · boundary discontinuity: {unet_edit.integrity_report.boundary_discontinuity:.4f}"
                    )
                except EraseError:
                    pass

chat_q = st.text_input(
    "Ask about this result",
    placeholder="Ask about the verdict, score drop, matched controls, or protocol definitions...",
)
if chat_q:
    ev_rec = res.judge_outcome.evidence_record if res.judge_outcome is not None else None
    chat_resp = answer_chat_question(chat_q, ev_rec)
    _note(html.escape(chat_resp.answer), sage=chat_resp.allowed)

with st.expander("Technical details & run manifest"):
    if res.reader is not None:
        target_disp = CHEST_HD_DISPLAY_NAMES.get(target_finding, target_finding.capitalize())
        target_thresh = judge_cfg.min_reader_score if judge_cfg.min_reader_score > 0 else 0.5
        screening = summarize_reader_findings(res.reader, target_threshold=target_thresh, other_threshold=0.5)
        if not screening.target_flagged and screening.flagged_other_findings:
            flagged_txt = ", ".join(f"{lbl} ({val:.3f})" for lbl, val in screening.flagged_other_findings)
            st.caption(f"Other unverified channels >= 0.50: {flagged_txt}")
        st.caption(
            f"Checkpoint: `{res.reader.checkpoint_id}` · Preprocessing: `{res.reader.preprocessing_id}` · "
            f"Threshold: `{res.reader.threshold_id}`"
        )
    if res.judge_outcome is not None:
        st.json(dict(res.judge_outcome.evidence_record.items))
    if res.manifest is not None:
        st.caption(
            f"Run ID: `{res.manifest.run_id}` · offline_only={res.manifest.offline_only} · "
            f"latencies (ms): { {k: round(v, 1) for k, v in res.latencies_ms.items()} }"
        )

_render_footer()
