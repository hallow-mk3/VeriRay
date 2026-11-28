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

# ── Carbon-Black × Claude Editorial Serif × Botanical Sage Design System (100% Local/Offline) ──
st.markdown(
    """
<style>
:root {
    --vr-bg: #0D0E0D;
    --vr-surface: #141614;
    --vr-surface-2: #1A1D1A;
    --vr-surface-hover: #202420;
    --vr-border: #262A27;
    --vr-border-strong: #353B36;
    --vr-ink: #ECEAE4;
    --vr-ink-soft: #C9C6BE;
    --vr-muted: #94928B;
    --vr-dim: #66655F;
    --vr-sage: #7FA88B;
    --vr-sage-bright: #98C1A4;
    --vr-sage-deep: #5E8C6A;
    --vr-sage-wash: rgba(127, 168, 139, 0.11);
    --vr-sage-border: rgba(127, 168, 139, 0.34);
    --vr-amber-wash: rgba(212, 168, 98, 0.11);
    --vr-amber-border: rgba(212, 168, 98, 0.36);
    --vr-amber-ink: #E5C48A;
    --vr-clay-wash: rgba(209, 114, 100, 0.12);
    --vr-clay-border: rgba(209, 114, 100, 0.38);
    --vr-clay-ink: #E8A398;
    --vr-slate-wash: rgba(154, 152, 144, 0.10);
    --vr-slate-border: rgba(154, 152, 144, 0.30);
    --vr-slate-ink: #C8C5BC;
    --vr-serif: Georgia, 'Iowan Old Style', Charter, 'Palatino Linotype', 'Book Antiqua', serif;
    --vr-sans: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
    --vr-mono: 'Cascadia Code', Consolas, 'SFMono-Regular', Menlo, monospace;
}

/* Force Carbon-Black canvas immediately even before server restart */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: var(--vr-bg) !important;
    color: var(--vr-ink) !important;
    font-family: var(--vr-sans);
}

header[data-testid="stHeader"] {
    background: rgba(13, 14, 13, 0.82) !important;
    backdrop-filter: blur(8px);
    border-bottom: 1px solid rgba(38, 42, 39, 0.45);
}

.block-container {
    padding-top: 3.6rem !important;
    padding-bottom: 4rem !important;
    max-width: 1320px !important;
}

/* Claude Editorial Typography */
h1, h2, h3, h4, .vr-serif {
    font-family: var(--vr-serif) !important;
    color: var(--vr-ink) !important;
    font-weight: 400 !important;
    letter-spacing: -0.02em !important;
}
h1 { font-size: 2.2rem !important; }
h2 { font-size: 1.48rem !important; margin-top: 1.5rem !important; }
h3 { font-size: 1.2rem !important; margin-top: 1.1rem !important; }
h4 { font-size: 1.05rem !important; color: var(--vr-ink-soft) !important; margin-top: 0.9rem !important; }

p, label, span, li {
    color: var(--vr-ink-soft);
}

code {
    font-family: var(--vr-mono) !important;
    background: var(--vr-surface-2) !important;
    color: var(--vr-sage-bright) !important;
    border: 1px solid var(--vr-border) !important;
    border-radius: 5px !important;
    padding: 0.1rem 0.38rem !important;
    font-size: 0.80em !important;
}

/* Masthead */
.vr-masthead {
    background: linear-gradient(180deg, rgba(26, 29, 26, 0.72) 0%, rgba(20, 22, 20, 0.92) 100%);
    border: 1px solid var(--vr-border);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.85rem;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04), 0 8px 24px rgba(0, 0, 0, 0.35);
}

.vr-brand-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.vr-orb {
    width: 11px;
    height: 11px;
    border-radius: 50%;
    background: var(--vr-sage);
    box-shadow: 0 0 12px rgba(127, 168, 139, 0.65);
    flex-shrink: 0;
}

.vr-brand-title {
    font-family: var(--vr-serif);
    font-size: 1.95rem;
    font-weight: 400;
    letter-spacing: -0.025em;
    color: var(--vr-ink);
    line-height: 1.1;
    margin: 0;
}

.vr-brand-tag {
    font-family: var(--vr-mono);
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--vr-sage-bright);
    background: var(--vr-sage-wash);
    border: 1px solid var(--vr-sage-border);
    padding: 0.18rem 0.55rem;
    border-radius: 999px;
}

.vr-brand-sub {
    color: var(--vr-muted);
    font-size: 0.88rem;
    margin-top: 0.28rem;
}

.vr-badge-cluster {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    align-items: center;
}

.vr-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.26rem 0.68rem;
    border-radius: 999px;
    font-size: 0.73rem;
    font-family: var(--vr-mono);
    background: var(--vr-surface-2);
    color: var(--vr-ink-soft);
    border: 1px solid var(--vr-border);
}
.vr-chip-sage {
    background: var(--vr-sage-wash);
    color: var(--vr-sage-bright);
    border-color: var(--vr-sage-border);
}

/* Guardrail Ribbon */
.vr-guardrail {
    background: rgba(20, 22, 20, 0.85);
    border: 1px solid var(--vr-border);
    border-left: 3px solid var(--vr-sage);
    border-radius: 10px;
    padding: 0.62rem 1rem;
    font-size: 0.83rem;
    color: var(--vr-muted);
    margin-bottom: 1.15rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}
.vr-guardrail strong {
    color: var(--vr-sage-bright);
    font-family: var(--vr-mono);
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-right: 0.5rem;
}

/* Segmented Navigation Bar (styles st.radio into sleek pill tabs) */
div[role="radiogroup"] {
    background: var(--vr-surface) !important;
    border: 1px solid var(--vr-border) !important;
    border-radius: 12px !important;
    padding: 0.32rem !important;
    gap: 0.32rem !important;
    display: flex !important;
    flex-wrap: wrap !important;
    margin-bottom: 1.1rem !important;
}
div[role="radiogroup"] > label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 9px !important;
    padding: 0.48rem 0.95rem !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: all 0.16s ease !important;
    flex: 1 1 auto !important;
    justify-content: center !important;
}
div[role="radiogroup"] > label:hover {
    background: var(--vr-surface-hover) !important;
}
div[role="radiogroup"] > label > div:first-child {
    display: none !important; /* Hide native radio circle */
}
div[role="radiogroup"] > label p {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: var(--vr-muted) !important;
}
div[role="radiogroup"] > label:has(input:checked) {
    background: var(--vr-sage-wash) !important;
    border-color: var(--vr-sage-border) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
div[role="radiogroup"] > label:has(input:checked) p {
    color: var(--vr-ink) !important;
}

/* Selectbox, Text Input & Checkbox dark styling */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: var(--vr-surface) !important;
    border-color: var(--vr-border-strong) !important;
    border-radius: 10px !important;
    color: var(--vr-ink) !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="input"] input {
    color: var(--vr-ink) !important;
}

/* Buttons */
div.stButton > button {
    background: var(--vr-surface-2) !important;
    color: var(--vr-ink-soft) !important;
    border: 1px solid var(--vr-border-strong) !important;
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.48rem 0.9rem !important;
    transition: all 0.15s ease !important;
}
div.stButton > button:hover {
    border-color: var(--vr-sage) !important;
    background: var(--vr-sage-wash) !important;
    color: var(--vr-sage-bright) !important;
}

/* File Uploader Dropzone */
section[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(180deg, #141614 0%, #111311 100%) !important;
    border: 1px dashed var(--vr-sage-border) !important;
    border-radius: 14px !important;
    padding: 1.35rem 1.25rem !important;
    transition: border-color 0.2s ease !important;
}
section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--vr-sage) !important;
}
section[data-testid="stFileUploaderDropzone"] button {
    background: var(--vr-sage-wash) !important;
    color: var(--vr-sage-bright) !important;
    border: 1px solid var(--vr-sage-border) !important;
    border-radius: 8px !important;
}

/* Metric Tiles */
div[data-testid="stMetric"] {
    background: var(--vr-surface) !important;
    border: 1px solid var(--vr-border) !important;
    border-radius: 13px !important;
    padding: 0.95rem 1.1rem !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03) !important;
}
div[data-testid="stMetricLabel"] p {
    color: var(--vr-muted) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.01em !important;
}
div[data-testid="stMetricValue"] {
    font-family: var(--vr-serif) !important;
    color: var(--vr-ink) !important;
    font-size: 1.62rem !important;
    font-variant-numeric: tabular-nums;
}
div[data-testid="stMetricDelta"] {
    font-family: var(--vr-mono) !important;
    font-size: 0.74rem !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: var(--vr-sans) !important;
    color: var(--vr-muted) !important;
    font-size: 0.85rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--vr-sage-bright) !important;
}
div[data-baseweb="tab-highlight"] {
    background-color: var(--vr-sage) !important;
}

/* Expanders */
details[data-testid="stExpander"] {
    background: var(--vr-surface) !important;
    border: 1px solid var(--vr-border) !important;
    border-radius: 12px !important;
}
details[data-testid="stExpander"] summary p {
    color: var(--vr-ink-soft) !important;
    font-size: 0.86rem !important;
}

/* Architectural Pipeline Strip (Empty State) */
.vr-arch-grid {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 0.6rem;
    margin: 1rem 0 1.2rem 0;
}
@media (max-width: 980px) {
    .vr-arch-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.vr-arch-step {
    background: var(--vr-surface);
    border: 1px solid var(--vr-border);
    border-radius: 12px;
    padding: 0.85rem 0.9rem;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.025);
}
.vr-arch-num {
    font-family: var(--vr-mono);
    font-size: 0.68rem;
    color: var(--vr-sage);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.vr-arch-title {
    font-family: var(--vr-serif);
    font-size: 0.98rem;
    color: var(--vr-ink);
    margin-bottom: 0.22rem;
}
.vr-arch-desc {
    font-size: 0.76rem;
    color: var(--vr-muted);
    line-height: 1.4;
}

/* Custom Callout Box (replaces ugly blue st.info) */
.vr-callout {
    background: var(--vr-surface);
    border: 1px solid var(--vr-border);
    border-radius: 12px;
    padding: 0.9rem 1.15rem;
    color: var(--vr-ink-soft);
    font-size: 0.88rem;
    line-height: 1.55;
    margin: 0.6rem 0;
}
.vr-callout-sage {
    background: var(--vr-sage-wash);
    border-color: var(--vr-sage-border);
    color: var(--vr-ink);
}
.vr-callout-amber {
    background: var(--vr-amber-wash);
    border-color: var(--vr-amber-border);
    color: var(--vr-amber-ink);
}

/* Verdict Hero Card */
.vr-verdict-card {
    border-radius: 15px;
    padding: 1.25rem 1.5rem;
    margin: 0.7rem 0 1.2rem 0;
    border: 1px solid var(--vr-border);
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.32);
}
.vr-verdict-supported {
    background: linear-gradient(135deg, rgba(127, 168, 139, 0.16) 0%, rgba(20, 22, 20, 0.95) 100%);
    border-color: var(--vr-sage);
}
.vr-verdict-not-supported {
    background: linear-gradient(135deg, rgba(212, 168, 98, 0.14) 0%, rgba(20, 22, 20, 0.95) 100%);
    border-color: var(--vr-amber-border);
}
.vr-verdict-shortcut {
    background: linear-gradient(135deg, rgba(209, 114, 100, 0.16) 0%, rgba(20, 22, 20, 0.95) 100%);
    border-color: var(--vr-clay-border);
}
.vr-verdict-indeterminate {
    background: linear-gradient(135deg, rgba(154, 152, 144, 0.12) 0%, rgba(20, 22, 20, 0.95) 100%);
    border-color: var(--vr-slate-border);
}
.vr-verdict-eyebrow {
    font-family: var(--vr-mono);
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--vr-muted);
    margin-bottom: 0.2rem;
}
.vr-verdict-heading {
    font-family: var(--vr-serif);
    font-size: 1.55rem;
    color: var(--vr-ink);
    margin-bottom: 0.3rem;
}
.vr-verdict-sub {
    font-size: 0.94rem;
    color: var(--vr-ink-soft);
    line-height: 1.5;
    margin-bottom: 0.65rem;
}
.vr-code-chip {
    display: inline-block;
    font-family: var(--vr-mono);
    font-size: 0.72rem;
    background: rgba(13, 14, 13, 0.65);
    color: var(--vr-ink-soft);
    border: 1px solid var(--vr-border-strong);
    border-radius: 6px;
    padding: 0.15rem 0.52rem;
    margin-right: 0.38rem;
    margin-top: 0.2rem;
}

/* 7-Gate Verification Grid */
.vr-gate-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(225px, 1fr));
    gap: 0.55rem;
    margin: 0.55rem 0 1.2rem 0;
}
.vr-gate-item {
    background: var(--vr-surface);
    border: 1px solid var(--vr-border);
    border-radius: 10px;
    padding: 0.62rem 0.85rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    font-size: 0.82rem;
    color: var(--vr-ink-soft);
}
.vr-gate-pass {
    font-family: var(--vr-mono);
    font-size: 0.69rem;
    letter-spacing: 0.05em;
    color: var(--vr-sage-bright);
    background: var(--vr-sage-wash);
    border: 1px solid var(--vr-sage-border);
    padding: 0.12rem 0.5rem;
    border-radius: 999px;
}
.vr-gate-fail {
    font-family: var(--vr-mono);
    font-size: 0.69rem;
    letter-spacing: 0.05em;
    color: var(--vr-clay-ink);
    background: var(--vr-clay-wash);
    border: 1px solid var(--vr-clay-border);
    padding: 0.12rem 0.5rem;
    border-radius: 999px;
}

/* Radiology PACS Viewport Header */
.vr-viewport-label {
    font-family: var(--vr-mono);
    font-size: 0.70rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--vr-muted);
    background: var(--vr-surface);
    border: 1px solid var(--vr-border);
    border-radius: 8px;
    padding: 0.32rem 0.65rem;
    margin-bottom: 0.4rem;
    display: flex;
    justify-content: space-between;
}
.vr-viewport-label span:last-child {
    color: var(--vr-sage);
}

/* Claude Editorial Explanation Card */
.vr-prose-card {
    background: var(--vr-surface);
    border: 1px solid var(--vr-border);
    border-radius: 14px;
    padding: 1.25rem 1.45rem;
    margin: 0.55rem 0 1.1rem 0;
}
.vr-prose-line {
    font-family: var(--vr-serif);
    font-size: 1.04rem;
    line-height: 1.7;
    color: var(--vr-ink);
    margin-bottom: 0.65rem;
}
.vr-prose-line:last-child { margin-bottom: 0; }
.vr-cite {
    font-family: var(--vr-mono);
    font-size: 0.70rem;
    background: var(--vr-sage-wash);
    color: var(--vr-sage-bright);
    border: 1px solid var(--vr-sage-border);
    border-radius: 5px;
    padding: 0.09rem 0.42rem;
    margin-left: 0.32rem;
}

/* Clean Carbon HTML Table */
.vr-table-wrap {
    border: 1px solid var(--vr-border);
    border-radius: 12px;
    overflow-x: auto;
    background: var(--vr-surface);
    margin: 0.5rem 0 1.1rem 0;
}
table.vr-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.83rem;
}
table.vr-table th {
    text-align: left;
    font-family: var(--vr-mono);
    font-size: 0.70rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--vr-muted);
    background: var(--vr-surface-2);
    padding: 0.65rem 0.85rem;
    border-bottom: 1px solid var(--vr-border);
}
table.vr-table td {
    padding: 0.62rem 0.85rem;
    border-bottom: 1px solid rgba(38, 42, 39, 0.65);
    color: var(--vr-ink-soft);
    font-variant-numeric: tabular-nums;
}
table.vr-table tr:last-child td { border-bottom: none; }
table.vr-table tr:hover td { background: rgba(255, 255, 255, 0.015); }
</style>
""",
    unsafe_allow_html=True,
)

_GATE_LABELS = {
    "locator_valid": "01 · Spatial locator quality",
    "mask_stability_passed": "02 · Threshold & dilation IoU",
    "controls_valid": "03 · Matched controls (K>=8)",
    "edit_integrity_passed": "04 · Pixel equality & drift gate",
    "fill_stability_passed": "05 · Multi-fill spread stability",
    "no_shortcut_flagged": "06 · Thoracic ROI shortcut gate",
    "target_distinguishable_from_controls": "07 · Target drop > controls",
}


def _callout(text: str, tone: str = "neutral") -> None:
    cls = {"sage": "vr-callout vr-callout-sage", "amber": "vr-callout vr-callout-amber"}.get(tone, "vr-callout")
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def _viewport_header(code: str, right_tag: str) -> None:
    st.markdown(
        f'<div class="vr-viewport-label"><span>{html.escape(code)}</span><span>{html.escape(right_tag)}</span></div>',
        unsafe_allow_html=True,
    )


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
            <div class="vr-verdict-eyebrow">Verification Judge Verdict</div>
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


def _render_html_table(columns: list[tuple[str, str]], rows: list[dict]) -> None:
    if not rows:
        return
    ths = "".join(f"<th>{html.escape(lbl)}</th>" for _, lbl in columns)
    trs = []
    for r in rows:
        tds = []
        for key, _ in columns:
            val = r.get(key, "—")
            if val is None:
                cell = '<span style="color:#66655F;">—</span>'
            elif key == "status":
                cell = (
                    '<span class="vr-gate-pass">ADMITTED</span>'
                    if val == "ADMITTED"
                    else '<span class="vr-code-chip" style="margin:0;">EXCLUDED</span>'
                )
            elif isinstance(val, float):
                cell = f"{val:.4f}"
            else:
                cell = html.escape(str(val))
            tds.append(f"<td>{cell}</td>")
        trs.append(f"<tr>{''.join(tds)}</tr>")
    st.markdown(
        f'<div class="vr-table-wrap"><table class="vr-table"><thead><tr>{ths}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def _render_pipeline_strip() -> None:
    steps = [
        ("01 · Gate", "Input & Bomb Gate", "Rejects color selfies, inverted films, corrupt files & oversized payloads."),
        ("02 · Reader", "512px ConvNeXt", "De-confounded thoracic Reader with SHA-256 checkpoint verification."),
        ("03 · Locator", "FPN + Grad-CAM++", "128x128 supervised pyramid fused with 16x16 gradient attribution."),
        ("04 · Eraser", "Multi-Fill Edits", "Telea, Gaussian blur, mean fill & optional Gated U-Net with exact pixel equality."),
        ("05 · Controls", "K=19 Matched ROIs", "Anatomically zone-matched comparison masks to isolate region specificity."),
        ("06 · Judge", "7-Gate Abstention", "Enforces stability, drift, thoracic ROI & Claim Verifier before verdict."),
    ]
    items = "".join(
        f'<div class="vr-arch-step"><div class="vr-arch-num">{s[0]}</div>'
        f'<div class="vr-arch-title">{s[1]}</div><div class="vr-arch-desc">{s[2]}</div></div>'
        for s in steps
    )
    st.markdown(f'<div class="vr-arch-grid">{items}</div>', unsafe_allow_html=True)


# ── Masthead & Guardrail ────────────────────────────────────────────────────────────────────
top_l, top_r = st.columns([7, 1])
with top_l:
    st.markdown(
        """
        <div class="vr-masthead">
            <div>
                <div class="vr-brand-row">
                    <span class="vr-orb"></span>
                    <p class="vr-brand-title">VeriRay</p>
                    <span class="vr-brand-tag">v2.0 · Research Judge</span>
                </div>
                <div class="vr-brand-sub">Counterfactual Perturbation, Matched-Control &amp; Shortcut-Aware Verification Protocol</div>
            </div>
            <div class="vr-badge-cluster">
                <span class="vr-chip vr-chip-sage">100% Offline</span>
                <span class="vr-chip vr-chip-sage">Zero-Disk RAM</span>
                <span class="vr-chip">SHA-256 Locked</span>
                <span class="vr-chip">512px Native</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with top_r:
    st.write("")
    if st.button("Reset", width="stretch", help="Clear in-memory session state"):
        st.session_state.clear()
        st.rerun()

st.markdown(
    f'<div class="vr-guardrail"><div><strong>Guardrail</strong>{html.escape(BEHAVIOUR_NOTICE)}</div>'
    f'<span style="font-family:var(--vr-mono); font-size:0.72rem; color:#7FA88B;">Admitted: pneumothorax · effusion · pneumonia</span></div>',
    unsafe_allow_html=True,
)

MODE_CHEST = "Chest X-ray Judge"
MODE_BRAIN = "Brain MRI (5-Class + NIfTI)"
MODE_DEMO = "3-Minute Stage Cases"
MODE_EVAL = "Locked-Test Evidence Pack"

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
    st.markdown("### Brain MRI — Within-Source Scores, FPN Lesion Map & Counterfactual Perturbation")
    st.caption(
        "Accepts 2D MRI slices (PNG/JPG/BMP) and 3D NIfTI volumes (.nii / .nii.gz) decoded 100% in RAM "
        "with canonical skull-peel, PSF equalization, and 12-shell radial intensity standardization."
    )
    try:
        bm_reader = BrainMRIReader.load(ROOT / "models")
    except BrainMRIError as exc:
        _render_verdict_card(Verdict.INDETERMINATE, str(exc), [exc.reason_code])
        st.stop()

    mri_file = st.file_uploader(
        "Drop a de-identified Brain MRI slice or 3D NIfTI volume (.png, .jpg, .bmp, .nii, .nii.gz)",
        type=["png", "jpg", "jpeg", "bmp", "nii", "gz"],
    )
    if mri_file is None:
        _callout(
            "Awaiting Brain MRI input. Non-MRI color photos, tiny arrays (&lt;64x64), and oversized NIfTI volumes "
            "fail closed automatically via the in-memory Brain Input Gate.",
            tone="sage",
        )
        st.stop()

    try:
        arr = decode_brain_upload(mri_file.getvalue(), mri_file.name)
        out = bm_reader.predict_and_cam(arr)
    except BrainMRIError as exc:
        _render_verdict_card(Verdict.INDETERMINATE, str(exc), [exc.reason_code])
        st.stop()
    except Exception:  # noqa: BLE001
        _render_verdict_card(Verdict.INDETERMINATE, "Unreadable or corrupt Brain MRI payload.", ["CORRUPT_OR_UNREADABLE_IMAGE"])
        st.stop()

    pred_cls = out["predicted_class"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Top Model Class (not a diagnosis)", pred_cls.replace("_", " ").title())
    m2.metric("Top Class Model Score", f"{out['scores'][pred_cls]:.4f}")
    m3.metric("FPN Lesion Peak", f"{out.get('supervised_lesion_peak', 0.0):.4f}")

    _render_html_table(
        [("class", "Model Class"), ("model_score", "Uncalibrated Score")],
        [{"class": k.replace("_", " "), "model_score": float(v)} for k, v in sorted(out["scores"].items(), key=lambda kv: kv[1], reverse=True)],
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        _viewport_header("VIEWPORT 01 · CANONICAL", "256x256")
        st.image(out["display_image"].astype("uint8"), width="stretch", clamp=True)
    with c2:
        _viewport_header("VIEWPORT 02 · CANDIDATE MAP", "FPN + CAM++")
        st.image(overlay_heatmap(out["display_image"], out["heatmap"]), width="stretch")
    with c3:
        _viewport_header("VIEWPORT 03 · BINARY MASK", "THR >= 0.45")
        st.image((out["target_mask"] * 255).astype("uint8"), width="stretch")

    if out["target_mask"].any() and not out["target_mask"].all():
        cf = erase(out["display_image"], out["target_mask"], "telea", seed=0)
        cf_out = bm_reader.predict_and_cam(cf.edited_image)
        target_drop = float(out["scores"][pred_cls] - cf_out["scores"][pred_cls])
        st.markdown(f"### Counterfactual Perturbation · `{SYNTHETIC_COUNTERFACTUAL_LABEL}`")
        cf_col1, cf_col2 = st.columns([1, 1.3])
        with cf_col1:
            _viewport_header("VIEWPORT 04 · TELEA INPAINT", "OUTSIDE MASK EXACT")
            st.image(render_labelled(cf.edited_image, scale=2), width="stretch")
        with cf_col2:
            k1, k2 = st.columns(2)
            k1.metric(f"Baseline `{pred_cls}` Score", f"{out['scores'][pred_cls]:.4f}")
            k2.metric("Target Edit Score Drop", f"{target_drop:+.4f}")
            ctrl_res = generate_matched_controls(out["target_mask"], ControlConfig(k=9, min_controls=3, seed=20261126))
            if ctrl_res.valid:
                c_drops = [
                    float(out["scores"][pred_cls] - bm_reader.predict_and_cam(erase(out["display_image"], cm, "telea", seed=ci + 1).edited_image)["scores"][pred_cls])
                    for ci, cm in enumerate(ctrl_res.masks)
                ]
                c_med = float(np.median(c_drops))
                emp_rank = 1 + int(np.sum(np.asarray(c_drops) >= target_drop))
                k3, k4 = st.columns(2)
                k3.metric(f"Control Median Drop (K={len(c_drops)})", f"{c_med:+.4f}")
                k4.metric("Empirical Rank (not p-value)", f"{emp_rank} / {len(c_drops) + 1}")
    st.caption(f"Checkpoint: `{out['checkpoint_id']}` · {BEHAVIOUR_NOTICE}")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════════════════════
# MODE 2: PRECOMPUTED 3-MINUTE STAGE DEMO CASES
# ═════════════════════════════════════════════════════════════════════════════════════════════
if mode == MODE_DEMO:
    demo_dir = ROOT / "artifacts" / "demo_cases"
    idx_path = demo_dir / "index.json"
    if not idx_path.is_file():
        _callout("Precomputed demo pack not generated yet. Run `python scripts/generate_demo_pack.py`.")
        st.stop()
    index = json.loads(idx_path.read_text(encoding="utf-8"))
    choice = st.selectbox("Select precomputed stage scenario", range(len(index)), format_func=lambda i: index[i]["title"])
    cdir = demo_dir / index[choice]["case_id"]
    bundle = json.loads((cdir / "bundle.json").read_text(encoding="utf-8"))
    st.caption(bundle["precomputed_notice"])
    v_enum = Verdict(bundle["verdict"])
    _render_verdict_card(v_enum, bundle["title"], bundle["reason_codes"])

    c1, c2, c3 = st.columns(3)
    with c1:
        _viewport_header("VIEWPORT 01 · RADIOGRAPH", "SYNTHETIC DEMO")
        st.image(str(cdir / "original.png"), width="stretch")
    with c2:
        _viewport_header("VIEWPORT 02 · CANDIDATE OVERLAY", "HEATMAP")
        st.image(str(cdir / "overlay.png"), width="stretch")
    with c3:
        _viewport_header("VIEWPORT 03 · BINARY MASK", "ROI")
        st.image(str(cdir / "mask.png"), width="stretch")

    if bundle["has_counterfactual"]:
        st.markdown(f"### Counterfactual · `{bundle['synthetic_label']}`")
        st.image(str(cdir / "counterfactual_labelled.png"), width=420)

    st.markdown("### Rule-Verified Grounded Explanation")
    _render_explanation_card([(s["text"], s["evidence_ids"]) for s in bundle["verified_explanation"]])

    demo_q = st.text_input("Query the 5-layer guardrailed chat on this precomputed EvidenceRecord:")
    if demo_q:
        ev = EvidenceRecord(
            record_id=bundle["case_id"],
            finding_id="pneumothorax",
            verdict=v_enum,
            reason_codes=tuple(bundle["reason_codes"]),
            items=bundle["evidence_items"],
        )
        ans = answer_chat_question(demo_q, ev)
        _callout(html.escape(ans.answer), tone="sage" if ans.allowed else "amber")

    with st.expander("Structured EvidenceRecord JSON"):
        st.json(bundle["evidence_items"])
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════════════════════
# MODE 3: LOCKED-TEST EVIDENCE PACK & METHODOLOGICAL AUDITS
# ═════════════════════════════════════════════════════════════════════════════════════════════
if mode == MODE_EVAL:
    pack_path = ROOT / "artifacts" / "eval_test_locked" / "test_evidence_pack.json"
    if not pack_path.is_file():
        _callout("Locked test evidence pack not found. Run `python scripts/run_locked_test_and_exp5.py` first.")
        st.stop()
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    rc_cmp = pack.get("risk_coverage_comparison", {}).get("operating_point", {})
    fj = rc_cmp.get("full_judge", {})
    ro = rc_cmp.get("reader_at_same_coverage", {})
    loc_rep = pack.get("localization_reporting_half", {})
    exp5 = pack.get("experiment_5_claim_verifier", {})

    st.markdown("### Locked-Test Evidence Pack (One-Shot Unlock · Zero Test Tuning)")
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
        "Post-Verifier Hallucination Rate",
        f"{(exp5.get('unsupported_rate_after_verifier') or 0.0):.1%}",
        f"down from {(exp5.get('unsupported_rate_before_verifier') or 0.0):.1%} raw VLM",
    )

    tab_judge, tab_chest, tab_brain, tab_safety = st.tabs(
        [
            "01 · Full Judge & Ablations (n=445)",
            "02 · 15→3 Admission Gate & NIH (n=25,596)",
            "03 · Brain MRI Within-Source Test (n=1,378)",
            "04 · Red-Team, Boundary & Latency",
        ]
    )

    with tab_judge:
        st.caption(pack.get("note", ""))
        rm = pack.get("reader_metrics", {})
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Reader AUROC", f"{rm.get('auroc', 0.0):.4f}")
        c2.metric("Reader AUPRC", f"{rm.get('auprc', 0.0):.4f}")
        c3.metric("Brier Score", f"{rm.get('brier', 0.0):.4f}")
        c4.metric("Sensitivity @ 0.5", f"{rm.get('sensitivity', 0.0):.2%}")
        c5.metric("Specificity @ 0.5", f"{rm.get('specificity', 0.0):.2%}")

        col_plot, col_verdicts = st.columns([1.45, 1])
        with col_plot:
            rc_img = ROOT / "artifacts" / "eval_test_locked" / "risk_coverage.png"
            if rc_img.is_file():
                _viewport_header("SELECTIVE RISK-COVERAGE", "LOCKED NIH TEST")
                st.image(str(rc_img), width="stretch")
        with col_verdicts:
            st.markdown("#### Verdict Distribution (`n = 445`)")
            vc = pack.get("verdict_counts", {})
            _render_html_table(
                [("verdict", "Verdict"), ("cases", "Count"), ("share", "Share")],
                [{"verdict": k, "cases": v, "share": f"{(v / max(rm.get('n', 445), 1)):.2%}"} for k, v in vc.items()],
            )
            fs = pack.get("fill_sensitivity", {})
            if fs:
                _callout(
                    f"<strong>Multi-Fill Sensitivity:</strong> Evaluated <code>{fs.get('n', 0)}</code> cases across "
                    f"<code>{', '.join(fs.get('methods', []))}</code> · Verdict flip rate: "
                    f"<strong>{fs.get('verdict_flip_rate', 0.0):.2%}</strong> (<code>{fs.get('flip_count', 0)}</code> flips gated by Judge).",
                )

        st.markdown("#### Protocol Ablations A–F")
        abl = pack.get("ablations", {})
        if abl.get("configurations"):
            _render_html_table(
                [
                    ("configuration", "Configuration"),
                    ("accepted_count", "Accepted"),
                    ("coverage", "Coverage"),
                    ("precision", "Precision"),
                    ("risk", "Error Risk"),
                ],
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

    with tab_chest:
        chest_lt_path = ROOT / "artifacts" / "eval_test_locked" / "chest_hd_locked_test.json"
        nih_lt_path = ROOT / "artifacts" / "eval_test_locked" / "external_nih_test.json"
        if chest_lt_path.is_file():
            clt = json.loads(chest_lt_path.read_text(encoding="utf-8"))
            nih_lt = json.loads(nih_lt_path.read_text(encoding="utf-8")).get("metrics", {}) if nih_lt_path.is_file() else {}
            rule = clt.get("frozen_admission_rule", {})
            st.markdown(
                f"#### Frozen 15→3 Finding Admission Audit (`n = {clt.get('test_scans_count', 0):,}` Locked Test Scans · "
                f"Macro AUROC `{clt.get('macro_locked_test_auroc_15_findings', 0.0):.4f}`)"
            )
            st.caption(
                f"Required on BOTH dev_val and locked_test: positives >= {rule.get('min_positives_per_split')}, "
                f"box/mask Dice >= {rule.get('min_pos_box_or_mask_dice')}, border-probe AUROC <= {rule.get('max_border_probe_auroc')}, "
                "and zero cross-source confound. Tuberculosis was dropped completely after border-only probes exposed a 97.5% framing shortcut."
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
                        "test_auroc": lt.get("auroc"),
                        "dice": lt.get("pos_box_or_mask_dice"),
                        "border_probe": lt.get("border_probe_test_auroc"),
                        "nih_auroc": nih_m.get("auroc"),
                        "nih_pg": f"{nih_m['pointing_game_hit_rate']:.2%}" if "pointing_game_hit_rate" in nih_m else None,
                        "gate_note": "Passed dev_val + locked_test" if admitted else "; ".join(reasons_fail),
                    }
                )
            rows_gate.sort(key=lambda r: (0 if r["status"] == "ADMITTED" else 1, -(r["test_auroc"] or 0.0)))
            _render_html_table(
                [
                    ("status", "Gate"),
                    ("finding", "Finding"),
                    ("supervision", "Supervision"),
                    ("test_pos", "Test Pos"),
                    ("test_auroc", "Within-Src AUROC"),
                    ("dice", "Box/Mask Dice"),
                    ("border_probe", "Border Probe"),
                    ("nih_auroc", "Zero-Shot NIH"),
                    ("nih_pg", "NIH Pointing"),
                    ("gate_note", "Audit Decision"),
                ],
                rows_gate,
            )

    with tab_brain:
        brain_lt_path = ROOT / "artifacts" / "eval_test_locked" / "brain_mri_locked_test.json"
        brain_aud_path = ROOT / "data" / "manifests" / "brain_mri_split_audit.json"
        if brain_lt_path.is_file():
            blt = json.loads(brain_lt_path.read_text(encoding="utf-8"))
            wi = blt.get("headline_within_isles_stroke_vs_normal", {})
            wb = blt.get("within_brats_glioma_vs_normal", {})
            w2 = blt.get("within_2d_4class", {})
            st.markdown(f"#### Within-Source Brain MRI Locked Test (`n = {blt.get('test_slices_count', 0):,}` Slices)")
            st.caption(blt.get("note", ""))
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Within-ISLES Stroke AUROC", f"{wi.get('auroc_p_stroke', 0.0):.4f}", f"Mask Dice {wi.get('stroke_mask_dice', 0.0):.4f}")
            b2.metric("Within-BraTS Glioma AUROC", f"{wb.get('auroc_p_glioma', 0.0):.4f}", f"Mask Dice {wb.get('glioma_mask_dice', 0.0):.4f}")
            b3.metric("Within-2D 4-Class Accuracy", f"{w2.get('accuracy_4cls', 0.0):.2%}", f"n={w2.get('n_slices', 0)} slices")
            b4.metric("Overall Lesion Pixel Dice", f"{blt.get('overall_lesion_segmentation_dice_locked_test', 0.0):.4f}")

            t_dices = w2.get("tumor_mask_dice_by_class", {})
            if t_dices:
                st.markdown("#### Figshare 2D Exact Pixel-Mask Tumor Dice (Locked Test)")
                _render_html_table(
                    [("tumor_class", "Tumor Class"), ("pixel_mask_dice", "Exact Pixel-Mask Dice")],
                    [{"tumor_class": k, "pixel_mask_dice": v} for k, v in t_dices.items()],
                )

            if brain_aud_path.is_file():
                baud = json.loads(brain_aud_path.read_text(encoding="utf-8"))
                p8 = baud.get("border_probe_after_canonical_8pct_outer_frame", {})
                p14 = baud.get("parenchymal_rim_probe_after_canonical_14pct", {})
                _callout(
                    f"<strong>De-Confounding Audit:</strong> Purged <code>{baud.get('dropped_synthetic_tr_aug_copies', 0)}</code> "
                    f"synthetic <code>Tr-aug</code> copies. Outer-frame 8% source probe dropped from "
                    f"<code>{baud.get('border_probe_before_canonical_8pct', {}).get('val_acc', 0.0):.2%}</code> to "
                    f"<code>{p8.get('val_acc', 0.0):.2%}</code> (chance <code>33.33%</code>); inner 14% cortical strip retains "
                    f"<code>{p14.get('val_acc', 0.0):.2%}</code> MRI sequence contrast (DWI vs T1ce), motivating strict within-source headlines.",
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
            s2.metric("Audit Red-Team (E2E)", f"{artd.get('blocked_e2e', 0)}/{artd.get('total_attacks', 0)}", f"Layer-1: {artd.get('pass_rate_layer1', 0.0):.1%}")
        if bnd_p.is_file():
            bndd = json.loads(bnd_p.read_text(encoding="utf-8"))
            s3.metric("Boundary Fail-Closed", f"{bndd.get('total_boundary_cases', 0)}/{bndd.get('total_boundary_cases', 0)}", f"{bndd.get('fail_closed_rate', 0.0):.1%} safe")
        if lat_p.is_file():
            latd = json.loads(lat_p.read_text(encoding="utf-8"))
            e2e_lat = latd.get("stage_latencies", {}).get("end_to_end", {})
            s4.metric("Median End-to-End Latency", f"{e2e_lat.get('p50_ms', 0.0):.0f} ms", f"p95: {e2e_lat.get('p95_ms', 0.0):.0f} ms")

        if art_p.is_file():
            artd = json.loads(art_p.read_text(encoding="utf-8"))
            st.markdown("#### Independent Adversarial Red-Team Breakdown (130 Attacks)")
            _render_html_table(
                [
                    ("category", "Attack Vector"),
                    ("total", "Total"),
                    ("blocked_e2e", "Blocked E2E"),
                    ("blocked_layer1", "Blocked Layer-1"),
                ],
                [{"category": k, **v} for k, v in artd.get("per_category", {}).items()],
            )
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════════════════════
# MODE 4 (DEFAULT): LIVE CHEST X-RAY VERIFICATION JUDGE
# ═════════════════════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading SHA-256 verified Reader from models/ ...")
def _load_reader(finding_id: str = "pneumothorax"):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return Reader.load_for_finding(ROOT / "models", finding_id, device=device), None
    except ReaderError as exc:
        return None, exc


ctrl_col1, ctrl_col2 = st.columns([1.4, 2])
with ctrl_col1:
    _avail_findings = list(SUPPORTED_FINDINGS)
    target_finding = st.selectbox(
        "Target verified finding",
        _avail_findings,
        index=0,
        format_func=lambda f: f"{CHEST_HD_DISPLAY_NAMES.get(f, f.capitalize())}  ·  {f}",
        help="Strictly restricted to findings passing positives>=100, Dice>=0.50, and border-probe<=0.80 on both dev_val and locked_test.",
    )
with ctrl_col2:
    st.write("")
    st.write("")
    use_medgemma = st.checkbox(
        "Enable MedGemma 4B Multimodal Claimant & Grounded Explainer (local 4-bit GPU, ~3.4 GB VRAM)",
        value=False,
    )

reader, load_err = _load_reader(target_finding)
if load_err is not None:
    _render_verdict_card(Verdict.INDETERMINATE, load_err.message, [load_err.reason_code])
    st.stop()

uploads = st.file_uploader(
    "Drop de-identified PA/AP chest radiographs (.png, .jpg, .bmp, .tif — single scan or batch queue)",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)
if not uploads:
    _render_pipeline_strip()
    _callout(
        "<strong>Ready for Verification:</strong> Drop one or more de-identified chest X-rays above, "
        "or switch to <strong>3-Minute Stage Cases</strong> / <strong>Locked-Test Evidence Pack</strong> "
        "in the segmented bar to inspect pre-verified counterfactual &amp; audit bundles.",
        tone="sage",
    )
    st.stop()

frozen_cfg_path = ROOT / "configs" / "frozen.yaml"
if frozen_cfg_path.is_file():
    loc_cfg, judge_cfg, _ = configs_from_yaml(frozen_cfg_path)
else:
    loc_cfg, judge_cfg = LocatorConfig(), JudgeConfig()

queue_results = []
status_bar = st.progress(0, text="Executing 7-gate Verification Judge...")
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
                    _callout(
                        f"<strong>Modality Mismatch (`{html.escape(upl.name)}`):</strong> Image geometry matches a "
                        f"<strong>Brain MRI</strong> ({_guess.confidence:.0%} heuristic score). "
                        f"Switch to <strong>{MODE_BRAIN}</strong> above.",
                        tone="amber",
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
    st.stop()

if len(queue_results) > 1:
    st.markdown(f"### Batch Verification Queue (`{len(queue_results)}` Scans)")
    _render_html_table(
        [
            ("case", "Scan"),
            ("verdict", "Judge Verdict"),
            ("score_raw", "Raw Score"),
            ("reasons", "Reason Codes"),
            ("total_ms", "Latency (ms)"),
        ],
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
    )
    selected_idx = st.selectbox(
        "Inspect scan",
        range(len(queue_results)),
        format_func=lambda i: f"{queue_results[i][0]}  —  {queue_results[i][1].verdict.value}",
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
            st.markdown("### MedGemma 4B Multimodal Claimant (Local 4-Bit GPU)")
            for c in vlm_claims:
                iou_txt = ""
                if c.box_xyxy is not None and res.locator is not None:
                    iou_txt = f" · IoU with Locator mask: {box_mask_iou(res.locator.target_mask, c.box_xyxy):.3f}"
                c_norm = c.finding.strip().lower().replace(" ", "_").replace("/", "_")
                if c_norm == "pleural_effusion":
                    c_norm = "effusion"
                tag = "verified target finding" if (c.is_verifiable_finding and c_norm == target_finding) else "mentioned by model, not verified"
                s_finding = sanitize_for_prompt(c.finding)
                s_status = sanitize_for_prompt(c.status)
                s_region = sanitize_for_prompt(c.region_description)
                st.write(f"- **{s_finding}** (`{s_status}`, {s_region}) — *{tag}*{iou_txt}")
            if vlm_expl_rep is not None:
                st.caption(
                    f"MedGemma grounded explainer: {len(vlm_expl_rep.verified_sentences)} verified sentences, "
                    f"{vlm_expl_rep.dropped_count} dropped by Claim Verifier · peak VRAM {vlm_peak:.0f} MB"
                )
                _render_explanation_card([(s.text, s.evidence_ids) for s in vlm_expl_rep.verified_sentences])
        except VLMError as exc:
            _callout(f"MedGemma unavailable (<code>{html.escape(exc.reason_code)}</code>): {html.escape(str(exc))}", tone="amber")

user_text = res.output.user_facing_text if res.output is not None else "The test could not produce a reliable result."
_render_verdict_card(res.verdict, user_text, res.reason_codes)

if res.judge_outcome is not None:
    st.markdown("#### 7-Gate Verification Checklist")
    _render_gate_grid(dict(res.judge_outcome.contract.gate_results))

if res.reader is not None and res.locator is not None and res.display_image is not None:
    target_disp = CHEST_HD_DISPLAY_NAMES.get(target_finding, target_finding.replace("_", " ").capitalize())
    disp_px = int(res.display_image.shape[0])
    score_lbl = SCORE_LABEL if target_finding == "pneumothorax" else f"Model score (uncalibrated {target_disp} output)"

    c1, c2, c3 = st.columns(3)
    with c1:
        _viewport_header("VIEWPORT 01 · RADIOGRAPH", f"NATIVE {disp_px}PX")
        st.image(res.display_image.astype("uint8"), width="stretch", clamp=True)
    with c2:
        _viewport_header("VIEWPORT 02 · CANDIDATE OVERLAY", "FPN + CAM++")
        st.image(overlay_heatmap(res.display_image, res.locator.heatmap), width="stretch")
    with c3:
        _viewport_header("VIEWPORT 03 · BINARY MASK", f"AREA {float(res.locator.target_mask.mean()):.1%}")
        st.image((res.locator.target_mask * 255).astype("uint8"), width="stretch")

    st.markdown("### Reader Telemetry & Provenance")
    m1, m2, m3 = st.columns(3)
    m1.metric(score_lbl, f"{res.reader.score_raw:.4f}")
    if res.reader.score_calibrated_optional is not None:
        m2.metric("Platt-Calibrated Score (val)", f"{res.reader.score_calibrated_optional:.4f}")
    else:
        m2.metric("Calibration State", "Uncalibrated Raw")
    m3.metric("Input Gate Status", res.gate.status.value if res.gate is not None else "N/A")

    target_thresh = judge_cfg.min_reader_score if judge_cfg.min_reader_score > 0 else 0.5
    screening = summarize_reader_findings(res.reader, target_threshold=target_thresh, other_threshold=0.5)
    if screening.all_scores_low:
        _callout(
            "<strong>Reader Screening Summary:</strong> No monitored thoracic channel exceeded threshold. "
            "Describes uncalibrated model scores only, never disease absence."
        )
    elif not screening.target_flagged and screening.flagged_other_findings:
        flagged_txt = ", ".join(f"{lbl} ({val:.3f})" for lbl, val in screening.flagged_other_findings)
        _callout(
            f"<strong>Cross-Channel Note:</strong> {html.escape(target_disp)} is below threshold while unverified "
            f"channel(s) scored &gt;= 0.50: <code>{html.escape(flagged_txt)}</code>.",
            tone="amber",
        )

    with st.expander("Monitored Uncoupled Channels & Checkpoint Provenance"):
        rows_unrel = [{"channel": target_disp + " (target)", "score": round(res.reader.score_raw, 4)}]
        for lbl, val in sorted(res.reader.unrelated_scores.items(), key=lambda kv: kv[1], reverse=True):
            rows_unrel.append({"channel": lbl, "score": round(float(val), 4)})
        _render_html_table([("channel", "Thoracic Channel"), ("score", "Raw Score")], rows_unrel)
        st.caption(
            f"Checkpoint: `{res.reader.checkpoint_id}` · Preprocessing: `{res.reader.preprocessing_id}` · "
            f"Threshold ID: `{res.reader.threshold_id}`"
        )

if res.judge_outcome is not None and res.judge_outcome.primary_target_edit is not None and res.display_image is not None:
    edit = res.judge_outcome.primary_target_edit
    st.markdown(f"### Counterfactual Perturbation · `{SYNTHETIC_COUNTERFACTUAL_LABEL}`")
    labelled_edit = render_labelled(edit.edited_image, scale=1 if res.display_image.shape[0] >= 512 else 2)
    col_a, col_b = st.columns(2)
    with col_a:
        _viewport_header("ORIGINAL RADIOGRAPH", f"{int(res.display_image.shape[0])}PX")
        st.image(res.display_image.astype("uint8"), width="stretch", clamp=True)
    with col_b:
        _viewport_header(f"COUNTERFACTUAL ({edit.fill_method.upper()})", "OUTSIDE MASK EXACT")
        st.image(labelled_edit, width="stretch")

    if (ROOT / "models" / "UNET_CHECKPOINT.json").is_file() and res.locator is not None:
        with st.expander("Optional Gated-Convolution U-Net Counterfactual"):
            try:
                unet = UNetEraser.load(ROOT / "models")
                unet_edit = unet.erase(res.display_image, res.locator.target_mask, seed=0)
                unet_rc, _, _ = reader.score(unet_edit.edited_image)
                unet_drop = float(res.reader.score_raw - unet_rc.score_raw) if res.reader is not None else 0.0
                st.image(render_labelled(unet_edit.edited_image, scale=1), width=420)
                st.caption(
                    f"Gated U-Net drop: {unet_drop:.4f} · boundary discontinuity: {unet_edit.integrity_report.boundary_discontinuity:.4f}"
                )
            except EraseError as exc:
                st.caption(f"Gated U-Net unavailable ({exc.reason_code})")

    jc = res.judge_outcome.contract
    te = dict(jc.target_effects)
    ce = dict(jc.control_effects)
    if "score_drops" in ce:
        drops = ce["score_drops"]
        st.markdown("### Target Perturbation vs. Anatomically Matched Controls")
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
            color="#7FA88B",
        )

if res.explanation is not None:
    st.markdown("### Rule-Verified Grounded Explanation")
    _render_explanation_card([(s.text, s.evidence_ids) for s in res.explanation.sentences])
    if res.explanation.unverified_other_claims:
        st.caption("Mentioned by claimant model, not verified: " + "; ".join(res.explanation.unverified_other_claims))

st.markdown("### Evidence-Grounded Chat (5-Layer Guardrailed)")
chat_q = st.text_input("Query this scan's verdict, score drop, matched controls, or VeriRay protocol definitions:")
if chat_q:
    ev_rec = res.judge_outcome.evidence_record if res.judge_outcome is not None else None
    chat_resp = answer_chat_question(chat_q, ev_rec)
    _callout(html.escape(chat_resp.answer), tone="sage" if chat_resp.allowed else "amber")

with st.expander("Audit Trail — Structured EvidenceRecord & RunManifest"):
    if res.judge_outcome is not None:
        st.json(dict(res.judge_outcome.evidence_record.items))
    if res.manifest is not None:
        st.caption(
            f"Run ID: `{res.manifest.run_id}` · offline_only={res.manifest.offline_only} · "
            f"latencies (ms): { {k: round(v, 1) for k, v in res.latencies_ms.items()} }"
        )
