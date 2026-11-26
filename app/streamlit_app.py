"""VeriRay local verification app: Gate -> Reader -> Locator -> Eraser + Matched Controls -> Judge.

Runs 100% offline on this machine; uploads stay in memory (clear-session button removes state).
Run from the repository root:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veriray.brain_mri import BrainMRIError, BrainMRIReader  # noqa: E402
from veriray.config import configs_from_yaml  # noqa: E402
from veriray.contracts import SYNTHETIC_COUNTERFACTUAL_LABEL, EvidenceRecord, Verdict  # noqa: E402
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

st.set_page_config(page_title="VeriRay (research prototype)", layout="wide")
st.title("VeriRay — Local Verification Judge")
st.warning(BEHAVIOUR_NOTICE)
st.caption("Finding: pneumothorax. Runs entirely offline on this machine; images are held in memory only.")

if st.button("Clear session"):
    st.session_state.clear()
    st.rerun()

mode = st.radio(
    "Mode",
    [
        "Live upload queue (Chest X-ray)",
        "Brain MRI (4-class Tumor + Grad-CAM++ + Counterfactual)",
        "Precomputed 3-minute demo cases (honestly labelled)",
        "Evaluation & Evidence Pack (Locked Test + Ablations)",
    ],
    horizontal=True,
)

if mode.startswith("Brain MRI"):
    try:
        bm_reader = BrainMRIReader.load(ROOT / "models")
    except BrainMRIError as exc:
        st.error(f"INDETERMINATE — {exc.reason_code}: {exc}")
        st.stop()
    mri_file = st.file_uploader("Upload a public, de-identified Brain MRI slice (PNG/JPG)", type=["png", "jpg", "jpeg", "bmp"])
    if mri_file is None:
        st.info("Upload a Brain MRI image to inspect model scores, Grad-CAM++, and counterfactual drop.")
        st.stop()
    from PIL import Image
    import io

    arr = np.asarray(Image.open(io.BytesIO(mri_file.getvalue())).convert("L"), dtype=np.float32)
    out = bm_reader.predict_and_cam(arr)
    st.metric("Top model class (uncalibrated model score, not a diagnosis)", f"{out['predicted_class']} ({out['scores'][out['predicted_class']]:.4f})")
    st.json({k: round(v, 4) for k, v in out["scores"].items()})
    c1, c2, c3 = st.columns(3)
    c1.image(out["display_image"].astype("uint8"), caption="Brain MRI (256px)", clamp=True)
    c2.image(overlay_heatmap(out["display_image"], out["heatmap"]), caption="Grad-CAM++ candidate region")
    c3.image((out["target_mask"] * 255).astype("uint8"), caption="Candidate mask")
    if out["target_mask"].any() and not out["target_mask"].all():
        cf = erase(out["display_image"], out["target_mask"], "telea", seed=0)
        cf_out = bm_reader.predict_and_cam(cf.edited_image)
        pred = out["predicted_class"]
        target_drop = float(out["scores"][pred] - cf_out["scores"][pred])
        st.subheader(f"Counterfactual ({SYNTHETIC_COUNTERFACTUAL_LABEL})")
        st.image(render_labelled(cf.edited_image, scale=2), caption=f"Telea counterfactual — {SYNTHETIC_COUNTERFACTUAL_LABEL}")
        st.write(f"Model score change on `{pred}` after masking candidate region: `{out['scores'][pred]:.4f}` → `{cf_out['scores'][pred]:.4f}` (drop: `{target_drop:.4f}`)")
        ctrl_res = generate_matched_controls(out["target_mask"], ControlConfig(k=9, min_controls=3, seed=20261126))
        if ctrl_res.valid:
            c_drops = []
            for ci, cmask in enumerate(ctrl_res.masks):
                ccf = erase(out["display_image"], cmask, "telea", seed=ci + 1)
                ccf_out = bm_reader.predict_and_cam(ccf.edited_image)
                c_drops.append(float(out["scores"][pred] - ccf_out["scores"][pred]))
            c_med = float(np.median(c_drops))
            emp_rank = 1 + int(np.sum(np.asarray(c_drops) >= target_drop))
            st.write(
                f"Matched controls (K={len(c_drops)}): median drop **{c_med:.4f}** · "
                f"Empirical rank: **{emp_rank}** / {len(c_drops) + 1} (empirical rank, not a p-value)"
            )
    st.caption(f"Checkpoint: `{out['checkpoint_id']}` · {BEHAVIOUR_NOTICE}")
    st.stop()

if mode.startswith("Precomputed"):
    demo_dir = ROOT / "artifacts" / "demo_cases"
    idx_path = demo_dir / "index.json"
    if not idx_path.is_file():
        st.info("Precomputed demo pack not generated yet. Run: `python scripts/generate_demo_pack.py`")
        st.stop()
    index = json.loads(idx_path.read_text(encoding="utf-8"))
    choice = st.selectbox("Select precomputed case", range(len(index)), format_func=lambda i: index[i]["title"])
    cdir = demo_dir / index[choice]["case_id"]
    bundle = json.loads((cdir / "bundle.json").read_text(encoding="utf-8"))
    st.info(bundle["precomputed_notice"])
    v_enum = Verdict(bundle["verdict"])
    {
        Verdict.MODEL_SUPPORTED: st.success,
        Verdict.NOT_SUPPORTED: st.warning,
        Verdict.SHORTCUT_RISK: st.error,
        Verdict.INDETERMINATE: st.error,
    }[v_enum](f"**{bundle['verdict']}** — Reason codes: `{', '.join(bundle['reason_codes'])}`")
    c1, c2, c3 = st.columns(3)
    c1.image(str(cdir / "original.png"), caption="Original (224px)")
    c2.image(str(cdir / "overlay.png"), caption="Grad-CAM++ heatmap overlay")
    c3.image(str(cdir / "mask.png"), caption="Candidate mask")
    if bundle["has_counterfactual"]:
        st.subheader(f"Counterfactual ({bundle['synthetic_label']})")
        st.image(str(cdir / "counterfactual_labelled.png"), caption=bundle["synthetic_label"])
    st.subheader("Verified explanation")
    for s in bundle["verified_explanation"]:
        st.write(f"- {s['text']} `[{', '.join(s['evidence_ids'])}]`")
    st.subheader("Evidence record")
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
        (st.write if ans.allowed else st.warning)(ans.answer)
    st.info(BEHAVIOUR_NOTICE)
    st.stop()


if mode.startswith("Evaluation"):
    pack_path = ROOT / "artifacts" / "eval_test_locked" / "test_evidence_pack.json"
    if not pack_path.is_file():
        st.info("Locked test evidence pack not found. Run `python scripts/run_locked_test_and_exp5.py` first.")
        st.stop()
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    st.caption(pack.get("note", "NIH ChestX-ray14 locked test evidence pack"))

    st.subheader("Reader metrics & Platt calibration (fitted on val only)")
    rm = pack.get("reader_metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUROC", f"{rm.get('auroc', 0.0):.4f}")
    c2.metric("AUPRC", f"{rm.get('auprc', 0.0):.4f}")
    c3.metric("Brier score", f"{rm.get('brier', 0.0):.4f}")
    c4.metric("ECE", f"{rm.get('ece', 0.0):.4f}")

    cal_path = ROOT / "models" / "platt_calibrator.json"
    if cal_path.is_file():
        cal_d = json.loads(cal_path.read_text(encoding="utf-8"))
        st.caption(
            f"Platt calibrator (`{cal_d.get('calibrator_id')}`): fitted on `{cal_d.get('fitted_split')}` "
            f"(n={cal_d.get('num_samples')}), coef={cal_d.get('coef', 0.0):.4f}, intercept={cal_d.get('intercept', 0.0):.4f}"
        )

    st.subheader("Selective risk-coverage: Reader-only vs. Full Judge")
    rc_cmp = pack.get("risk_coverage_comparison", {}).get("operating_point", {})
    fj = rc_cmp.get("full_judge", {})
    ro = rc_cmp.get("reader_at_same_coverage", {})
    col_fj, col_ro = st.columns(2)
    col_fj.metric("Full Judge precision", f"{fj.get('precision', 0.0):.2%}", f"coverage {fj.get('coverage', 0.0):.2%} ({fj.get('accepted_count', 0)}/{fj.get('n', 0)})")
    col_ro.metric("Reader-only @ matched coverage", f"{ro.get('precision', 0.0):.2%}", f"delta {rc_cmp.get('precision_delta_judge_minus_reader', 0.0):+.4f}")

    rc_img = ROOT / "artifacts" / "eval_test_locked" / "risk_coverage.png"
    if rc_img.is_file():
        st.image(str(rc_img), caption="Locked test risk-coverage curve")

    st.subheader("Ablations A–F & fill-method sensitivity")
    abl = pack.get("ablations", {})
    if abl.get("configurations"):
        st.dataframe(abl["configurations"], use_container_width=True)
    if pack.get("fill_sensitivity"):
        st.json(pack["fill_sensitivity"])

    c_loc, c_exp5 = st.columns(2)
    with c_loc:
        st.subheader("Localization (reporting-half boxes)")
        st.json(pack.get("localization_reporting_half", {}))
    with c_exp5:
        st.subheader("Experiment 5: MedGemma Claim Verifier")
        st.json(pack.get("experiment_5_claim_verifier", {}))

    c_sex, c_age = st.columns(2)
    with c_sex:
        st.subheader("Subgroup metrics (Sex)")
        st.json(pack.get("subgroups_sex", {}))
    with c_age:
        st.subheader("Subgroup metrics (Age band)")
        st.json(pack.get("subgroups_age_band", {}))

    brain_path = ROOT / "models" / "BRAIN_MRI_CHECKPOINT.json"
    if brain_path.is_file():
        bm = json.loads(brain_path.read_text(encoding="utf-8"))
        st.subheader(f"Brain MRI 4-class benchmark (`{bm.get('filename')}`)")
        st.metric("Test accuracy (n=1600)", f"{bm.get('best_test_accuracy', 0.0):.2%}")

    st.subheader("Verdict counts & latencies (ms)")
    st.json({"verdict_counts": pack.get("verdict_counts", {}), "latencies": pack.get("latencies", {})})
    st.info(BEHAVIOUR_NOTICE)
    st.stop()


@st.cache_resource(show_spinner="Loading Reader from models/ ...")
def _load_reader():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return Reader.load_preferred(ROOT / "models", device=device), None
    except ReaderError as exc:
        return None, exc


reader, load_err = _load_reader()
if load_err is not None:
    st.error(f"INDETERMINATE — {load_err.reason_code}: {load_err.message}")
    st.stop()

use_medgemma = st.checkbox(
    "Enable MedGemma 4B Claimant & Grounded Explainer (4-bit local GPU, ~3.4 GB VRAM)",
    value=False,
)

uploads = st.file_uploader(
    "Public, de-identified chest X-rays (PNG/JPEG/BMP/TIFF — single image or live case queue)",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
)
if not uploads:
    st.info("Load one or more public, de-identified images to begin.")
    st.stop()

frozen_cfg_path = ROOT / "configs" / "frozen.yaml"
if frozen_cfg_path.is_file():
    loc_cfg, judge_cfg, _ = configs_from_yaml(frozen_cfg_path)
else:
    loc_cfg, judge_cfg = LocatorConfig(), JudgeConfig()

queue_results = []
status_bar = st.progress(0, text="Processing case queue...")
for idx, upl in enumerate(uploads):
    status_bar.progress((idx + 1) / len(uploads), text=f"Processing case {idx + 1}/{len(uploads)}: {upl.name}")

    # ── Modality mismatch guard ──────────────────────────────────────────────
    # Detect when a Brain MRI is uploaded to the Chest X-ray queue and show
    # a clear redirect instead of a confusing INDETERMINATE verdict.
    try:
        from PIL import Image as _PILImage
        import io as _io
        _raw = np.asarray(
            _PILImage.open(_io.BytesIO(upl.getvalue())).convert("L"),
            dtype=np.float32,
        )
        _guess = guess_modality(_raw)
        if _guess.is_brain_mri_suspected and _guess.confidence >= 0.60:
            st.warning(
                f"🧠 **Wrong mode detected for `{upl.name}`**  \n"
                "This image looks like a **Brain MRI** (not a chest X-ray).  \n"
                "👉 Switch to **Brain MRI (4-class Tumor + Grad-CAM++ + Counterfactual)** "
                "mode above and re-upload for correct analysis.  \n"
                f"*(Modality heuristic confidence: {_guess.confidence:.0%} — "
                "not a diagnosis; switch modes and let the Brain MRI reader decide.)*"
            )
            # Still show the signals in an expander for transparency
            with st.expander("Modality detection signals (debug)"):
                st.json({k: round(v, 4) for k, v in _guess.signals.items()})
                st.caption(_guess.reason)
            continue  # skip running the Chest X-ray pipeline on this file
    except Exception:  # noqa: BLE001 — modality check must never crash the queue
        pass
    # ────────────────────────────────────────────────────────────────────────

    r_item = run_verification(
        upl.getvalue(),
        filename_hint=upl.name,
        reader=reader,
        models_dir=ROOT / "models",
        manifest_dir=ROOT / "artifacts" / "tmp" / "runs",
        locator_cfg=loc_cfg,
        judge_cfg=judge_cfg,
    )
    queue_results.append((upl.name, r_item))

if not queue_results:
    st.info(
        "⬆️ All uploaded file(s) were redirected to the Brain MRI mode.  \n"
        "Switch the **Mode** selector above to **Brain MRI** and re-upload."
    )
    st.stop()

if len(queue_results) > 1:
    st.subheader(f"Case queue summary ({len(queue_results)} cases)")
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
        use_container_width=True,
    )
    selected_idx = st.selectbox(
        "Inspect case",
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
            st.subheader("MedGemma 4B Claimant (local 4-bit GPU)")
            for c in vlm_claims:
                iou_txt = ""
                if c.box_xyxy is not None and res.locator is not None:
                    iou_txt = f" · IoU with Grad-CAM++ mask: {box_mask_iou(res.locator.target_mask, c.box_xyxy):.3f}"
                tag = "verified target finding" if c.is_verifiable_finding else "mentioned by the model, not verified"
                s_finding = sanitize_for_prompt(c.finding)
                s_status = sanitize_for_prompt(c.status)
                s_region = sanitize_for_prompt(c.region_description)
                st.write(f"- **{s_finding}** (`{s_status}`, {s_region}) — *{tag}*{iou_txt}")
            if vlm_expl_rep is not None:
                st.caption(
                    f"MedGemma grounded explainer: {len(vlm_expl_rep.verified_sentences)} verified sentences, "
                    f"{vlm_expl_rep.dropped_count} dropped by Claim Verifier, peak VRAM {vlm_peak:.0f} MB"
                )
                for s in vlm_expl_rep.verified_sentences:
                    st.write(f"  - {s.text} `[{', '.join(s.evidence_ids)}]`")
        except VLMError as exc:
            st.warning(f"MedGemma unavailable ({exc.reason_code}): {exc}")

verdict_banner = {
    Verdict.MODEL_SUPPORTED: st.success,
    Verdict.NOT_SUPPORTED: st.warning,
    Verdict.SHORTCUT_RISK: st.error,
    Verdict.INDETERMINATE: st.error,
}[res.verdict]
user_text = res.output.user_facing_text if res.output is not None else "The test could not produce a reliable result."
verdict_banner(f"**{res.verdict.value}** — {user_text}  \nReason codes: `{', '.join(res.reason_codes)}`")

st.subheader("Input gate")
if res.gate is not None:
    st.write(f"Status: **{res.gate.status.value}** — reasons: {', '.join(res.gate.reason_codes)}")
    flagged = [k for k, v in res.gate.quality_flags.items() if v]
    if flagged:
        st.write("Quality flags: " + ", ".join(flagged))

if res.reader is not None and res.locator is not None and res.display_image is not None:
    c1, c2, c3 = st.columns(3)
    c1.image(res.display_image.astype("uint8"), caption="Original (224px model-space view)", clamp=True)
    c2.image(
        overlay_heatmap(res.display_image, res.locator.heatmap),
        caption="Grad-CAM++ heatmap overlay (candidate region only)",
    )
    c3.image((res.locator.target_mask * 255).astype("uint8"), caption="Candidate mask")

    st.subheader("Reader")
    if res.reader.score_calibrated_optional is not None:
        m1, m2 = st.columns(2)
        m1.metric(SCORE_LABEL, f"{res.reader.score_raw:.4f}")
        m2.metric("Platt-calibrated model score (fitted on val; not a diagnosis)", f"{res.reader.score_calibrated_optional:.4f}")
    else:
        st.metric(SCORE_LABEL, f"{res.reader.score_raw:.4f}")
    target_thresh = judge_cfg.min_reader_score if judge_cfg.min_reader_score > 0 else 0.5
    screening = summarize_reader_findings(res.reader, target_threshold=target_thresh, other_threshold=0.5)
    if screening.all_scores_low:
        st.info(
            "Reader screening summary: No monitored thoracic finding exceeded the model-score threshold "
            "(all Reader outputs are low). This describes uncalibrated model scores only, not a clinical assessment."
        )
    elif not screening.target_flagged and screening.flagged_other_findings:
        flagged_txt = ", ".join(f"{lbl} ({val:.3f})" for lbl, val in screening.flagged_other_findings)
        st.warning(
            f"Pneumothorax score is below threshold, while other unverified Reader label(s) scored >= 0.50: {flagged_txt}"
        )
    with st.expander("All monitored Reader scores (uncalibrated)"):
        all_scores = {"Pneumothorax": round(res.reader.score_raw, 4)}
        for lbl, val in sorted(res.reader.unrelated_scores.items(), key=lambda kv: kv[1], reverse=True):
            all_scores[lbl] = round(float(val), 4)
        st.json(all_scores)
    st.write(
        f"Checkpoint ID: `{res.reader.checkpoint_id}`  \n"
        f"Preprocessing ID: `{res.reader.preprocessing_id}`  \n"
        f"Threshold ID: `{res.reader.threshold_id}` (uncalibrated model score)"
    )

if res.judge_outcome is not None and res.judge_outcome.primary_target_edit is not None and res.display_image is not None:
    edit = res.judge_outcome.primary_target_edit
    st.subheader(f"Counterfactual ({SYNTHETIC_COUNTERFACTUAL_LABEL})")
    view_choice = st.radio(
        "Compare view",
        ["Original vs. Synthetic Counterfactual", "Side-by-side"],
        horizontal=True,
    )
    labelled_edit = render_labelled(edit.edited_image, scale=2)
    if view_choice == "Side-by-side":
        col_a, col_b = st.columns(2)
        col_a.image(res.display_image.astype("uint8"), caption="Original (224px)", clamp=True)
        col_b.image(labelled_edit, caption=f"{edit.fill_method} — {SYNTHETIC_COUNTERFACTUAL_LABEL}")
    else:
        show_cf = st.toggle("Show synthetic counterfactual", value=True)
        if show_cf:
            st.image(labelled_edit, caption=f"{edit.fill_method} — {SYNTHETIC_COUNTERFACTUAL_LABEL}")
        else:
            st.image(res.display_image.astype("uint8"), caption="Original (224px)", clamp=True)

    if (ROOT / "models" / "UNET_CHECKPOINT.json").is_file() and res.locator is not None:
        with st.expander("Gated-Convolution U-Net counterfactual comparison (optional)"):
            try:
                unet = UNetEraser.load(ROOT / "models")
                unet_edit = unet.erase(res.display_image, res.locator.target_mask, seed=0)
                unet_rc, _, _ = reader.score(unet_edit.edited_image)
                unet_drop = float(res.reader.score_raw - unet_rc.score_raw) if res.reader is not None else 0.0
                st.image(render_labelled(unet_edit.edited_image, scale=2), caption=f"gated_unet — {SYNTHETIC_COUNTERFACTUAL_LABEL}")
                st.caption(f"Gated U-Net score drop: {unet_drop:.4f} | boundary discontinuity: {unet_edit.integrity_report.boundary_discontinuity:.4f}")
            except EraseError as exc:
                st.caption(f"Gated U-Net unavailable ({exc.reason_code})")

    st.subheader("Verification Judge evidence")
    jc = res.judge_outcome.contract
    te = dict(jc.target_effects)
    ce = dict(jc.control_effects)
    if "score_drops" in ce:
        drops = ce["score_drops"]
        st.write(
            f"Target score drop (`{te.get('primary_fill_method')}`): **{te.get('primary_score_drop', 0.0):.4f}** · "
            f"Matched controls (K={ce.get('num_controls')}): median drop **{ce.get('median_score_drop', 0.0):.4f}** · "
            f"Empirical rank: **{te.get('empirical_rank')}** / {int(ce.get('num_controls', 0)) + 1} "
            "(empirical rank, not a p-value)"
        )
        st.bar_chart({
            "Target": [te.get("primary_score_drop", 0.0)],
            "Control median": [ce.get("median_score_drop", 0.0)],
            "Control max": [float(np.max(drops)) if drops else 0.0],
        })
    col_g, col_s = st.columns(2)
    col_g.write("Gate results:")
    col_g.json(dict(jc.gate_results))
    col_s.write("Evidence record:")
    col_s.json(dict(res.judge_outcome.evidence_record.items))

if res.explanation is not None:
    st.subheader("Verified explanation")
    for sent in res.explanation.sentences:
        st.write(f"- {sent.text} `[{', '.join(sent.evidence_ids)}]`")
    if res.explanation.unverified_other_claims:
        st.caption("Mentioned by the model, not verified: " + "; ".join(res.explanation.unverified_other_claims))

st.subheader("Evidence-grounded chat")
chat_q = st.text_input("Ask about this run's verdict, score, controls, or general definitions:")
if chat_q:
    ev_rec = res.judge_outcome.evidence_record if res.judge_outcome is not None else None
    chat_resp = answer_chat_question(chat_q, ev_rec)
    if chat_resp.allowed:
        st.write(chat_resp.answer)
    else:
        st.warning(chat_resp.answer)

st.subheader("System")
if res.manifest is not None:
    st.write(
        f"Run ID: `{res.manifest.run_id}` · offline_only={res.manifest.offline_only} · "
        f"latencies (ms): { {k: round(v) for k, v in res.latencies_ms.items()} }"
    )
st.info(BEHAVIOUR_NOTICE)
