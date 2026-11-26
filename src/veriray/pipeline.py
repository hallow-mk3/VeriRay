"""VeriRay pipelines: Step 2 (gate -> reader -> locator) and Step 3 (full verification Judge)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from veriray.anatomy import LungSegmenter
from veriray.contracts import (
    RESEARCH_DISCLAIMER,
    VERDICT_USER_LANGUAGE,
    InputContract,
    InputGateContract,
    LocatorContract,
    OutputContract,
    ReaderContract,
    Verdict,
)
from veriray.explainer import ExplainerOutput, explain_verified
from veriray.gate import GateConfig, run_input_gate
from veriray.judge import JudgeConfig, JudgeOutcome, run_judge
from veriray.locator import LOCATOR_ID, LocatorConfig, LocatorError, locate
from veriray.manifest import RunManifest, create_run_manifest, write_run_manifest
from veriray.reader import PREPROCESSING_ID, Reader, ReaderError

BEHAVIOUR_NOTICE = (
    "Score and heatmap describe model behaviour, not a diagnosis. "
    "Research prototype — not for clinical use."
)
SCORE_LABEL = "Model score (uncalibrated Pneumothorax output of the Reader)"
SEED = 0


@dataclass
class Step2Result:
    verdict: Verdict | None
    reason_codes: tuple[str, ...]
    failed_stage: str | None = None
    detail: str = ""
    gate: InputGateContract | None = None
    reader: ReaderContract | None = None
    locator: LocatorContract | None = None
    display_image: np.ndarray | None = None
    manifest: RunManifest | None = None
    manifest_path: Path | None = None
    latencies_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class VerificationResult:
    output: OutputContract | None  # None only if Reader failed to load (no checkpoint hash for manifest)
    verdict: Verdict
    reason_codes: tuple[str, ...]
    failed_stage: str | None = None
    detail: str = ""
    gate: InputGateContract | None = None
    reader: ReaderContract | None = None
    locator: LocatorContract | None = None
    judge_outcome: JudgeOutcome | None = None
    explanation: ExplainerOutput | None = None
    display_image: np.ndarray | None = None
    manifest: RunManifest | None = None
    manifest_path: Path | None = None
    latencies_ms: dict[str, float] = field(default_factory=dict)


def _fail(stage: str, codes: list[str], detail: str, **kw: Any) -> Step2Result:
    return Step2Result(Verdict.INDETERMINATE, tuple(codes), stage, detail, **kw)


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _run_frontend(
    source: Any,
    *,
    models_dir: Path | str,
    filename_hint: str | None,
    raw_metadata: Mapping[str, Any] | None,
    reader: Reader | None,
    gate_cfg: GateConfig,
    locator_cfg: LocatorConfig,
    device: str | None = None,
) -> tuple[
    Reader | None,
    InputContract | None,
    InputGateContract | None,
    ReaderContract | None,
    LocatorContract | None,
    np.ndarray | None,
    dict[str, float],
    tuple[str, tuple[str, ...], str] | None,
]:
    """Shared Gate -> Reader -> Locator frontend; returns failure tuple (stage, codes, detail) on error."""
    lat: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        if reader is None:
            reader = Reader.load(models_dir, device=device or _default_device())
    except ReaderError as exc:
        return None, None, None, None, None, None, lat, ("reader_load", (exc.reason_code,), exc.message)
    except Exception as exc:  # noqa: BLE001
        return None, None, None, None, None, None, lat, (
            "reader_load", ("READER_UNEXPECTED_ERROR",), f"{type(exc).__name__}: {exc}",
        )
    lat["reader_load"] = (time.perf_counter() - t0) * 1000

    t = time.perf_counter()
    inp, gate = run_input_gate(
        source,
        model_id=reader.checkpoint_id,
        preprocessing_id=PREPROCESSING_ID,
        raw_metadata=raw_metadata,
        filename_hint=filename_hint,
        config=gate_cfg,
    )
    lat["gate"] = (time.perf_counter() - t) * 1000
    if inp is None or not gate.accepted:
        return reader, None, gate, None, None, None, lat, (
            "input_gate", (f"GATE_{gate.status.value}", *gate.reason_codes), "",
        )

    try:
        t = time.perf_counter()
        reader_out, x, display = reader.score(inp.image_tensor)
        lat["reader"] = (time.perf_counter() - t) * 1000
    except ReaderError as exc:
        return reader, inp, gate, None, None, None, lat, ("reader", (exc.reason_code,), exc.message)
    except Exception as exc:  # noqa: BLE001
        return reader, inp, gate, None, None, None, lat, (
            "reader", ("READER_UNEXPECTED_ERROR",), f"{type(exc).__name__}: {exc}",
        )

    try:
        t = time.perf_counter()
        loc = locate(reader, x, locator_cfg)
        lat["locator"] = (time.perf_counter() - t) * 1000
    except LocatorError as exc:
        return reader, inp, gate, reader_out, None, display, lat, ("locator", (exc.reason_code,), str(exc))
    except Exception as exc:  # noqa: BLE001
        return reader, inp, gate, reader_out, None, display, lat, (
            "locator", ("LOCATOR_UNEXPECTED_ERROR",), f"{type(exc).__name__}: {exc}",
        )

    return reader, inp, gate, reader_out, loc, display, lat, None


def run_step2(
    source: Any,
    *,
    models_dir: Path | str = "models",
    manifest_dir: Path | str | None = "artifacts/tmp/runs",
    filename_hint: str | None = None,
    raw_metadata: Mapping[str, Any] | None = None,
    reader: Reader | None = None,
    gate_cfg: GateConfig | None = None,
    locator_cfg: LocatorConfig | None = None,
) -> Step2Result:
    gate_cfg, locator_cfg = gate_cfg or GateConfig(), locator_cfg or LocatorConfig()
    reader, inp, gate, reader_out, loc, display, lat, err = _run_frontend(
        source,
        models_dir=models_dir,
        filename_hint=filename_hint,
        raw_metadata=raw_metadata,
        reader=reader,
        gate_cfg=gate_cfg,
        locator_cfg=locator_cfg,
    )
    if err is not None:
        stage, codes, detail = err
        return _fail(stage, list(codes), detail, gate=gate, reader=reader_out, latencies_ms=lat)

    assert reader is not None and inp is not None and gate is not None and reader_out is not None and loc is not None

    reasons: tuple[str, ...] = ()
    verdict = None
    failed = None
    if loc.is_weak_or_unstable:
        verdict = Verdict.INDETERMINATE
        failed = "locator_quality"
        reasons = tuple(
            "LOCALIZATION_" + k.upper()
            for k in ("empty", "diffuse", "implausibly_large", "near_full_image", "anatomically_implausible", "unstable")
            if loc.localization_flags.get(k, False)
        ) or ("LOCALIZATION_EMPTY",)

    manifest = None
    manifest_path = None
    try:
        record = reader.record
        manifest = create_run_manifest(
            run_config={
                "step": 2,
                "gate": asdict(gate_cfg),
                "locator_id": LOCATOR_ID,
                "locator": asdict(locator_cfg),
                "preprocessing_id": PREPROCESSING_ID,
                "checkpoint_id": reader.checkpoint_id,
                "input_image_sha256_of_gate_output": _sha(inp.image_tensor),
                "gate_reason_codes": list(gate.reason_codes),
                "locator_flags": dict(loc.localization_flags),
                "model_score": reader_out.score_raw,
            },
            model_id=reader.checkpoint_id,
            checkpoint_hashes={record.weights_id: record.sha256},
            preprocessing_id=PREPROCESSING_ID,
            dataset_split_id="none-single-image-step2",
            random_seeds={"torch": SEED, "numpy": SEED},
            stage_latencies_ms=lat,
        )
        if manifest_dir is not None:
            manifest_path = write_run_manifest(manifest, Path(manifest_dir) / manifest.run_id / "run_manifest.json")
    except Exception as exc:  # noqa: BLE001 - a manifest failure is a failure (fail closed)
        return _fail("manifest", ["MANIFEST_WRITE_FAILED"], f"{type(exc).__name__}: {exc}", gate=gate,
                     reader=reader_out, locator=loc, display_image=display, latencies_ms=lat)

    return Step2Result(verdict, reasons, failed, "", gate, reader_out, loc, display, manifest, manifest_path, lat)


def run_verification(
    source: Any,
    *,
    models_dir: Path | str = "models",
    manifest_dir: Path | str | None = "artifacts/tmp/runs",
    filename_hint: str | None = None,
    raw_metadata: Mapping[str, Any] | None = None,
    reader: Reader | None = None,
    gate_cfg: GateConfig | None = None,
    locator_cfg: LocatorConfig | None = None,
    judge_cfg: JudgeConfig | None = None,
    lung_mask: np.ndarray | None = None,
    audited_shortcut_cue: bool = False,
    dataset_split_id: str = "none-single-image",
) -> VerificationResult:
    """Full offline verification pipeline: Gate -> Reader -> Locator -> Eraser + Controls -> Judge."""
    gate_cfg = gate_cfg or GateConfig()
    locator_cfg = locator_cfg or LocatorConfig()
    judge_cfg = judge_cfg or JudgeConfig()

    reader, inp, gate, reader_out, loc, display, lat, err = _run_frontend(
        source,
        models_dir=models_dir,
        filename_hint=filename_hint,
        raw_metadata=raw_metadata,
        reader=reader,
        gate_cfg=gate_cfg,
        locator_cfg=locator_cfg,
    )
    if reader is None:
        stage, codes, detail = err or ("reader_load", ("READER_CHECKPOINT_MISSING",), "")
        return VerificationResult(None, Verdict.INDETERMINATE, codes, stage, detail, latencies_ms=lat)

    record = reader.record
    judge_outcome: JudgeOutcome | None = None
    failed_stage: str | None = None
    detail = ""
    verdict = Verdict.INDETERMINATE
    reasons: tuple[str, ...]

    if err is not None:
        failed_stage, reasons, detail = err
    else:
        assert inp is not None and gate is not None and reader_out is not None and loc is not None and display is not None
        t = time.perf_counter()
        try:
            # Automatically compute lung mask if optional offline PSPNet weights are present in models/
            effective_lung_mask = lung_mask
            if effective_lung_mask is None:
                seg = LungSegmenter.load_optional(models_dir, device=str(reader.device))
                if seg is not None:
                    effective_lung_mask = seg.segment_lungs(display)
            # Score the 224x224 model-space view as baseline so counterfactual drops compare identical frames.
            baseline_224, _, _ = reader.score(display)
            judge_outcome = run_judge(
                reader,
                display,
                baseline_224,
                loc,
                cfg=judge_cfg,
                locator_cfg=locator_cfg,
                lung_mask=effective_lung_mask,
                audited_shortcut_cue=audited_shortcut_cue,
            )
            lat["judge"] = (time.perf_counter() - t) * 1000
            verdict = judge_outcome.contract.verdict
            reasons = judge_outcome.contract.reason_codes
            if verdict is Verdict.INDETERMINATE:
                failed_stage = "judge"
        except ReaderError as exc:
            lat["judge"] = (time.perf_counter() - t) * 1000
            verdict = Verdict.INDETERMINATE
            reasons = (exc.reason_code,)
            failed_stage = "judge"
            detail = exc.message
        except Exception as exc:  # noqa: BLE001 - fail closed, never raise
            lat["judge"] = (time.perf_counter() - t) * 1000
            verdict = Verdict.INDETERMINATE
            reasons = ("JUDGE_UNEXPECTED_ERROR",)
            failed_stage = "judge"
            detail = f"{type(exc).__name__}: {exc}"

    manifest: RunManifest | None = None
    manifest_path: Path | None = None
    try:
        manifest = create_run_manifest(
            run_config={
                "step": 3,
                "gate": asdict(gate_cfg),
                "locator_id": LOCATOR_ID,
                "locator": asdict(locator_cfg),
                "judge": asdict(judge_cfg),
                "preprocessing_id": PREPROCESSING_ID,
                "checkpoint_id": reader.checkpoint_id,
                "input_image_sha256": _sha(inp.image_tensor) if inp is not None else None,
                "verdict": verdict.value,
                "reason_codes": list(reasons),
            },
            model_id=reader.checkpoint_id,
            checkpoint_hashes={record.weights_id: record.sha256},
            preprocessing_id=PREPROCESSING_ID,
            dataset_split_id=dataset_split_id,
            random_seeds={"torch": SEED, "numpy": SEED, "controls": judge_cfg.seed},
            stage_latencies_ms=lat,
        )
        if manifest_dir is not None:
            manifest_path = write_run_manifest(manifest, Path(manifest_dir) / manifest.run_id / "run_manifest.json")
    except Exception as exc:  # noqa: BLE001 - manifest failure fails closed
        return VerificationResult(
            None, Verdict.INDETERMINATE, ("MANIFEST_WRITE_FAILED",), "manifest",
            f"{type(exc).__name__}: {exc}", gate=gate, reader=reader_out, locator=loc,
            judge_outcome=judge_outcome, display_image=display, latencies_ms=lat,
        )

    if err is not None or judge_outcome is None:
        out = OutputContract.fail_closed(reasons, manifest, failed_stage or "frontend")
        explanation = None
    else:
        explanation = explain_verified(judge_outcome.evidence_record)
        out = OutputContract(
            verdict=verdict,
            reason_codes=reasons,
            evidence_summary=dict(judge_outcome.evidence_record.items),
            disclaimer=RESEARCH_DISCLAIMER,
            run_manifest=manifest,
            user_facing_text=" ".join(s.text for s in explanation.sentences),
        )

    return VerificationResult(
        output=out,
        verdict=verdict,
        reason_codes=reasons,
        failed_stage=failed_stage,
        detail=detail,
        gate=gate,
        reader=reader_out,
        locator=loc,
        judge_outcome=judge_outcome,
        explanation=explanation,
        display_image=display,
        manifest=manifest,
        manifest_path=manifest_path,
        latencies_ms=lat,
    )


def overlay_heatmap(display_0_255: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """RGB uint8 overlay of a heatmap on the 224px model-space view (display only)."""
    import cv2

    base = np.clip(display_0_255, 0, 255).astype(np.uint8)
    color = cv2.cvtColor(cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    base_rgb = np.stack([base] * 3, axis=-1)
    return np.clip((1 - alpha) * base_rgb + alpha * color, 0, 255).astype(np.uint8)
