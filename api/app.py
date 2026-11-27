"""Local FastAPI service for VeriRay (100% offline, schema-validated, fail-closed).

Run locally (binds 127.0.0.1 only):
    uvicorn api.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI, File, Query, UploadFile

from api.schemas import (
    BatchVerifyResponse,
    ChatRequest,
    ChatResponseSchema,
    HealthResponse,
    VerifiedSentenceSchema,
    VerifyResponse,
)
from veriray.config import configs_from_yaml
from veriray.contracts import RESEARCH_DISCLAIMER, SUPPORTED_FINDINGS, VERDICT_USER_LANGUAGE, EvidenceRecord, Verdict
from veriray.judge import JudgeConfig
from veriray.locator import LocatorConfig
from veriray.pipeline import VerificationResult, run_verification
from veriray.reader import Reader, ReaderError
from veriray.safety import answer_chat_question, sanitize_for_prompt

ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_BATCH_FILES = 20


def _to_verify_response(sample_name: str, res: VerificationResult, finding_id: str = "pneumothorax") -> VerifyResponse:
    safe_name = sanitize_for_prompt(sample_name) or "unnamed"
    expl = []
    if res.explanation is not None:
        expl = [
            VerifiedSentenceSchema(text=s.text, evidence_ids=list(s.evidence_ids), verified=s.verified)
            for s in res.explanation.sentences
        ]
    ev_items: dict[str, Any] = {}
    gate_results: dict[str, bool] = {}
    if res.judge_outcome is not None:
        ev_items = dict(res.judge_outcome.evidence_record.items)
        gate_results = dict(res.judge_outcome.contract.gate_results)

    return VerifyResponse(
        sample_name=safe_name,
        finding=finding_id,  # type: ignore[arg-type]
        verdict=res.verdict.value,  # type: ignore[arg-type]
        reason_codes=list(res.reason_codes),
        user_facing_text=res.output.user_facing_text if res.output is not None else VERDICT_USER_LANGUAGE[Verdict.INDETERMINATE],
        disclaimer=res.output.disclaimer if res.output is not None else RESEARCH_DISCLAIMER,
        model_score_raw=res.reader.score_raw if res.reader is not None else None,
        checkpoint_id=res.reader.checkpoint_id if res.reader is not None else None,
        preprocessing_id=res.reader.preprocessing_id if res.reader is not None else None,
        gate_status=res.gate.status.value if res.gate is not None else None,
        gate_reasons=list(res.gate.reason_codes) if res.gate is not None else [],
        gate_results=gate_results,
        evidence_items=ev_items,
        verified_explanation=expl,
        run_id=res.manifest.run_id if res.manifest is not None else None,
        latencies_ms={k: round(v, 3) for k, v in res.latencies_ms.items()},
    )


def _fail_closed_response(sample_name: str, reason_code: str) -> VerifyResponse:
    return VerifyResponse(
        sample_name=sanitize_for_prompt(sample_name) or "unnamed",
        verdict="INDETERMINATE",
        reason_codes=[reason_code],
        user_facing_text=VERDICT_USER_LANGUAGE[Verdict.INDETERMINATE],
        disclaimer=RESEARCH_DISCLAIMER,
    )


def create_app(
    *,
    models_dir: Path | str = ROOT / "models",
    manifest_dir: Path | str | None = ROOT / "artifacts" / "tmp" / "runs",
    reader: Reader | None = None,
) -> FastAPI:
    """Create a local FastAPI app; fails closed to INDETERMINATE on any model/input/runtime fault."""
    app = FastAPI(title="VeriRay Local API", version="2.0.0", docs_url="/docs", redoc_url=None)
    state: dict[str, Any] = {
        "models_dir": Path(models_dir),
        "manifest_dir": Path(manifest_dir) if manifest_dir is not None else None,
        "reader": reader,
        "load_error": None,
    }

    frozen_path = ROOT / "configs" / "frozen.yaml"
    if frozen_path.is_file():
        loc_cfg, judge_cfg, _ = configs_from_yaml(frozen_path)
    else:
        loc_cfg, judge_cfg = LocatorConfig(), JudgeConfig()

    def _get_reader() -> tuple[Reader | None, str | None]:
        if state["reader"] is not None:
            return state["reader"], None
        try:
            r = Reader.load_preferred(state["models_dir"])
            state["reader"] = r
            state["load_error"] = None
            return r, None
        except ReaderError as exc:
            state["load_error"] = exc.reason_code
            return None, exc.reason_code
        except Exception:  # noqa: BLE001
            state["load_error"] = "READER_UNEXPECTED_ERROR"
            return None, "READER_UNEXPECTED_ERROR"

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "service": "VeriRay Local REST API",
            "offline_only": True,
            "endpoints": ["/health", "/verify", "/verify-batch", "/chat", "/docs"],
            "streamlit_ui": "http://127.0.0.1:8502",
        }

    def _get_reader_for_finding(finding_norm: str) -> tuple[Reader | None, str | None]:
        if finding_norm == "pneumothorax":
            return _get_reader()
        if state["reader"] is not None:
            # Re-target injected reader (e.g. in tests) without mutating state["reader"]
            base_r: Reader = state["reader"]
            canon = finding_norm.capitalize()
            try:
                idx = next(i for i, lbl in enumerate(base_r.record.label_list) if lbl.lower() == finding_norm)
            except StopIteration:
                return None, "UNSUPPORTED_FINDING"
            from dataclasses import replace as _dc_replace
            new_rec = _dc_replace(base_r.record, finding_label=canon, finding_index=idx)
            return Reader(base_r.model, new_rec, base_r.device, calibrator=None), None
        try:
            return Reader.load_for_finding(state["models_dir"], finding_norm), None
        except ReaderError as exc:
            return None, exc.reason_code
        except Exception:  # noqa: BLE001
            return None, "READER_UNEXPECTED_ERROR"

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        r, err = _get_reader()
        return HealthResponse(
            status="ok" if r is not None else "degraded",
            reader_loaded=r is not None,
            checkpoint_id=r.checkpoint_id if r is not None else None,
            reason_code=err,
        )

    @app.post("/verify", response_model=VerifyResponse)
    async def verify_single(
        file: UploadFile = File(...),
        finding_id: str = Query("pneumothorax", description="Target finding: pneumothorax, effusion, or cardiomegaly"),
    ) -> VerifyResponse:
        name = file.filename or "upload.png"
        finding_norm = finding_id.strip().lower()
        if finding_norm not in SUPPORTED_FINDINGS:
            return _fail_closed_response(name, "UNSUPPORTED_FINDING")
        r, err = _get_reader_for_finding(finding_norm)
        if r is None:
            return _fail_closed_response(name, err or "READER_CHECKPOINT_MISSING")
        try:
            payload = await file.read()
            if len(payload) > MAX_UPLOAD_BYTES:
                return _fail_closed_response(name, "INPUT_TOO_LARGE")
            res = run_verification(
                payload,
                filename_hint=name,
                reader=r,
                models_dir=state["models_dir"],
                manifest_dir=state["manifest_dir"],
                locator_cfg=loc_cfg,
                judge_cfg=judge_cfg,
                finding_id=finding_norm,
            )
            return _to_verify_response(name, res, finding_id=finding_norm)
        except Exception:  # noqa: BLE001 - never expose 500 stack traces
            return _fail_closed_response(name, "API_INTERNAL_ERROR")

    @app.post("/verify-batch", response_model=BatchVerifyResponse)
    async def verify_batch(
        files: list[UploadFile] = File(...),
        finding_id: str = Query("pneumothorax", description="Target finding: pneumothorax, effusion, or cardiomegaly"),
    ) -> BatchVerifyResponse:
        t0 = time.perf_counter()
        results: list[VerifyResponse] = []
        counts: dict[str, int] = {}
        finding_norm = finding_id.strip().lower()
        if finding_norm not in SUPPORTED_FINDINGS:
            for f in files[:MAX_BATCH_FILES]:
                vr = _fail_closed_response(f.filename or "upload.png", "UNSUPPORTED_FINDING")
                results.append(vr)
                counts[vr.verdict] = counts.get(vr.verdict, 0) + 1
            return BatchVerifyResponse(
                total_cases=len(results),
                verdict_counts=counts,
                results=results,
                total_elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            )
        r, err = _get_reader_for_finding(finding_norm)
        for f in files[:MAX_BATCH_FILES]:
            name = f.filename or "upload.png"
            if r is None:
                vr = _fail_closed_response(name, err or "READER_CHECKPOINT_MISSING")
            else:
                try:
                    payload = await f.read()
                    if len(payload) > MAX_UPLOAD_BYTES:
                        vr = _fail_closed_response(name, "INPUT_TOO_LARGE")
                    else:
                        res = run_verification(
                            payload,
                            filename_hint=name,
                            reader=r,
                            models_dir=state["models_dir"],
                            manifest_dir=state["manifest_dir"],
                            locator_cfg=loc_cfg,
                            judge_cfg=judge_cfg,
                            finding_id=finding_norm,
                        )
                        vr = _to_verify_response(name, res, finding_id=finding_norm)
                except Exception:  # noqa: BLE001
                    vr = _fail_closed_response(name, "API_INTERNAL_ERROR")
            results.append(vr)
            counts[vr.verdict] = counts.get(vr.verdict, 0) + 1
        return BatchVerifyResponse(
            total_cases=len(results),
            verdict_counts=counts,
            results=results,
            total_elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 3),
        )

    @app.post("/chat", response_model=ChatResponseSchema)
    def chat(req: ChatRequest) -> ChatResponseSchema:
        ev = None
        if req.verdict is not None and req.evidence_items:
            try:
                ev = EvidenceRecord(
                    record_id="api-chat",
                    finding_id="pneumothorax",
                    verdict=Verdict(req.verdict),
                    reason_codes=tuple(req.reason_codes or ["API_CHAT"]),
                    items=req.evidence_items,
                )
            except Exception:  # noqa: BLE001
                ev = None
        resp = answer_chat_question(req.question, ev)
        return ChatResponseSchema(
            allowed=resp.allowed,
            answer=resp.answer,
            cited_evidence_ids=list(resp.cited_evidence_ids),
            refusal_reason_codes=list(resp.refusal_reason_codes),
        )

    return app


app = create_app()
