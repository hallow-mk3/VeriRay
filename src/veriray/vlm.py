"""MedGemma 4B multimodal Claimant & grounded Explainer adapter (Blueprint 26 Nov §5, §6, §8).

100% offline at runtime: loads ONLY from a verified local directory (``models/medgemma``)
with ``local_files_only=True``. Never downloads weights at runtime (use
``scripts/fetch_weights.py --medgemma`` once after accepting the Hugging Face licence).

Memory discipline (RTX 5070 8 GB): supports 4-bit / 8-bit loading and explicit
:meth:`MedGemmaRunner.unload` to free VRAM between stages.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np
from PIL import Image
import torch

from veriray.contracts import EvidenceRecord, ExplanationSentence, VLMClaim
from veriray.verifier import VerifierReport, verify_explanation

DEFAULT_MEDGEMMA_ID = "google/medgemma-4b-it"


class VLMError(Exception):
    """Base error with machine-readable INDETERMINATE reason code."""

    def __init__(self, reason_code: str, message: str = "") -> None:
        super().__init__(f"{reason_code}: {message}" if message else reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True)
class VLMConfig:
    model_dir: Path = Path("models/medgemma")
    quantization: str = "4bit"  # "4bit" | "8bit" | "bf16"
    max_new_tokens: int = 512
    temperature: float = 0.0


def parse_vlm_claims(raw_text: str) -> tuple[VLMClaim, ...]:
    """Parse strict JSON claim list from MedGemma image-mode output; fail closed on invalid JSON."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise VLMError("VLM_EMPTY_OUTPUT", "model returned empty text")

    # Strip optional markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.IGNORECASE | re.DOTALL)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise VLMError("VLM_INVALID_JSON", str(exc)) from exc

    items = payload.get("claims") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise VLMError("VLM_INVALID_JSON", "expected a JSON list or {'claims': [...]}")

    claims: list[VLMClaim] = []
    for entry in items:
        if not isinstance(entry, dict):
            raise VLMError("VLM_INVALID_JSON", "each claim must be a JSON object")
        try:
            box = entry.get("box_xyxy")
            box_tuple = tuple(float(v) for v in box) if box is not None else None
            claims.append(
                VLMClaim(
                    finding=str(entry["finding"]),
                    status=str(entry["status"]),
                    region_description=str(entry["region_description"]),
                    box_xyxy=box_tuple,  # type: ignore[arg-type]
                    raw_json_valid=True,
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise VLMError("VLM_INVALID_CLAIM_SCHEMA", f"{type(exc).__name__}: {exc}") from exc
    return tuple(claims)


def parse_explainer_candidates(raw_text: str) -> list[ExplanationSentence]:
    """Parse JSON [{"text": ..., "evidence_ids": [...]}] into unverified ExplanationSentence objects."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.IGNORECASE | re.DOTALL)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise VLMError("VLM_EXPLAINER_INVALID_JSON", str(exc)) from exc

    items = payload.get("sentences") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise VLMError("VLM_EXPLAINER_INVALID_JSON", "expected a JSON list or {'sentences': [...]}")

    out: list[ExplanationSentence] = []
    for entry in items:
        if not isinstance(entry, dict) or "text" not in entry or "evidence_ids" not in entry:
            raise VLMError("VLM_EXPLAINER_INVALID_SCHEMA", "each item needs 'text' and 'evidence_ids'")
        ids = entry["evidence_ids"]
        if not isinstance(ids, list):
            raise VLMError("VLM_EXPLAINER_INVALID_SCHEMA", "'evidence_ids' must be a list")
        out.append(ExplanationSentence(text=str(entry["text"]), evidence_ids=tuple(str(x) for x in ids), verified=False))
    return out


def build_grounded_explainer_prompt(evidence: EvidenceRecord, *, language: str = "en") -> str:
    """Build text-only prompt containing ONLY structured EvidenceRecord (never image/filename/age/sex)."""
    payload = {
        "finding": evidence.finding_id,
        "verdict": evidence.verdict.value,
        "reason_codes": list(evidence.reason_codes),
        "language": language,
        "evidence_items": dict(evidence.items),
    }
    return (
        "Write 2 to 4 concise plain-language sentences describing ONLY the recorded test evidence below. "
        "Do NOT diagnose, recommend treatment, name drugs/dosages, or invent any number not in evidence_items. "
        "Return strict JSON: {\"sentences\": [{\"text\": \"...\", \"evidence_ids\": [\"E01_...\"]}]}.\n"
        f"EVIDENCE_JSON={json.dumps(payload, sort_keys=True)}"
    )


def box_mask_iou(mask_2d: np.ndarray, box_xyxy: tuple[float, float, float, float]) -> float:
    """Compute IoU between a 2D boolean mask and a normalized (x1,y1,x2,y2) claimant box."""
    m = np.asarray(mask_2d, dtype=bool)
    h, w = m.shape
    x1, y1, x2, y2 = box_xyxy
    ix1, iy1 = max(0, int(np.floor(x1 * w))), max(0, int(np.floor(y1 * h)))
    ix2, iy2 = min(w, int(np.ceil(x2 * w))), min(h, int(np.ceil(y2 * h)))
    box_mask = np.zeros((h, w), dtype=bool)
    if ix2 > ix1 and iy2 > iy1:
        box_mask[iy1:iy2, ix1:ix2] = True
    inter = int(np.logical_and(m, box_mask).sum())
    union = int(np.logical_or(m, box_mask).sum())
    return 0.0 if union == 0 else inter / float(union)


def run_grounded_vlm_explainer(
    evidence: EvidenceRecord,
    text_generator: Callable[[str], str],
    *,
    language: str = "en",
) -> VerifierReport:
    """Run any text generator on structured EvidenceRecord and filter through the Claim Verifier."""
    prompt = build_grounded_explainer_prompt(evidence, language=language)
    try:
        raw = text_generator(prompt)
        candidates = parse_explainer_candidates(raw)
    except Exception as exc:  # noqa: BLE001 - count parse/generation failure instead of masking as 0% drop rate
        candidates = [ExplanationSentence(text=f"VLM output failed validation ({type(exc).__name__}).", evidence_ids=())]
    return verify_explanation(candidates, evidence)


class MedGemmaRunner:
    """Offline-only wrapper around a local Hugging Face MedGemma checkpoint."""

    def __init__(self, model: Any, processor: Any, cfg: VLMConfig, peak_vram_mb: float = 0.0) -> None:
        self._model = model
        self._processor = processor
        self.cfg = cfg
        self.peak_vram_mb = peak_vram_mb

    @classmethod
    def load(cls, cfg: VLMConfig = VLMConfig()) -> "MedGemmaRunner":
        model_dir = Path(cfg.model_dir)
        if not model_dir.is_dir() or not (model_dir / "config.json").is_file():
            raise VLMError(
                "VLM_CHECKPOINT_MISSING",
                f"Local MedGemma directory '{model_dir}' not found. Run scripts/fetch_weights.py --medgemma first.",
            )
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore[import-not-found]
        except ImportError as exc:
            raise VLMError("VLM_DEPENDENCY_MISSING", f"transformers not installed: {exc}") from exc

        kwargs: dict[str, Any] = {"local_files_only": True, "device_map": "auto"}
        if cfg.quantization in {"4bit", "8bit"}:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]
            except ImportError as exc:
                raise VLMError("VLM_DEPENDENCY_MISSING", f"bitsandbytes/transformers config missing: {exc}") from exc
            if cfg.quantization == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        try:
            processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
            model = AutoModelForImageTextToText.from_pretrained(str(model_dir), **kwargs)
            model.eval()
        except Exception as exc:  # noqa: BLE001
            raise VLMError("VLM_LOAD_FAILED", f"{type(exc).__name__}: {exc}") from exc

        peak_mb = (
            float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
            if torch.cuda.is_available()
            else 0.0
        )
        return cls(model, processor, cfg, peak_mb)

    def claim_on_image(self, gray_0_255: np.ndarray) -> tuple[VLMClaim, ...]:
        """Image mode: produce structured VLMClaim items for one 2D image."""
        if self._model is None:
            raise VLMError("VLM_NOT_LOADED", "model was unloaded")
        u8 = np.clip(np.rint(gray_0_255), 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(u8).convert("RGB")
        prompt = (
            "List chest X-ray findings as strict JSON: "
            "{\"claims\": [{\"finding\": \"pneumothorax\", \"status\": \"present|absent|uncertain\", "
            "\"region_description\": \"...\", \"box_xyxy\": [x1, y1, x2, y2]}]}. "
            "Normalized coordinates in [0,1]. Output JSON only."
        )
        messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": prompt}]}]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens, do_sample=False)
        decoded = self._processor.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        return parse_vlm_claims(decoded)

    def generate_text(self, prompt: str) -> str:
        """Text-only mode for grounded explanation (receives EvidenceRecord JSON only)."""
        if self._model is None:
            raise VLMError("VLM_NOT_LOADED", "model was unloaded")
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens, do_sample=False)
        return str(self._processor.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True))

    def unload(self) -> None:
        """Release model/processor references and clear CUDA cache so other models can run."""
        self._model = None
        self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
