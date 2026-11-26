"""Reader (Blueprint v2.0 §5): documented chest-X-ray DenseNet-121 from torchxrayvision.

Loads weights from the local ``models/`` directory ONLY. There is no download path and
no network use here (the one allowed fetch lives in scripts/fetch_weights.py). Fails
closed with typed errors if the checkpoint file/record is missing or its SHA-256 does
not match. The output is a *model score*, never a probability that a patient has disease.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchxrayvision as xrv

from veriray.calibration import PlattCalibrator
from veriray.contracts import ReaderContract

PREPROCESSING_ID = "xrv-centercrop-cv2area224-norm1024-v1"
THRESHOLD_ID = "none-uncalibrated-v0"  # no threshold chosen yet (frozen on tuning data in a later step)
CALIBRATOR_RECORD = "platt_calibrator.json"
INPUT_RESOLUTION = 224
CHECKPOINT_RECORD = "CHECKPOINT.json"


class ReaderError(Exception):
    """Base class; ``reason_code`` is the machine-readable INDETERMINATE reason."""

    reason_code = "READER_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class CheckpointMissingError(ReaderError):
    reason_code = "READER_CHECKPOINT_MISSING"


class CheckpointHashMismatchError(ReaderError):
    reason_code = "READER_CHECKPOINT_HASH_MISMATCH"


class CheckpointRecordError(ReaderError):
    reason_code = "READER_CHECKPOINT_RECORD_INVALID"


class CheckpointLoadError(ReaderError):
    reason_code = "READER_CHECKPOINT_LOAD_FAILED"


class PreprocessingError(ReaderError):
    reason_code = "READER_PREPROCESSING_FAILED"


class InferenceError(ReaderError):
    reason_code = "READER_INFERENCE_FAILED"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocess(gray_0_255: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    """Exactly the torchxrayvision recipe: scale to ~[-1024, 1024], 1 channel, center crop, 224px.

    Returns (tensor of shape 1x1x224x224, the 224x224 model-space view scaled to 0..255 for display).
    """
    try:
        img = np.asarray(gray_0_255, dtype=np.float32)
        if img.ndim != 2 or img.size == 0 or not np.all(np.isfinite(img)):
            raise ValueError(f"expected finite non-empty 2D array, got shape {img.shape}")
        if img.min() < -1e-3 or img.max() > 255.0 + 1e-3:
            raise ValueError("pixel values must be in [0, 255] before normalization")
        img = np.clip(img, 0.0, 255.0)
        norm = xrv.utils.normalize(img, 255)  # -> roughly [-1024, 1024]
        norm = norm[None, ...]  # single channel, CxHxW
        norm = xrv.datasets.XRayCenterCrop()(norm)
        norm = xrv.datasets.XRayResizer(INPUT_RESOLUTION, engine="cv2")(norm)
        if norm.shape != (1, INPUT_RESOLUTION, INPUT_RESOLUTION) or not np.all(np.isfinite(norm)):
            raise ValueError(f"unexpected preprocessed shape {norm.shape}")
        if norm.min() < -1024.5 or norm.max() > 1024.5:
            raise ValueError("normalized range outside [-1024, 1024]")
    except Exception as exc:  # noqa: BLE001
        raise PreprocessingError(f"{type(exc).__name__}: {exc}") from exc
    display = np.clip((norm[0] / 1024.0 + 1.0) * 127.5, 0.0, 255.0).astype(np.float32)
    return torch.from_numpy(norm.astype(np.float32))[None], display


@dataclass(frozen=True)
class CheckpointRecord:
    weights_id: str
    filename: str
    sha256: str
    label_list: tuple[str, ...]
    finding_label: str
    finding_index: int
    source_url: str
    license: str
    training_datasets: tuple[str, ...]

    @property
    def checkpoint_id(self) -> str:
        return f"{self.weights_id}@sha256:{self.sha256[:12]}"


def read_checkpoint_record(models_dir: Path) -> CheckpointRecord:
    rec_path = Path(models_dir) / CHECKPOINT_RECORD
    if not rec_path.is_file():
        raise CheckpointMissingError(f"{rec_path} not found; run scripts/fetch_weights.py once (online).")
    try:
        d: dict[str, Any] = json.loads(rec_path.read_text(encoding="utf-8"))
        rec = CheckpointRecord(
            weights_id=str(d["weights_id"]),
            filename=str(d["filename"]),
            sha256=str(d["sha256"]).lower(),
            label_list=tuple(d["label_list"]),
            finding_label=str(d["finding_label"]),
            finding_index=int(d["finding_index"]),
            source_url=str(d["source_url"]),
            license=str(d["license"]),
            training_datasets=tuple(d["training_datasets"]),
        )
    except Exception as exc:  # noqa: BLE001
        raise CheckpointRecordError(f"{rec_path}: {type(exc).__name__}: {exc}") from exc
    if len(rec.sha256) != 64:
        raise CheckpointRecordError("record sha256 is not a 64-char digest")
    if rec.finding_label != "Pneumothorax" or rec.finding_index >= len(rec.label_list) or \
            rec.label_list[rec.finding_index] != "Pneumothorax":
        raise CheckpointRecordError("label list does not contain Pneumothorax at finding_index")
    return rec


class Reader:
    """Pneumothorax Reader. Construct with :meth:`load`; never downloads."""

    def __init__(
        self,
        model: torch.nn.Module,
        record: CheckpointRecord,
        device: torch.device,
        calibrator: PlattCalibrator | None = None,
    ) -> None:
        self.model = model
        self.record = record
        self.device = device
        self.finding_index = record.finding_index
        self.checkpoint_id = record.checkpoint_id
        self.preprocessing_id = PREPROCESSING_ID
        self.calibrator = calibrator

    @classmethod
    def load(cls, models_dir: Path | str, device: str = "cpu") -> "Reader":
        models_dir = Path(models_dir)
        rec = read_checkpoint_record(models_dir)
        weights_path = models_dir / rec.filename
        if not weights_path.is_file():
            raise CheckpointMissingError(f"weights file {weights_path} not found; run scripts/fetch_weights.py.")
        actual = _sha256_file(weights_path)
        if actual != rec.sha256:
            # Never unpickle a file whose hash does not match the recorded one.
            raise CheckpointHashMismatchError(
                f"SHA-256 mismatch for {weights_path.name}: expected {rec.sha256}, got {actual}."
            )
        try:
            # The file is a pickled torch module (torchxrayvision format). Unpickling executes code, so it is
            # only done AFTER the SHA-256 above matched the recorded hash.
            saved = torch.load(weights_path, map_location="cpu", weights_only=False)
            state = saved.state_dict() if hasattr(saved, "state_dict") else dict(saved)
            model = xrv.models.DenseNet(weights=None, num_classes=len(rec.label_list), in_channels=1)
            model.load_state_dict(state)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            dev = torch.device(device)
            model.to(dev)
        except Exception as exc:  # noqa: BLE001
            raise CheckpointLoadError(f"{type(exc).__name__}: {exc}") from exc
        cal_path = models_dir / CALIBRATOR_RECORD
        calibrator = PlattCalibrator.load(cal_path) if cal_path.is_file() else None
        return cls(model, rec, dev, calibrator=calibrator)

    @classmethod
    def load_preferred(cls, models_dir: Path | str, device: str = "cpu") -> "Reader":
        """Load NIH_TRAINED_CHECKPOINT.json if present (with SHA-256 check), else fall back to CHECKPOINT.json."""
        mdir = Path(models_dir)
        nih_rec = mdir / "NIH_TRAINED_CHECKPOINT.json"
        if not nih_rec.is_file():
            return cls.load(mdir, device=device)
        try:
            d: dict[str, Any] = json.loads(nih_rec.read_text(encoding="utf-8"))
            rec = CheckpointRecord(
                weights_id=str(d["weights_id"]),
                filename=str(d["filename"]),
                sha256=str(d["sha256"]).lower(),
                label_list=tuple(d["label_list"]),
                finding_label=str(d["finding_label"]),
                finding_index=int(d["finding_index"]),
                source_url=str(d.get("source_url", f"local:{d['filename']}")),
                license=str(d.get("license", "torchxrayvision Apache-2.0; NIH ChestX-ray14 terms apply")),
                training_datasets=tuple(d.get("training_datasets", ("NIH ChestX-ray14 train split ONLY",))),
            )
        except Exception as exc:  # noqa: BLE001
            raise CheckpointRecordError(f"{nih_rec}: {type(exc).__name__}: {exc}") from exc
        wpath = mdir / rec.filename
        if not wpath.is_file():
            raise CheckpointMissingError(f"weights file {wpath} not found")
        actual = _sha256_file(wpath)
        if actual != rec.sha256:
            raise CheckpointHashMismatchError(
                f"SHA-256 mismatch for {wpath.name}: expected {rec.sha256}, got {actual}."
            )
        try:
            state = torch.load(wpath, map_location="cpu", weights_only=True)
            model = xrv.models.DenseNet(weights=None, num_classes=len(rec.label_list), in_channels=1)
            model.load_state_dict(state)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            dev = torch.device(device)
            model.to(dev)
        except Exception as exc:  # noqa: BLE001
            raise CheckpointLoadError(f"{type(exc).__name__}: {exc}") from exc
        cal_path = mdir / CALIBRATOR_RECORD
        calibrator = PlattCalibrator.load(cal_path) if cal_path.is_file() else None
        return cls(model, rec, dev, calibrator=calibrator)

    # --- single code path shared with the Locator (Grad-CAM++) -------------------------------
    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Last convolutional feature layer: ReLU(features.norm5(...)), shape 1x1024x7x7 (same ops as xrv)."""
        return F.relu(self.model.features(x.to(self.device)))

    def head_logits(self, feature_map: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(feature_map, (1, 1)).flatten(1)
        return self.model.classifier(pooled)

    def score(self, gray_0_255: np.ndarray) -> tuple[ReaderContract, torch.Tensor, np.ndarray]:
        """Return (ReaderContract, model-input tensor, 224px display view).

        score_raw = sigmoid(logit) of the Pneumothorax output: a *model score* in [0, 1], uncalibrated.
        Also records ``unrelated_scores`` for every other non-empty output label.
        """
        x, display = preprocess(gray_0_255)
        try:
            with torch.no_grad():
                logits = self.head_logits(self.feature_map(x))[0]
                probs = torch.sigmoid(logits)
                raw = float(probs[self.finding_index].item())
                unrelated = {
                    lbl: float(probs[idx].item())
                    for idx, lbl in enumerate(self.record.label_list)
                    if idx != self.finding_index and lbl
                }
            cal_score = self.calibrator.calibrate(raw) if self.calibrator is not None else None
            thresh_id = self.calibrator.calibrator_id if self.calibrator is not None else THRESHOLD_ID
            contract = ReaderContract(
                score_raw=raw,
                score_calibrated_optional=cal_score,
                threshold_id=thresh_id,
                checkpoint_id=self.checkpoint_id,
                preprocessing_id=self.preprocessing_id,
                unrelated_scores=unrelated,
            )
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"{type(exc).__name__}: {exc}") from exc
        return contract, x, display


@dataclass(frozen=True)
class ReaderScreeningSummary:
    """Summary of all monitored Reader outputs for UI/reporting (never a diagnosis)."""

    target_score: float
    target_flagged: bool
    flagged_other_findings: tuple[tuple[str, float], ...]
    all_scores_low: bool
    max_other_label: str | None
    max_other_score: float


def summarize_reader_findings(
    rc: ReaderContract,
    *,
    target_threshold: float = 0.5,
    other_threshold: float = 0.5,
) -> ReaderScreeningSummary:
    """Summarize whether the target finding or any other monitored label exceeded threshold."""
    target_flagged = float(rc.score_raw) >= float(target_threshold)
    flagged_others = tuple(
        sorted(
            ((lbl, float(val)) for lbl, val in rc.unrelated_scores.items() if float(val) >= float(other_threshold)),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
    if rc.unrelated_scores:
        max_lbl, max_val = max(rc.unrelated_scores.items(), key=lambda kv: float(kv[1]))
        max_other_label: str | None = str(max_lbl)
        max_other_score = float(max_val)
    else:
        max_other_label = None
        max_other_score = 0.0
    return ReaderScreeningSummary(
        target_score=float(rc.score_raw),
        target_flagged=target_flagged,
        flagged_other_findings=flagged_others,
        all_scores_low=(not target_flagged) and (not flagged_others),
        max_other_label=max_other_label,
        max_other_score=max_other_score,
    )
