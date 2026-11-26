"""Validation-only score calibration for the Reader (Blueprint v2 §5, 26 Nov §9).

Fits a 1D logistic (Platt) scaler on validation-split ``score_raw`` values ONLY.
Never fit on the locked test split. Outputs remain *calibrated model scores*, NEVER
probabilities that a patient has disease.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class PlattCalibrator:
    coef: float
    intercept: float
    fitted_split: str
    num_samples: int
    calibrator_id: str = "platt-val-v1"

    def __post_init__(self) -> None:
        if self.fitted_split.strip().lower() == "test":
            raise ValueError("Refusing calibrator fitted on the locked test split.")

    def calibrate(self, score_raw: float) -> float:
        """Map raw score in [0, 1] -> calibrated model score in [0, 1]."""
        p = float(np.clip(score_raw, 1e-6, 1.0 - 1e-6))
        logit = math.log(p / (1.0 - p))
        z = self.coef * logit + self.intercept
        return float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, z)))))

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "calibrator_id": self.calibrator_id,
                    "coef": self.coef,
                    "intercept": self.intercept,
                    "fitted_split": self.fitted_split,
                    "num_samples": self.num_samples,
                    "note": "Calibrated model score only; never a probability of disease.",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: Path | str) -> "PlattCalibrator":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            coef=float(d["coef"]),
            intercept=float(d["intercept"]),
            fitted_split=str(d["fitted_split"]),
            num_samples=int(d["num_samples"]),
            calibrator_id=str(d.get("calibrator_id", "platt-val-v1")),
        )


def fit_platt_calibrator(
    scores_raw: Iterable[float],
    y_true: Iterable[int],
    *,
    split_name: str = "val",
) -> PlattCalibrator:
    """Fit Platt scaling on validation data only; refuses ``split_name='test'``."""
    if split_name.strip().lower() == "test":
        raise ValueError("Calibration must never be fitted on the locked test split.")
    s = np.clip(np.asarray(list(scores_raw), dtype=np.float64), 1e-6, 1.0 - 1e-6)
    y = np.asarray(list(y_true), dtype=int)
    if len(s) == 0 or len(s) != len(y) or len(np.unique(y)) < 2:
        raise ValueError("Need non-empty binary labels from both classes to fit calibrator.")
    logits = np.log(s / (1.0 - s)).reshape(-1, 1)
    lr = LogisticRegression(C=1.0, solver="lbfgs")
    lr.fit(logits, y)
    return PlattCalibrator(
        coef=float(lr.coef_[0, 0]),
        intercept=float(lr.intercept_[0]),
        fitted_split=split_name,
        num_samples=int(len(y)),
    )
