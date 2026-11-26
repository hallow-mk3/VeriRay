"""Configuration loader/writer for VeriRay (links configs/default.yaml and frozen.yaml to modules)."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from veriray.judge import JudgeConfig
from veriray.locator import LocatorConfig


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def configs_from_yaml(path: Path | str) -> tuple[LocatorConfig, JudgeConfig, dict[str, Any]]:
    raw = load_yaml_config(path)
    loc_d = dict(raw.get("locator") or {})
    loc_d.pop("locator_id", None)
    jdg_d = dict(raw.get("judge") or {})
    if "comparison_fill_methods" in jdg_d and isinstance(jdg_d["comparison_fill_methods"], list):
        jdg_d["comparison_fill_methods"] = tuple(jdg_d["comparison_fill_methods"])
    return LocatorConfig(**loc_d), JudgeConfig(**jdg_d), raw


def freeze_config(
    locator_cfg: LocatorConfig,
    judge_cfg: JudgeConfig,
    *,
    out_path: Path | str = Path("configs/frozen.yaml"),
    fitted_split: str = "val",
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, str]:
    """Write a frozen configuration YAML + SHA-256 digest; refuses if fitted on 'test'."""
    if fitted_split.strip().lower() == "test":
        raise ValueError("Configuration must never be tuned or frozen using the locked test split.")
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "frozen": True,
        "fitted_split": fitted_split,
        "finding": "pneumothorax",
        "locator": asdict(locator_cfg),
        "judge": {
            **asdict(judge_cfg),
            "comparison_fill_methods": list(judge_cfg.comparison_fill_methods),
        },
        "tuning_metadata": dict(metadata or {}),
    }
    text = yaml.safe_dump(payload, sort_keys=False)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    target.write_text(text, encoding="utf-8")
    target.with_suffix(".sha256.json").write_text(
        json.dumps({"config_path": str(target), "sha256": digest, "fitted_split": fitted_split}, indent=2) + "\n",
        encoding="utf-8",
    )
    return target, digest
