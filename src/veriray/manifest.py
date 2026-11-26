"""Reproducible RunManifest dataclass and atomic JSON writer/reader (Section 4.1 & Appendix A)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from types import MappingProxyType
from typing import Any, Mapping
import uuid

from veriray.contracts import SUPPORTED_FINDING


def sha256_bytes(data: bytes) -> str:
    """Compute hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Compute hex SHA-256 digest of a file on disk in streaming 1 MiB chunks."""
    file_path = Path(path)
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_mapping(config: Mapping[str, Any]) -> str:
    """Compute deterministic SHA-256 digest of a JSON-serializable configuration mapping."""
    canonical = json.dumps(dict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(canonical)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Section 4.1 run provenance record stored for every reported experiment.

    Stores run configuration, model/checkpoint hashes, dataset split identifiers,
    random seeds, offline enforcement flag, and runtime/hardware metadata.
    """

    run_id: str
    timestamp_utc: str
    finding_id: str
    run_config: Mapping[str, Any]
    config_sha256: str
    model_id: str
    checkpoint_hashes: Mapping[str, str]
    preprocessing_id: str
    dataset_split_id: str
    random_seeds: Mapping[str, int]
    offline_only: bool = True
    python_version: str = field(default_factory=platform.python_version)
    platform_summary: str = field(
        default_factory=lambda: f"{platform.system()} {platform.release()} ({platform.machine()})"
    )
    stage_latencies_ms: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.run_id or not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string.")
        if not self.timestamp_utc or not self.timestamp_utc.strip():
            raise ValueError("timestamp_utc must be a non-empty ISO-8601 string.")
        norm_finding = self.finding_id.strip().lower()
        if norm_finding != SUPPORTED_FINDING:
            raise ValueError(
                f"RunManifest finding_id must be '{SUPPORTED_FINDING}' in v2.0; got '{self.finding_id}'."
            )
        object.__setattr__(self, "finding_id", norm_finding)

        if not self.offline_only:
            raise ValueError("Blueprint v2.0 requires offline_only=True (no external API uploads).")
        if len(self.config_sha256) != 64:
            raise ValueError("config_sha256 must be a 64-character hex SHA-256 digest.")
        if not self.model_id or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string.")
        if not self.preprocessing_id or not self.preprocessing_id.strip():
            raise ValueError("preprocessing_id must be a non-empty string.")
        if not self.dataset_split_id or not self.dataset_split_id.strip():
            raise ValueError("dataset_split_id must be a non-empty string.")

        cp_dict = {str(k): str(v) for k, v in dict(self.checkpoint_hashes).items()}
        if not cp_dict:
            raise ValueError("checkpoint_hashes must record at least one model/checkpoint hash.")
        for name, digest in cp_dict.items():
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()):
                raise ValueError(f"Checkpoint '{name}' must have a valid 64-char hex SHA-256 hash; got '{digest}'.")

        seeds_dict = {str(k): int(v) for k, v in dict(self.random_seeds).items()}
        if not seeds_dict:
            raise ValueError("random_seeds must record at least one seed.")

        lat_dict = {str(k): float(v) for k, v in dict(self.stage_latencies_ms).items()}

        object.__setattr__(self, "run_config", MappingProxyType(dict(self.run_config)))
        object.__setattr__(self, "checkpoint_hashes", MappingProxyType(cp_dict))
        object.__setattr__(self, "random_seeds", MappingProxyType(seeds_dict))
        object.__setattr__(self, "stage_latencies_ms", MappingProxyType(lat_dict))

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain JSON-serializable dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp_utc": self.timestamp_utc,
            "finding_id": self.finding_id,
            "run_config": dict(self.run_config),
            "config_sha256": self.config_sha256,
            "model_id": self.model_id,
            "checkpoint_hashes": dict(self.checkpoint_hashes),
            "preprocessing_id": self.preprocessing_id,
            "dataset_split_id": self.dataset_split_id,
            "random_seeds": dict(self.random_seeds),
            "offline_only": self.offline_only,
            "python_version": self.python_version,
            "platform_summary": self.platform_summary,
            "stage_latencies_ms": dict(self.stage_latencies_ms),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunManifest":
        """Reconstruct and validate a RunManifest from a parsed JSON dictionary."""
        return cls(
            run_id=str(payload["run_id"]),
            timestamp_utc=str(payload["timestamp_utc"]),
            finding_id=str(payload["finding_id"]),
            run_config=dict(payload["run_config"]),
            config_sha256=str(payload["config_sha256"]),
            model_id=str(payload["model_id"]),
            checkpoint_hashes=dict(payload["checkpoint_hashes"]),
            preprocessing_id=str(payload["preprocessing_id"]),
            dataset_split_id=str(payload["dataset_split_id"]),
            random_seeds=dict(payload["random_seeds"]),
            offline_only=bool(payload.get("offline_only", True)),
            python_version=str(payload.get("python_version", platform.python_version())),
            platform_summary=str(payload.get("platform_summary", "")),
            stage_latencies_ms=dict(payload.get("stage_latencies_ms", {})),
        )


def create_run_manifest(
    *,
    run_config: Mapping[str, Any],
    model_id: str,
    checkpoint_hashes: Mapping[str, str],
    preprocessing_id: str,
    dataset_split_id: str,
    random_seeds: Mapping[str, int],
    finding_id: str = SUPPORTED_FINDING,
    stage_latencies_ms: Mapping[str, float] | None = None,
    run_id: str | None = None,
) -> RunManifest:
    """Create a validated RunManifest with computed canonical config SHA-256."""
    now = datetime.now(timezone.utc)
    generated_id = run_id or f"vr-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    cfg_copy = json.loads(json.dumps(dict(run_config)))
    cfg_hash = sha256_mapping(cfg_copy)
    return RunManifest(
        run_id=generated_id,
        timestamp_utc=now.isoformat(),
        finding_id=finding_id,
        run_config=cfg_copy,
        config_sha256=cfg_hash,
        model_id=model_id,
        checkpoint_hashes=checkpoint_hashes,
        preprocessing_id=preprocessing_id,
        dataset_split_id=dataset_split_id,
        random_seeds=random_seeds,
        offline_only=True,
        stage_latencies_ms=stage_latencies_ms or {},
    )


def write_run_manifest(manifest: RunManifest, output_path: Path | str) -> Path:
    """Atomically write a validated RunManifest to disk as formatted UTF-8 JSON."""
    if not isinstance(manifest, RunManifest):
        raise TypeError("manifest must be a RunManifest instance.")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    serialized = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    tmp_path.write_text(serialized, encoding="utf-8")
    tmp_path.replace(target)
    return target


def read_run_manifest(path: Path | str) -> RunManifest:
    """Read and validate a RunManifest JSON file from disk."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return RunManifest.from_dict(raw)
