"""API load & injected-fault test runner (Blueprint 26 Nov §9 Experiment 9)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path
import time
from typing import Any

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from api.app import create_app
from api.schemas import VerifyResponse
from veriray.reader import Reader


def _png_bytes(seed: int = 1) -> bytes:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
    img = 90 + 50 * np.sin(xx / 31.0) * np.cos(yy / 29.0) + rng.normal(0, 6, size=(256, 256))
    buf = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def run_api_load_and_fault_test(
    *,
    reader: Reader | None,
    missing_models_dir: Path,
    n_sequential: int = 10,
    n_concurrent: int = 6,
) -> dict[str, Any]:
    """Measure sequential/concurrent p50/p95 latency, schema-valid rate, and injected-fault handling."""
    app_ok = create_app(manifest_dir=None, reader=reader)
    client_ok = TestClient(app_ok)
    payload = _png_bytes()

    latencies: list[float] = []
    schema_ok = 0
    errors = 0
    total = 0

    def _one_call(client: TestClient, data: bytes, fname: str) -> tuple[bool, bool, float]:
        t0 = time.perf_counter()
        resp = client.post("/verify", files={"file": (fname, data, "image/png")})
        ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code != 200:
            return False, False, ms
        try:
            VerifyResponse.model_validate(resp.json())
            return True, True, ms
        except Exception:  # noqa: BLE001
            return True, False, ms

    # 1. Sequential load
    for i in range(n_sequential):
        total += 1
        ok_http, ok_schema, ms = _one_call(client_ok, payload, f"seq_{i}.png")
        latencies.append(ms)
        schema_ok += int(ok_schema)
        errors += int(not ok_http)

    # 2. Concurrent load
    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = [pool.submit(_one_call, client_ok, payload, f"conc_{i}.png") for i in range(n_concurrent)]
        for f in futs:
            total += 1
            ok_http, ok_schema, ms = f.result()
            latencies.append(ms)
            schema_ok += int(ok_schema)
            errors += int(not ok_http)

    # 3. Injected faults: corrupt image + missing model directory
    fault_checks: list[bool] = []
    resp_corrupt = client_ok.post("/verify", files={"file": ("corrupt.png", b"not-a-png", "image/png")})
    fault_checks.append(
        resp_corrupt.status_code == 200 and VerifyResponse.model_validate(resp_corrupt.json()).verdict == "INDETERMINATE"
    )

    app_bad = create_app(models_dir=missing_models_dir, manifest_dir=None, reader=None)
    client_bad = TestClient(app_bad)
    resp_missing = client_bad.post("/verify", files={"file": ("ok.png", payload, "image/png")})
    fault_checks.append(
        resp_missing.status_code == 200 and VerifyResponse.model_validate(resp_missing.json()).verdict == "INDETERMINATE"
    )

    arr = np.asarray(latencies, dtype=float)
    return {
        "total_load_requests": total,
        "http_error_rate": errors / float(max(1, total)),
        "schema_valid_rate": schema_ok / float(max(1, total)),
        "p50_ms": float(np.percentile(arr, 50)) if len(arr) else None,
        "p95_ms": float(np.percentile(arr, 95)) if len(arr) else None,
        "injected_faults_passed": sum(fault_checks),
        "injected_faults_total": len(fault_checks),
    }
