"""End-to-end Step 2 tests with the network blocked, plus static no-network-library checks."""

from __future__ import annotations

import json
import re
import socket
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import synthetic_xray, write_checkpoint_dir
from veriray.contracts import Verdict, validate_safe_language
from veriray.manifest import read_run_manifest
from veriray.pipeline import BEHAVIOUR_NOTICE, run_step2

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def no_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network access attempted")

    class NoSocket(socket.socket):
        def __init__(self, *a, **k):
            boom()

    monkeypatch.setattr(socket, "socket", NoSocket)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    monkeypatch.setattr(socket, "gethostbyname", boom)


def test_end_to_end_offline(no_network, fake_models_dir, tmp_path):
    res = run_step2(synthetic_xray(), models_dir=fake_models_dir, manifest_dir=tmp_path / "runs")
    assert res.failed_stage in (None, "locator_quality")  # a random-init model may legitimately give a weak mask
    assert res.gate and res.gate.accepted
    assert res.reader and 0.0 <= res.reader.score_raw <= 1.0
    assert res.locator is not None and res.display_image.shape == (224, 224)
    if res.failed_stage is None:
        assert res.verdict is None  # no Judge yet: no verdict is ever invented
    else:
        assert res.verdict is Verdict.INDETERMINATE and res.reason_codes
    # Run manifest: config hash, checkpoint hash, seeds, offline flag.
    m = read_run_manifest(res.manifest_path)
    assert m.offline_only and m.random_seeds["torch"] == 0
    assert m.checkpoint_hashes["fake-test-weights"] == json.loads((fake_models_dir / "CHECKPOINT.json").read_text())["sha256"]
    assert len(m.config_sha256) == 64


def test_end_to_end_missing_weights_is_indeterminate(no_network, tmp_path):
    res = run_step2(synthetic_xray(), models_dir=tmp_path / "empty", manifest_dir=None)
    assert res.verdict is Verdict.INDETERMINATE and res.reason_codes == ("READER_CHECKPOINT_MISSING",)


def test_end_to_end_corrupt_weights_is_indeterminate(no_network, tmp_path):
    d = write_checkpoint_dir(tmp_path / "m")
    p = d / "fake-densenet121.pt"
    p.write_bytes(p.read_bytes()[:-100])
    res = run_step2(synthetic_xray(), models_dir=d, manifest_dir=None)
    assert res.verdict is Verdict.INDETERMINATE and res.reason_codes == ("READER_CHECKPOINT_HASH_MISMATCH",)


def test_end_to_end_bad_input_is_indeterminate(no_network, fake_reader, tmp_path):
    res = run_step2(b"definitely not an image", reader=fake_reader, manifest_dir=None)
    assert res.verdict is Verdict.INDETERMINATE and res.reason_codes[0] == "GATE_REJECT"
    assert res.reader is None and res.locator is None


def test_behaviour_notice_text_and_safe_language():
    assert BEHAVIOUR_NOTICE == ("Score and heatmap describe model behaviour, not a diagnosis. "
                                "Research prototype — not for clinical use.")
    validate_safe_language(BEHAVIOUR_NOTICE)
    app = (ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "BEHAVIOUR_NOTICE" in app and "probability" not in app.lower()


NETWORK_PATTERN = re.compile(
    r"^\s*(?:import|from)\s+(urllib|requests|http|httpx|aiohttp|socket|ftplib|smtplib|websocket|websockets|urllib3|boto3|openai|google\.)"
    r"|\burlopen\b|\burlretrieve\b|xrv\.utils\.download|get_weights|\.download\(",
    re.M,
)


def test_no_network_libraries_in_src_or_app():
    offenders = []
    for folder in ("src", "app"):
        for py in (ROOT / folder).rglob("*.py"):
            if NETWORK_PATTERN.search(py.read_text(encoding="utf-8")):
                offenders.append(str(py))
    assert not offenders, offenders
