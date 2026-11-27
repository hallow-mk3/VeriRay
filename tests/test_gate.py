"""Input Gate tests: bad input => REJECT/RETAKE with machine-readable reasons, never an exception."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tests.conftest import synthetic_xray
from veriray.contracts import InputContract, InputGateContract, InputGateStatus
from veriray.gate import run_input_gate

KW = dict(model_id="m", preprocessing_id="p")


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_good_image_accepted_and_metadata_allowlisted():
    inp, gate = run_input_gate(_png_bytes(synthetic_xray()), raw_metadata={"view": "PA", "PatientName": "X", "sample_id": "s1"}, **KW)
    assert gate.status is InputGateStatus.ACCEPT and isinstance(inp, InputContract)
    assert dict(inp.image_metadata_allowlist) == {"view": "PA", "sample_id": "s1"} or "PatientName" not in inp.image_metadata_allowlist


def test_unsupported_format_rejected(tmp_path: Path):
    p = tmp_path / "scan.pdf"
    p.write_bytes(b"%PDF-1.4 whatever")
    inp, gate = run_input_gate(p, **KW)
    assert inp is None and gate.status is InputGateStatus.REJECT and gate.reason_codes == ("UNSUPPORTED_FILE_FORMAT",)


def test_corrupt_file_rejected(tmp_path: Path):
    good = _png_bytes(synthetic_xray())
    p = tmp_path / "bad.png"
    p.write_bytes(good[: len(good) // 3])
    inp, gate = run_input_gate(p, **KW)
    assert inp is None and gate.status is InputGateStatus.REJECT and "CORRUPT_OR_UNREADABLE_IMAGE" in gate.reason_codes


def test_random_bytes_with_image_extension_rejected(tmp_path: Path):
    p = tmp_path / "noise.png"
    p.write_bytes(np.random.default_rng(0).bytes(4096))
    _, gate = run_input_gate(p, **KW)
    assert gate.status is InputGateStatus.REJECT


@pytest.mark.parametrize("shape,code", [((100, 100), "IMAGE_DIMENSIONS_TOO_SMALL"), ((9000, 300), "IMAGE_DIMENSIONS_TOO_LARGE")])
def test_wrong_dimensions_rejected(shape, code):
    arr = np.clip(synthetic_xray(300, 300)[0:1, :].repeat(shape[0], axis=0)[:, : min(shape[1], 300)], 0, 255)
    if shape[1] > arr.shape[1]:
        arr = np.pad(arr, ((0, 0), (0, shape[1] - arr.shape[1])), mode="edge")
    arr = arr.astype(np.uint8)
    inp, gate = run_input_gate(arr, **KW)
    assert inp is None and gate.status is InputGateStatus.REJECT and code in gate.reason_codes


def test_too_dark_is_retake():
    dark = (synthetic_xray() * 0.02).astype(np.uint8)
    inp, gate = run_input_gate(dark, **KW)
    assert inp is None and gate.status is InputGateStatus.RETAKE and "IMAGE_TOO_DARK" in gate.reason_codes


def test_blank_image_is_retake():
    inp, gate = run_input_gate(np.full((300, 300), 128, np.uint8), **KW)
    assert inp is None and gate.status is InputGateStatus.RETAKE and "LOW_CONTRAST_OR_BLANK_IMAGE" in gate.reason_codes


@pytest.mark.parametrize("junk", [
    "not an image", 12345, None, {"a": 1}, b"", b"hello, this is plain text", b"\x00" * 10,
    np.array(["a", "b"]), np.zeros((4, 4, 4, 4)), np.full((300, 300), np.nan),
])
def test_non_image_never_raises(junk):
    inp, gate = run_input_gate(junk, **KW)
    assert inp is None and isinstance(gate, InputGateContract)
    assert gate.status in (InputGateStatus.REJECT, InputGateStatus.RETAKE)
    assert gate.reason_codes and all(c.isupper() or "_" in c for c in gate.reason_codes)


def test_missing_file_and_text_file_rejected(tmp_path: Path):
    _, g1 = run_input_gate(tmp_path / "nope.png", **KW)
    assert g1.reason_codes == ("FILE_NOT_FOUND",)
    t = tmp_path / "notes.txt"
    t.write_text("hello")
    _, g2 = run_input_gate(t, **KW)
    assert g2.reason_codes == ("UNSUPPORTED_FILE_FORMAT",)


def test_lateral_view_rejected():
    _, gate = run_input_gate(synthetic_xray(), raw_metadata={"view": "LATERAL"}, **KW)
    assert gate.status is InputGateStatus.REJECT and gate.reason_codes == ("UNSUPPORTED_VIEW",)


def test_color_scene_rejected():
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 255, size=(300, 300, 3), dtype=np.uint8)
    _, gate = run_input_gate(rgb, **KW)
    assert gate.status is InputGateStatus.REJECT


def test_cleaner_is_not_imported_anywhere():
    root = Path(__file__).resolve().parents[1]
    for folder in ("src", "app", "scripts", "tests"):
        for py in (root / folder).rglob("*.py"):
            if py.name in {"cleaner.py", "test_gate.py"}:
                continue
            text = py.read_text(encoding="utf-8")
            assert "veriray.cleaner" not in text and "import cleaner" not in text, py


def test_oversized_bytes_input_rejected():
    from veriray.gate import MAX_INPUT_BYTES

    inp, gate = run_input_gate(b"\x00" * (MAX_INPUT_BYTES + 1), **KW)
    assert inp is None and gate.status is InputGateStatus.REJECT and "INPUT_TOO_LARGE" in gate.reason_codes


def test_inverted_xray_rejected():
    """An intensity-inverted radiograph (dark mediastinum, bright corners) must be rejected."""
    normal = synthetic_xray()  # bright lungs on darker background — real CXR appearance
    inverted = (255 - normal).astype(np.uint8)
    inp, gate = run_input_gate(inverted, **KW)
    assert inp is None, "inverted radiograph must not produce an InputContract"
    assert gate.status is InputGateStatus.REJECT
    assert "NON_RADIOGRAPH_INVERTED" in gate.reason_codes


def test_normal_xray_not_flagged_as_inverted():
    """The inversion heuristic must NOT reject a correctly-oriented synthetic X-ray."""
    inp, gate = run_input_gate(synthetic_xray(), **KW)
    assert gate.status is InputGateStatus.ACCEPT, (
        f"normal X-ray was incorrectly rejected: {gate.reason_codes}"
    )
