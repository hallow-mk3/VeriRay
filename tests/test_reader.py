"""Reader tests: fail closed on missing/corrupt weights; preprocessing range; score consistency."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.conftest import synthetic_xray, write_checkpoint_dir
from veriray.reader import (
    PREPROCESSING_ID,
    CheckpointHashMismatchError,
    CheckpointMissingError,
    CheckpointRecordError,
    PreprocessingError,
    Reader,
    preprocess,
)


def test_missing_models_dir_fails_closed(tmp_path: Path):
    with pytest.raises(CheckpointMissingError) as e:
        Reader.load(tmp_path / "nothing_here")
    assert e.value.reason_code == "READER_CHECKPOINT_MISSING"


def test_missing_weights_file_fails_closed(tmp_path: Path):
    d = write_checkpoint_dir(tmp_path / "m")
    (d / "fake-densenet121.pt").unlink()
    with pytest.raises(CheckpointMissingError):
        Reader.load(d)


def test_corrupt_weights_hash_mismatch_fails_closed(tmp_path: Path):
    d = write_checkpoint_dir(tmp_path / "m")
    f = d / "fake-densenet121.pt"
    data = bytearray(f.read_bytes())
    data[len(data) // 2] ^= 0xFF
    f.write_bytes(bytes(data))
    with pytest.raises(CheckpointHashMismatchError) as e:
        Reader.load(d)
    assert e.value.reason_code == "READER_CHECKPOINT_HASH_MISMATCH"


def test_bad_record_fails_closed(tmp_path: Path):
    d = write_checkpoint_dir(tmp_path / "m")
    rec = json.loads((d / "CHECKPOINT.json").read_text())
    rec["finding_index"] = 0  # not Pneumothorax
    (d / "CHECKPOINT.json").write_text(json.dumps(rec))
    with pytest.raises(CheckpointRecordError):
        Reader.load(d)
    (d / "CHECKPOINT.json").write_text("{not json")
    with pytest.raises(CheckpointRecordError):
        Reader.load(d)


def test_unpickle_never_happens_before_hash_check(tmp_path: Path):
    # A hash-matching-but-garbage file is a load failure, a hash-mismatching one is rejected before loading.
    d = write_checkpoint_dir(tmp_path / "m", weights_bytes_override=b"garbage-not-a-torch-file")
    from veriray.reader import CheckpointLoadError

    with pytest.raises(CheckpointLoadError):
        Reader.load(d)


def test_preprocessing_range_and_shape():
    for img, lo, hi in [
        (np.zeros((300, 500), np.float32), -1024.0, -1024.0),
        (np.full((300, 500), 255, np.float32), 1024.0, 1024.0),
    ]:
        t, disp = preprocess(img)
        assert t.shape == (1, 1, 224, 224) and t.dtype == torch.float32
        assert float(t.min()) == pytest.approx(lo, abs=1e-3) and float(t.max()) == pytest.approx(hi, abs=1e-3)
        assert disp.shape == (224, 224)
    t, _ = preprocess(synthetic_xray().astype(np.float32))
    assert -1024.0 <= float(t.min()) and float(t.max()) <= 1024.0
    # Mid-grey 127.5 maps to ~0 (xrv: (2*(x/255)-1)*1024).
    t, _ = preprocess(np.full((224, 224), 127.5, np.float32))
    assert abs(float(t.mean())) < 1e-2
    # Display round-trip and float32 epsilon tolerance at [0, 255] boundaries
    edge = np.full((224, 224), 255.0 + 1e-4, np.float32)
    edge[0, 0] = -1e-4
    _, disp_edge = preprocess(edge)
    assert 0.0 <= float(disp_edge.min()) and float(disp_edge.max()) <= 255.0
    t2, disp2 = preprocess(disp_edge)
    assert t2.shape == (1, 1, 224, 224) and 0.0 <= float(disp2.min()) and float(disp2.max()) <= 255.0


def test_preprocessing_center_crops_long_side():
    img = np.zeros((300, 600), np.float32)
    img[:, :150] = 255  # left quarter is cropped away (center 300x300 keeps columns 150..450)
    t, _ = preprocess(img)
    assert float(t.max()) == pytest.approx(-1024.0, abs=1e-3)


@pytest.mark.parametrize("bad", [np.zeros((3, 3, 3)), np.full((10, 10), 300.0), np.full((10, 10), np.nan), np.zeros((0, 5))])
def test_preprocessing_rejects_bad_arrays(bad):
    with pytest.raises(PreprocessingError):
        preprocess(bad)


def test_score_contract_and_matches_xrv_forward(fake_reader: Reader):
    contract, x, _ = fake_reader.score(synthetic_xray().astype(np.float32))
    assert 0.0 <= contract.score_raw <= 1.0 and contract.score_calibrated_optional is None
    assert contract.checkpoint_id.startswith("fake-test-weights@sha256:") and contract.preprocessing_id == PREPROCESSING_ID
    # Same ops as xrv's own forward with sigmoid and no operating-point normalisation.
    m = fake_reader.model
    m.apply_sigmoid, m.op_threshs = True, None
    with torch.no_grad():
        ref = float(m(x)[0, 3])
    assert contract.score_raw == pytest.approx(ref, abs=1e-6)


def test_reader_module_has_no_network_or_download_path():
    src = (Path(__file__).resolve().parents[1] / "src" / "veriray" / "reader.py").read_text(encoding="utf-8")
    assert "get_weights" not in src and "urllib" not in src and "requests" not in src and "weights=None" in src
