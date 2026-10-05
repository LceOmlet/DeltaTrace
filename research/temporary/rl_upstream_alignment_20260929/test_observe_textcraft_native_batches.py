"""CPU observer checks using saved actual TextCraft IDs, never a fake DataProto.

The metadata checks run with stdlib alone. The original DataProto save roundtrip
requires the existing upstream dependencies and is explicitly skipped when they
are unavailable; a stand-in serializer is not accepted as evidence for it.
"""

import importlib.util
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "observe_textcraft_native_batches", HERE / "observe_textcraft_native_batches.py")
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


def _saved_sample():
    records = json.loads((HERE / "textcraft-degradation-20261005" /
                          "checkpoint-matched-records.json").read_text(encoding="utf-8"))
    return records[0]["minimum_records"][0]["data"]["samples"][0]


def _saved_turn():
    sample = _saved_sample()
    # The real receipt identifies this current response with source_start:end.
    # trace context/compute counts describe the broader finite readout, not this
    # response boundary; they must not be used to reconstruct it.
    start, end = sample["source_start"], sample["source_end"]
    ids = sample["selected_input_ids"]
    assert 0 <= start < end <= len(ids)
    return {
        "native_prompt_ids": ids[:start],
        "native_response_ids": ids[start:end],
        "output_ids": ids[start:end],
        "start": start,
        "end": end,
    }


def _original_dataproto():
    DataProto = pytest.importorskip("verl").DataProto
    source = Path(inspect.getsourcefile(DataProto))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "2ae51f003f72d6ad0f288d5e2d8e94ad172a69422239a8612ba6d1d9dffeb4aa"
    ), "roundtrip evidence must use the actual original VERL protocol"
    return DataProto


def test_metadata_preserves_saved_ids_and_spans_without_prefix_copy():
    turn = _saved_turn()
    before = json.dumps(turn, sort_keys=True)
    metadata = observer._turn_metadata(turn)
    assert metadata["native_response_ids"] == turn["native_response_ids"]
    assert metadata["native_prompt_ids_length"] == len(turn["native_prompt_ids"])
    assert metadata["transport_output_ids_length"] == len(turn["output_ids"])
    assert metadata["start"] == turn["start"]
    assert metadata["end"] == turn["end"]
    assert "native_prompt_ids" not in metadata
    assert metadata["missing_fields"] == []
    assert json.dumps(turn, sort_keys=True) == before


def test_missing_fields_are_not_zero_or_inferred():
    metadata = observer._turn_metadata({})
    assert metadata["native_response_ids"] is None
    assert metadata["native_prompt_ids_length"] is None
    assert metadata["transport_output_ids_length"] is None
    assert set(metadata["missing_fields"]) == {
        "start", "end", "native_prompt_ids", "native_response_ids", "output_ids"}
    assert "is_action_valid" not in metadata
    assert "truncated" not in metadata


def test_disabled_does_not_access_objects(monkeypatch):
    monkeypatch.delenv(observer.OUTPUT_DIRECTORY_ENV, raising=False)
    assert observer.observe(object(), object()) is None


def test_original_dataproto_roundtrip_preserves_input(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pytest.importorskip("tensordict")
    pytest.importorskip("ray")
    pytest.importorskip("pandas")
    DataProto = _original_dataproto()

    sample = _saved_sample()
    ids = torch.tensor([sample["selected_input_ids"]], dtype=torch.long, device="cpu")
    data = DataProto.from_dict(
        tensors={"input_ids": ids},
        non_tensors={"traj_uid": np.array([sample["traj_uid"]], dtype=object)},
        meta_info={"source_line": 727},
    )
    owner = SimpleNamespace(handlers=[], records=[], credit_responses=data)
    original_batch = data.batch
    original_tensor = data.batch["input_ids"]
    original_metadata = data.non_tensor_batch["traj_uid"]
    monkeypatch.setenv(observer.OUTPUT_DIRECTORY_ENV, str(tmp_path))
    status = observer.observe(owner, data)
    assert status["status"] == "saved"
    assert data.batch is original_batch
    assert data.batch["input_ids"] is original_tensor
    assert data.batch["input_ids"].data_ptr() == ids.data_ptr()
    assert torch.equal(data.batch["input_ids"], ids)
    assert data.non_tensor_batch["traj_uid"] is original_metadata
    assert data.meta_info == {"source_line": 727}
    saved = DataProto.load_from_disk(str(Path(status["directory"]) / "output.pkl"))
    assert torch.equal(saved.batch["input_ids"], ids)
    assert saved.non_tensor_batch["traj_uid"].tolist() == [sample["traj_uid"]]
    assert saved.meta_info == data.meta_info


def test_original_save_failure_is_diagnostic_only(tmp_path, monkeypatch, caplog):
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensordict")
    pytest.importorskip("ray")
    pytest.importorskip("pandas")
    DataProto = _original_dataproto()

    sample = _saved_sample()
    ids = torch.tensor([sample["selected_input_ids"]], dtype=torch.long, device="cpu")
    data = DataProto.from_dict(tensors={"input_ids": ids})
    original_batch = data.batch
    owner = SimpleNamespace(handlers=[], records=[])
    monkeypatch.setenv(observer.OUTPUT_DIRECTORY_ENV, str(tmp_path))

    def fail_save(self, path):
        raise OSError("diagnostic save failed")

    monkeypatch.setattr(DataProto, "save_to_disk", fail_save)
    status = observer.observe(owner, data)
    assert status["status"] == "diagnostic_log_failed"
    assert data.batch is original_batch
    assert torch.equal(data.batch["input_ids"], ids)
    assert "diagnostic_log_failed" in caplog.text
