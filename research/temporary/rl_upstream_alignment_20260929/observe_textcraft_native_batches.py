"""Optional, read-only observations of the original TextCraft collect output.

Call ``observe(owner, output)`` only after the native collect has returned
normally, outside the environment's try/except. No parser, reward, DT call,
forward or optimizer is invoked here. An unset output-directory flag is a no-op.

VERL protocol SHA256 2ae51f003f72d6ad0f288d5e2d8e94ad172a69422239a8612ba6d1d9dffeb4aa
has a save side effect: save_to_disk -> pickle.dump -> __getstate__ (260--269)
rebinds self.batch to contiguous/consolidated storage for TensorDict >= 0.5.
Official select(deepcopy=True) (399--432) copies metadata only. Therefore this
module additionally uses the official TensorDict.clone(recurse=False) to give
the snapshot its own container before calling its original save_to_disk.
Original tensors remain shared only for reading; the original output is not
rebound, moved, or changed.

JSON omits native prompt-prefix lists. The output snapshot preserves the
original output arrays, including the owner's existing truncation. When present,
the owner's existing credit_responses snapshot also preserves native per-turn
prompt/response arrays. A missing credit_responses object is recorded explicitly;
no unsaved prefix is reconstructed. JSON retains existing native response IDs,
spans, prefix/transport lengths, and handler values; missing fields remain missing.
It does not infer validity, truncation, executed actions, or exception results.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time
import uuid


OUTPUT_DIRECTORY_ENV = "DT_TEXTCRAFT_NATIVE_OBSERVATION_DIR"
_LOG = logging.getLogger(__name__)


def _scalar(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if item is not None:
        value = item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported original scalar type: {type(value).__name__}")


def _turn_metadata(record):
    missing = []
    result = {}
    for key in ("start", "end"):
        result[key] = _scalar(record[key]) if key in record else None
        if key not in record:
            missing.append(key)
    for source, target in (("native_prompt_ids", "native_prompt_ids_length"),
                           ("output_ids", "transport_output_ids_length")):
        result[target] = len(record[source]) if source in record else None
        if source not in record:
            missing.append(source)
    result["native_response_ids"] = (list(record["native_response_ids"])
                                     if "native_response_ids" in record else None)
    if "native_response_ids" not in record:
        missing.append("native_response_ids")
    result["missing_fields"] = missing
    return result


def _metadata(owner, output):
    missing = []
    handlers = getattr(owner, "handlers", None)
    records = getattr(owner, "records", None)
    if handlers is None:
        missing.append("owner.handlers")
        handlers = []
    if records is None:
        missing.append("owner.records")
        records = []
    batch = output.batch
    has_rounds = batch is not None and "task_rounds" in batch.keys()
    rows = []
    for index, handler in enumerate(handlers):
        row_missing = []
        row = {}
        for key in ("item_id", "score", "done"):
            row[key] = _scalar(getattr(handler, key)) if hasattr(handler, key) else None
            if not hasattr(handler, key):
                row_missing.append(key)
        row["task_rounds"] = _scalar(batch["task_rounds"][index]) if has_rounds else None
        if not has_rounds:
            row_missing.append("output.batch.task_rounds")
        row["turns"] = ([_turn_metadata(record) for record in records[index]]
                        if index < len(records) else None)
        if index >= len(records):
            row_missing.append("owner.records[index]")
        row["missing_fields"] = row_missing
        rows.append(row)
    return {"handlers": rows, "missing_fields": missing}


def _snapshot(data):
    snapshot = data.select(deepcopy=True)
    if data.batch is not None:
        snapshot.batch = data.batch.clone(recurse=False)
    return snapshot


def observe(owner, output):
    """Save existing native objects without changing them; report log failures.

    Returns None when disabled, or a diagnostic status dictionary. Failures are
    caught here so they cannot become environment failures or change score/done.
    This module does not install or invoke itself in the production collector.
    """
    directory = os.environ.get(OUTPUT_DIRECTORY_ENV)
    if not directory:
        return None
    destination = None
    try:
        metadata = _metadata(owner, output)
        destination = Path(directory) / f"batch-{time.time_ns()}-{uuid.uuid4().hex}"
        destination.mkdir(parents=True, exist_ok=False)
        _snapshot(output).save_to_disk(str(destination / "output.pkl"))
        credit = getattr(owner, "credit_responses", None)
        if credit is not None:
            _snapshot(credit).save_to_disk(str(destination / "credit_responses.pkl"))
        metadata["snapshots"] = {
            "output": "output.pkl",
            "credit_responses": "credit_responses.pkl" if credit is not None else None,
        }
        if credit is None:
            metadata["missing_fields"].append("owner.credit_responses")
        (destination / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {"status": "saved", "directory": str(destination)}
    except Exception as error:
        _LOG.exception("TextCraft diagnostic_log_failed; original collect output is unchanged")
        return {
            "status": "diagnostic_log_failed",
            "directory": str(destination) if destination is not None else None,
            "error_type": type(error).__name__,
            "error": str(error),
        }
