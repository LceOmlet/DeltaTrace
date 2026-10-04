"""Diagnostic-only snapshots of original endpoint/replay base Linear inputs.

Use inside one already scheduled shared_warm runner.attribute call. No forward
is added or replaced. CPU snapshot/comparison costs invalidate speed comparison
for the instrumented call; this module never saves weights or model outputs.
"""
from __future__ import annotations

import math
import time
from typing import Any

import torch


class NativeProjectionInputAudit:
    """Observe the first two native calls to each original MLP base Linear."""

    def __init__(self, layers: Any):
        self.layers = list(layers)
        self.handles = []
        self.snapshots = {}
        self.records = []
        self.active = False
        self.snapshot_bytes = 0
        self.peak_snapshot_bytes = 0
        self.copy_seconds = 0.0
        self.comparison_seconds = 0.0
        self.copied_input_bytes = 0

    @staticmethod
    def _info(value):
        return {
            "shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "device": str(value.device),
            "input_bytes": value.numel() * value.element_size(),
        }

    @staticmethod
    def _max_difference(first, second):
        # Chunked FP64 subtraction keeps comparison temporaries bounded even
        # when the actual suffix is large. These are CPU snapshots only.
        first, second = first.reshape(-1), second.reshape(-1)
        maximum = 0.0
        for start in range(0, first.numel(), 1 << 20):
            difference = (first[start:start + (1 << 20)].to(torch.float64)
                          - second[start:start + (1 << 20)].to(torch.float64)).abs()
            value = difference.max().item()
            if not math.isfinite(value):
                return None, "nonfinite_difference"
            maximum = max(maximum, value)
        return maximum, "measured"

    def _observe(self, base, key, record, args, kwargs):
        record["call_count"] += 1
        call = record["call_count"]
        value = args[0] if args else kwargs.get("input")
        if not isinstance(value, torch.Tensor):
            # Preserve the original call; expose the diagnostic limitation.
            record["diagnostic_errors"].append({"call": call, "input_type": type(value).__name__})
            return None
        info = self._info(value)
        weight = getattr(base, "weight", None)
        info["base_weight_metadata"] = ({
            "shape": list(weight.shape), "stride": list(weight.stride()),
            "dtype": str(weight.dtype), "device": str(weight.device),
        } if isinstance(weight, torch.Tensor) else None)
        if call > 2:
            record.setdefault("extra_calls", []).append(info)
            return None
        started = time.perf_counter()
        snapshot = value.detach().to(device="cpu", copy=True, non_blocking=False)
        elapsed = time.perf_counter() - started
        info["cpu_snapshot_seconds"] = elapsed
        self.copy_seconds += elapsed
        self.copied_input_bytes += info["input_bytes"]
        if call == 1:
            record["root_input"] = info
            record["comparison_status"] = "not_replayed"
            self.snapshots[key] = snapshot
            self.snapshot_bytes += info["input_bytes"]
            self.peak_snapshot_bytes = max(self.peak_snapshot_bytes, self.snapshot_bytes)
        else:
            record["replay_input"] = info
            first = self.snapshots.pop(key, None)
            if first is None:
                record["comparison_status"] = "missing_first_snapshot"
                return None
            self.snapshot_bytes -= record["root_input"]["input_bytes"]
            self.peak_snapshot_bytes = max(self.peak_snapshot_bytes,
                self.snapshot_bytes + record["root_input"]["input_bytes"] + info["input_bytes"])
            started = time.perf_counter()
            same_shape = tuple(first.shape) == tuple(snapshot.shape)
            same_dtype = first.dtype == snapshot.dtype
            same_stride = record["root_input"]["stride"] == info["stride"]
            same_device = record["root_input"]["device"] == info["device"]
            same_weight_metadata = (record["root_input"]["base_weight_metadata"]
                                    == info["base_weight_metadata"])
            equal = bool(torch.equal(first, snapshot)) if same_shape else False
            if same_shape:
                maximum, maximum_status = self._max_difference(first, snapshot)
            else:
                maximum, maximum_status = None, "shape_mismatch"
            record.update({
                "shape_equal": same_shape, "dtype_equal": same_dtype,
                "stride_equal": same_stride, "device_equal": same_device,
                "base_weight_metadata_equal": same_weight_metadata,
                "torch_equal": equal, "inputs_equal": same_shape and same_dtype and equal,
                "maximum_absolute_difference": maximum,
                "maximum_difference_status": maximum_status,
                "comparison_status": "compared",
                "cpu_comparison_seconds": time.perf_counter() - started,
            })
            self.comparison_seconds += record["cpu_comparison_seconds"]
            del first, snapshot
        return None

    def _cleanup(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.snapshots.clear()
        self.snapshot_bytes = 0
        self.active = False

    def __enter__(self):
        if self.active or self.records:
            raise RuntimeError("Create one fresh diagnostic audit per runner.attribute call")
        self.active = True
        try:
            for index, layer in enumerate(self.layers):
                for name in ("gate_proj", "up_proj", "down_proj"):
                    projection = getattr(layer.mlp, name)
                    getter = getattr(projection, "get_base_layer", None)
                    if callable(getter):
                        base = getter()
                    elif isinstance(projection, torch.nn.Linear):
                        base = projection
                    else:
                        raise TypeError(f"No original base Linear for layer {index} {name}")
                    key = (index, name)
                    record = {
                        "layer": index, "projection": name,
                        "original_projection_type": type(projection).__module__ + "." + type(projection).__qualname__,
                        "original_base_type": type(base).__module__ + "." + type(base).__qualname__,
                        "call_count": 0, "comparison_status": "not_called", "diagnostic_errors": [],
                    }
                    self.records.append(record)
                    def before(_module, args, kwargs, key=key, record=record):
                        return self._observe(_module, key, record, args, kwargs)
                    self.handles.append(base.register_forward_pre_hook(before, with_kwargs=True))
            return self
        except BaseException:
            self._cleanup()
            raise

    def __exit__(self, kind, error, traceback):
        self._cleanup()
        return False

    def report(self):
        compared = [record for record in self.records if record["comparison_status"] == "compared"]
        return {
            "scope": "One original shared_warm endpoint root and reverse layer replay; first/second base calls only",
            "expected_projections": len(self.layers) * 3,
            "observed_projections": len(self.records),
            "compared_projections": len(compared),
            "equal_projections": sum(record["inputs_equal"] for record in compared),
            "equal_stride_projections": sum(record["stride_equal"] for record in compared),
            "equal_device_projections": sum(record["device_equal"] for record in compared),
            "equal_base_weight_metadata_projections": sum(record["base_weight_metadata_equal"] for record in compared),
            "all_expected_inputs_equal": len(compared) == len(self.layers) * 3
                and all(record["call_count"] == 2 and record["inputs_equal"] for record in self.records),
            "original_forward_calls_added": 0,
            "hooks_remaining": len(self.handles), "snapshots_remaining": len(self.snapshots),
            "copied_input_bytes": self.copied_input_bytes,
            "peak_snapshot_logical_bytes": self.peak_snapshot_bytes,
            "cpu_snapshot_seconds": self.copy_seconds,
            "cpu_comparison_seconds": self.comparison_seconds,
            "timing_scope": "Blocking CPU input copies include preceding device waits; CPU comparisons are diagnostic overhead, never a DT speed measurement",
            "projections": self.records,
        }
