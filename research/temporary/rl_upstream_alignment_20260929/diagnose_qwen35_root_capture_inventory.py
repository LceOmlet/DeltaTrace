"""Diagnostic metadata for one original Qwen3.5 root, with owner captures.

The caller supplies fresh, unchanged owner capture objects and schedules the
existing root. This module adds no forward, tensor copy, offload or finite rule.
It releases captures at each layer exit, so sums are inventory upper estimates,
not a measured simultaneous root-tape allocation or an attribution speed test.
"""
from __future__ import annotations

from contextlib import ExitStack
import inspect
import sys
from typing import Any, Callable


class NativeRootCaptureInventory:
    """Record the first original call to each layer while this scope is active.

    ``capture_factory(index, layer, args, kwargs)`` returns ``(decoder, mixer)``
    or ``(decoder, mixer, external_tensors)``. The optional mapping contains
    existing owner cache tensors, or a callback returning that mapping after
    layer/context exit; only exact storage aliases are excluded.
    The factory owns all capture options, including destination and suffix cut.
    Scope this around the existing endpoint root, not prefix-bank preparation.
    """

    def __init__(self, layers: Any, capture_factory: Callable):
        self.layers = list(layers)
        self.capture_factory = capture_factory
        self.records = []
        self.seen_calls = [0] * len(self.layers)
        self.handles = []
        self.active = None
        self.entered = False

    @staticmethod
    def _storage(value):
        storage = value.untyped_storage()
        return (str(value.device), storage.data_ptr(), storage.nbytes())

    @staticmethod
    def _info(value):
        return {
            "shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "device": str(value.device),
            "storage_offset": value.storage_offset(),
            "logical_bytes": value.numel() * value.element_size(),
            "storage_bytes": value.untyped_storage().nbytes(),
        }

    @staticmethod
    def _source_paths(capture):
        paths = []
        for cls in type(capture).__mro__:
            try:
                path = inspect.getfile(cls)
            except (TypeError, OSError):
                continue
            if path not in paths:
                paths.append(path)
        return paths

    def _inventory(self, index, dc, mc, external, status):
        groups = {}
        external_groups = {}
        errors = []
        if callable(external):
            try:
                external = external()
            except Exception as error:
                errors.append({"field": "external", "error": repr(error)})
                external = {}
        for name, value in external.items():
            if value is None or not hasattr(value, "untyped_storage"):
                continue
            try:
                external_groups.setdefault(self._storage(value), []).append(name)
            except Exception as error:
                errors.append({"field": "external." + name, "error": repr(error)})
        fields = []
        for scope, values in (("decoder", dc.values), ("mixer", mc.values),
                              ("endpoints", getattr(mc, "endpoints", {}))):
            for name, value in values.items():
                field = {"field": scope + "." + name}
                if value is None or not hasattr(value, "untyped_storage"):
                    field["value_type"] = type(value).__name__
                else:
                    try:
                        key = self._storage(value)
                        group = groups.setdefault(key, {"group": len(groups),
                            "device": key[0], "storage_bytes": key[2],
                            "fields": [], "external_aliases": external_groups.get(key, [])})
                        group["fields"].append(field["field"])
                        field.update(self._info(value), storage_group=group["group"])
                    except Exception as error:
                        errors.append({"field": field["field"], "error": repr(error)})
                fields.append(field)
        storage = list(groups.values())
        return {
            "layer": index, "block_type": self.layers[index].block_type,
            "status": status, "metadata_read_after_context_exit": True,
            "decoder_type": type(dc).__module__ + "." + type(dc).__qualname__,
            "mixer_type": type(mc).__module__ + "." + type(mc).__qualname__,
            "decoder_source_paths": self._source_paths(dc),
            "mixer_source_paths": self._source_paths(mc),
            "decoder_calls": dict(dc.calls), "mixer_calls": dict(mc.calls),
            "mixer_input_shape": list(mc.input_shape) if getattr(mc, "input_shape", None) is not None else None,
            "gdn_coefficient_start": getattr(mc, "coefficient_start", None),
            "scale": getattr(mc, "scale", None),
            "dense_aliases": dict(getattr(mc, "dense_aliases", {})),
            "external_mapping_supplied": bool(external),
            "fields": fields, "storage_groups": storage, "diagnostic_errors": errors,
            "logical_payload_bytes": sum(f.get("logical_bytes", 0) for f in fields),
            "unique_storage_bytes": sum(g["storage_bytes"] for g in storage),
            "external_alias_storage_bytes": sum(g["storage_bytes"] for g in storage if g["external_aliases"]),
            "storage_without_known_external_alias_bytes": sum(g["storage_bytes"] for g in storage if not g["external_aliases"]),
        }

    def _finish(self, exc_info):
        active = self.active
        if active is None:
            return
        self.active = None
        index, stack, dc, mc, external, post_handle = active
        try:
            stack.__exit__(*exc_info)
            self.records.append(self._inventory(index, dc, mc, external,
                "complete" if exc_info[0] is None else "original_call_raised"))
        finally:
            if post_handle is not None:
                post_handle.remove()
            dc.values.clear()
            mc.values.clear()
            getattr(mc, "endpoints", {}).clear()

    def _start(self, index, layer, args, kwargs):
        self.seen_calls[index] += 1
        if self.seen_calls[index] != 1:
            return None
        if self.active is not None:
            raise RuntimeError("Root inventory expects the original sequential decoder call.")
        captures = self.capture_factory(index, layer, args, kwargs)
        dc, mc = captures[:2]
        external = captures[2] if len(captures) == 3 else {}
        stack = ExitStack()
        self.active = (index, stack, dc, mc, external, None)
        try:
            stack.enter_context(dc)
            stack.enter_context(mc)
            # Register after the owner's decoder output hook: its calls/fields
            # must be complete before metadata is read and capture refs freed.
            def after(_layer, _args, _kwargs, _output):
                self._finish(sys.exc_info())
                return None
            post = layer.register_forward_hook(after, with_kwargs=True, always_call=True)
            self.active = (index, stack, dc, mc, external, post)
        except BaseException:
            self._finish(sys.exc_info())
            raise
        return None

    def __enter__(self):
        if self.entered:
            raise RuntimeError("Root inventory context is single use.")
        self.entered = True
        try:
            for index, layer in enumerate(self.layers):
                def before(module, args, kwargs, index=index):
                    return self._start(index, module, args, kwargs)
                self.handles.append(layer.register_forward_pre_hook(before, with_kwargs=True))
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def __exit__(self, *exc_info):
        try:
            self._finish(exc_info)
        finally:
            for handle in reversed(self.handles):
                handle.remove()
            self.handles.clear()
        return False

    def report(self):
        return {
            "scope": "One original root; original capture metadata only; no tensor copies added by inventory; layer captures released immediately",
            "expected_layers": len(self.layers), "recorded_layers": len(self.records),
            "complete_layers": sum(r["status"] == "complete" for r in self.records),
            "seen_layer_calls": list(self.seen_calls), "original_forward_calls_added": 0,
            "hooks_remaining": len(self.handles), "active_capture_remaining": self.active is not None,
            "sum_per_layer_logical_payload_bytes": sum(r["logical_payload_bytes"] for r in self.records),
            "sum_per_layer_unique_storage_bytes": sum(r["unique_storage_bytes"] for r in self.records),
            "sum_per_layer_known_external_alias_bytes": sum(r["external_alias_storage_bytes"] for r in self.records),
            "sum_per_layer_storage_without_known_external_alias_bytes": sum(r["storage_without_known_external_alias_bytes"] for r in self.records),
            "storage_scope": "Exact aliases are grouped within each live layer only; sums are not measured tape peak/increment. Missing external mapping leaves prefix-cache ownership unclassified.",
            "rows": list(self.records),
        }
