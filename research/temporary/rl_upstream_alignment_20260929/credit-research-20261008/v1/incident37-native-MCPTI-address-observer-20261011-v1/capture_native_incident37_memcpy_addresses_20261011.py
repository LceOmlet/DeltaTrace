"""Observe original incident update's memcpy addresses via installed MCPTI.

Compose the unchanged GEMM and post-exit CPU shard observers. No tensor value
reads before update, no extra GPU copies, model calls, or synchronization.
Runtime callback timestamps describe host API calls, not GPU completion order.
This diagnostic is outside every production launch path and is not a repair.
"""
import ctypes
import functools
import hashlib
import json
from pathlib import Path
import sys
import time

import capture_native_incident37_postfailure_shard_20261011 as post

POST_SHA = "a5686e51dedbba9fab2039378b6acb1a42a91f3a7292ab4d5c38f530a73dc889"
LIBRARY = Path("/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/"
               "textcraft-native-MCPTI-address-observer-20261011-v3/memcpy_callback.so")
LIBRARY_SHA = "b2ac3a4562aa55dda525476d8e1ae0a4250aa676880aa2a407aefb08c34d18b7"


def make_observed_worker(native_factory):
    import ray
    actor = original_post_factory(native_factory)
    native_class = actor.__ray_metadata__.modified_class.__ray_actor_class__
    native_update = native_class.update_actor

    class NativeMemcpyAddressesObserved(native_class):
        @functools.wraps(native_update)
        def update_actor(self, data):
            output = self._diagnostic_async_head_output
            source_metadata = []
            names = {id(m): n for n, m in self.actor_module_fsdp.named_modules()}
            for module in self.actor_module_fsdp.modules():
                get_state = getattr(module, "_get_fsdp_state", None)
                group = get_state()._fsdp_param_group if get_state else None
                if group is None:
                    continue
                for p in group.fsdp_params:
                    info = p._module_info
                    name = names.get(id(info.module), "")
                    if name.endswith("lm_head") and info.param_name == "weight":
                        value = p._sharded_param_data
                        source_metadata.append(dict(
                            parameter=name + ".weight", device=str(value.device),
                            dtype=str(value.dtype), shape=list(value.shape),
                            pointer=value.data_ptr(), bytes=value.numel() * value.element_size(),
                            shard_rank=p.mesh_info.shard_mesh_rank,
                            padded_shard_shape=list(p.padded_sharded_param_size)))

            assert hashlib.sha256(LIBRARY.read_bytes()).hexdigest() == LIBRARY_SHA
            lib = ctypes.CDLL(str(LIBRARY))
            lib.dt_diagnostic_start.argtypes = [ctypes.c_char_p]
            lib.dt_diagnostic_start.restype = ctypes.c_int
            lib.dt_diagnostic_stop.restype = ctypes.c_int
            trace = output / f"rank{self.rank}-native-memcpy-addresses.jsonl"
            start_status = lib.dt_diagnostic_start(str(trace).encode())
            if start_status != 0:
                raise RuntimeError(f"MCPTI diagnostic subscription failed: {start_status}")
            try:
                return native_update(self, data)
            finally:
                failure = sys.exc_info()[1]
                stop_status = lib.dt_diagnostic_stop()
                report = dict(
                    rank=self.rank, unix=time.time(), source=__file__,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    library=str(LIBRARY), library_sha256=LIBRARY_SHA,
                    start_status=start_status, stop_status=stop_status,
                    native_update_error=repr(failure) if failure is not None else None,
                    actual_flat_source_metadata_before_update=source_metadata,
                    original_exception_preserved=True, added_GPU_value_reads=0,
                    added_GPU_copies=0, added_synchronizations=0, extra_model_calls=0,
                    production_changes=0, new_numerical_tolerance=None)
                if trace.exists():
                    with trace.open() as f:
                        report["callback_summary"] = json.loads(next(f))
                    report["trace"] = dict(path=str(trace), bytes=trace.stat().st_size,
                        sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
                (output / f"rank{self.rank}-native-memcpy-addresses.json").write_text(
                    json.dumps(report, indent=2) + "\n")

    return ray.remote(NativeMemcpyAddressesObserved)


original_post_factory = post.make_observed_worker


def main():
    assert hashlib.sha256(Path(post.__file__).read_bytes()).hexdigest() == POST_SHA
    post.make_observed_worker = make_observed_worker
    try:
        post.main()
    finally:
        post.make_observed_worker = original_post_factory


if __name__ == "__main__":
    main()
