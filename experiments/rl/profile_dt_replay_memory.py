"""Add passive allocation diagnostics to the existing exact-32k capacity test.

Forward hooks report live capture storage at the last decoder's projection
boundaries; they do not modify model outputs, finite rules, or execution order.
Run with verify_dt_context_capacity.py's arguments, redirecting stdout to a log.
"""
import json
import os
import runpy
import time

import torch
import native_attention_capture
import qwen35_decoder_finite
import qwen35_dense_finite_runner  # bind the explicitly staged owner before producer imports
from accelerated.qwen35 import qwen35_code_local_capture

# Measure the existing synchronous capture copies at the first GDN replay.
# This observes the owner implementation without substituting tensor transport.
native_gdn_copy = qwen35_code_local_capture.NativeGDNCapture.copy
native_gdn_exit = qwen35_code_local_capture.NativeGDNCapture.__exit__


def timed_gdn_copy(self, value):
    if value is None or self.module.layer_idx != 30:
        return native_gdn_copy(self, value)
    start = time.perf_counter()
    result = native_gdn_copy(self, value)
    records = getattr(self, '_copy_cost_records', [])
    records.append(dict(shape=list(value.shape), stride=list(value.stride()),
                        bytes=value.numel()*value.element_size(),
                        seconds=time.perf_counter()-start))
    self._copy_cost_records = records
    return result


def report_gdn_exit(self, *exc):
    try:
        return native_gdn_exit(self, *exc)
    finally:
        if hasattr(self, '_copy_cost_records'):
            print('GDN_CAPTURE_COPY_COST', json.dumps(self._copy_cost_records), flush=True)


qwen35_code_local_capture.NativeGDNCapture.copy = timed_gdn_copy
qwen35_code_local_capture.NativeGDNCapture.__exit__ = report_gdn_exit

if os.environ.get('DT_TEST_OFFLOAD_REPLAY_MIXER') == '1':
    native_init = qwen35_dense_finite_runner.Qwen35DenseFiniteRunner.__init__
    def offloaded_init(self, *args, **kwargs):
        kwargs['offload_replay_mixer'] = True
        kwargs['gdn_head_batch_size'] = 8
        kwargs['compile_gdn_scalar_rules'] = True
        return native_init(self, *args, **kwargs)
    qwen35_dense_finite_runner.Qwen35DenseFiniteRunner.__init__ = offloaded_init


active = {}


def storage_report():
    seen = set()
    groups = {}
    for group, capture in active.items():
        sizes = {}
        for name, value in capture.values.items():
            if not isinstance(value, torch.Tensor) or not value.is_cuda:
                continue
            storage = value.untyped_storage()
            identity = (storage.data_ptr(), storage.nbytes())
            if identity not in seen:
                sizes[name] = storage.nbytes()
                seen.add(identity)
        groups[group] = sizes
    free, total = torch.cuda.mem_get_info()
    return dict(device_used_bytes=total-free, captures=groups,
                unique_capture_bytes=sum(n for group in groups.values() for n in group.values()))


native_attention_enter = native_attention_capture.NativeAttentionCapture.__enter__
native_attention_exit = native_attention_capture.NativeAttentionCapture.__exit__


def attention_enter(self):
    active['attention'] = self
    return native_attention_enter(self)


def attention_exit(self, *exc):
    try:
        return native_attention_exit(self, *exc)
    finally:
        active.pop('attention', None)


native_decoder_enter = qwen35_decoder_finite.NativeDecoderCapture.__enter__
native_decoder_exit = qwen35_decoder_finite.NativeDecoderCapture.__exit__


def decoder_enter(self):
    value = native_decoder_enter(self)
    active['decoder'] = self
    def record(phase):
        if self.layer.block_type == 'full_attention' and self.layer.self_attn.layer_idx == 31:
            print('REPLAY_MEMORY', json.dumps(dict(phase=phase, **storage_report())), flush=True)
    for name in ('gate_proj', 'up_proj', 'down_proj'):
        module = getattr(self.layer.mlp, name)
        self.handles.append(module.register_forward_pre_hook(lambda *_a, name=name: record(name+'_before')))
        self.handles.append(module.register_forward_hook(lambda *_a, name=name: record(name+'_after')))
    return value


def decoder_exit(self, *exc):
    try:
        return native_decoder_exit(self, *exc)
    finally:
        active.pop('decoder', None)


native_attention_capture.NativeAttentionCapture.__enter__ = attention_enter
native_attention_capture.NativeAttentionCapture.__exit__ = attention_exit
qwen35_decoder_finite.NativeDecoderCapture.__enter__ = decoder_enter
qwen35_decoder_finite.NativeDecoderCapture.__exit__ = decoder_exit
runpy.run_path(os.environ['DT_ROOT']+'/experiments/rl/verify_dt_context_capacity.py', run_name='__main__')
