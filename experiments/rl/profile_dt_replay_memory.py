"""Add passive allocation diagnostics to the existing exact-32k capacity test.

Forward hooks report live capture storage at the last decoder's projection
boundaries; they do not modify model outputs, finite rules, or execution order.
Run with verify_dt_context_capacity.py's arguments, redirecting stdout to a log.
"""
import json
import importlib.util
import os
import runpy
import time
from pathlib import Path

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

# A diagnostic may stop as soon as it has observed the requested bottleneck.
# BaseException deliberately bypasses the capacity verifier's ordinary failure
# handler; its finally block still releases distributed resources. Neither its
# partial output nor this profile is reported as a completed capacity check.
class DiagnosticStop(BaseException):
    pass


limit = int(os.environ.get('DT_PROFILE_GDN_LAYERS', '0'))
if limit:
    profile_output = Path(os.environ['DT_PROFILE_OUTPUT'])
    records = []
    original_gdn = qwen35_dense_finite_runner.gdn_finite_pullback
    previous_gdn = None
    if os.environ.get('DT_PROFILE_PREVIOUS_OWNER'):
        previous = Path(os.environ['DT_PROFILE_PREVIOUS_OWNER'])
        def load_previous(name, filename):
            spec = importlib.util.spec_from_file_location(name,previous/filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        capture_owner = load_previous('previous_capture_owner','native_attention_capture.py')
        gdn_owner = load_previous('previous_gdn_owner','qwen35_gdn_finite.py')
        gdn_owner.copy_capture_tensor = capture_owner.copy_capture_tensor
        previous_gdn = gdn_owner.gdn_finite_pullback

    def observed_gdn(module, *args, **kwargs):
        if previous_gdn is not None:
            # Identical real layer captures and upstream; each owner consumes
            # only its own shallow dictionaries. CPU output comparisons occur
            # outside the measured calls and avoid retaining a GPU output.
            reference = None
            for name, function in [('current_cold', original_gdn), ('previous_warm',previous_gdn),
                                   ('current_warm',original_gdn)]:
                operands = (dict(args[0]),dict(args[1]),*args[2:])
                torch.cuda.synchronize()
                start = time.perf_counter()
                value = function(module,*operands,**kwargs)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                actual = value[0].detach().cpu()
                if reference is None:
                    reference = actual
                assert torch.equal(reference,actual), name
                records.append(dict(layer=module.layer_idx,name=name,seconds=elapsed,exact_output=True))
                profile_output.write_text(json.dumps(dict(scope='Same real 32k layer, captures and upstream; isolated transport comparison',
                    status='running',records=records),indent=2)+'\n')
                print('GDN_IN_CONTEXT_COMPARE',json.dumps(records[-1]),flush=True)
                del value,actual,operands
            profile_output.write_text(json.dumps(dict(scope='Partial real 32k DT diagnostic; not capacity or training acceptance',
                status='observed_requested_phases',records=records),indent=2)+'\n')
            raise DiagnosticStop()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            torch.cuda.synchronize()
            start = time.perf_counter()
            value = original_gdn(module, *args, **kwargs)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
        operators = sorted(profile.key_averages(), key=lambda e:e.self_cpu_time_total, reverse=True)
        record = dict(layer=module.layer_idx, seconds=elapsed,
            operators=[dict(name=e.key, calls=e.count, self_cpu_seconds=e.self_cpu_time_total/1e6,
                            inclusive_cpu_seconds=e.cpu_time_total/1e6) for e in operators[:20]])
        records.append(record)
        profile_output.write_text(json.dumps(dict(scope='Partial real 32k DT diagnostic; not capacity or training acceptance',
            status='observed_requested_phases' if len(records)==limit else 'running', records=records), indent=2)+'\n')
        print('GDN_IN_CONTEXT_PROFILE', json.dumps(record), flush=True)
        if len(records) == limit:
            raise DiagnosticStop()
        return value

    qwen35_dense_finite_runner.gdn_finite_pullback = observed_gdn

try:
    runpy.run_path(os.environ.get('DT_CAPACITY_SCRIPT',os.environ['DT_ROOT']+'/experiments/rl/verify_dt_context_capacity.py'), run_name='__main__')
except DiagnosticStop:
    print('DIAGNOSTIC_STOP: requested GDN phases recorded; remaining DT/PPO intentionally not run', flush=True)
