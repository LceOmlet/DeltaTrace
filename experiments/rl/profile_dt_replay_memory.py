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
import native_dense_attention_capture
import qwen35_decoder_finite
import qwen35_dense_finite_runner  # bind the explicitly staged owner before producer imports
from accelerated.qwen35 import qwen35_code_local_capture

# Measure the existing synchronous capture copies at the first GDN replay.
# This observes the owner implementation without substituting tensor transport.
native_gdn_copy = qwen35_code_local_capture.NativeGDNCapture.copy
native_gdn_exit = qwen35_code_local_capture.NativeGDNCapture.__exit__


def timed_gdn_copy(self, value, **kwargs):
    if value is None or self.module.layer_idx != 30:
        return native_gdn_copy(self, value, **kwargs)
    start = time.perf_counter()
    result = native_gdn_copy(self, value, **kwargs)
    records = getattr(self, '_copy_cost_records', [])
    records.append(dict(shape=list(value.shape), stride=list(value.stride()),
                        bytes=value.numel()*value.element_size(), destination=str(result.device),
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
    return dict(device_used_bytes=total-free, allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(), captures=groups,
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
        mixer = self.layer.self_attn if self.layer.block_type == 'full_attention' else self.layer.linear_attn
        if mixer.layer_idx in (30, 31):
            print('REPLAY_MEMORY', json.dumps(dict(layer=mixer.layer_idx, phase=phase, **storage_report())), flush=True)
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


if os.environ.get('DT_PROFILE_DENSE_ALIASES') == '1':
    alias_records = []
    original_retain_dense = native_dense_attention_capture.NativeDenseAttentionCapture.retain_dense
    original_alias_exit = native_dense_attention_capture.NativeDenseAttentionCapture.__exit__

    def compare_dense_capture(self, name, value):
        # Observe the same actual operand at the native FA call. The previous
        # route copies it independently; the new route reuses its verified
        # interface view. Synchronize each measurement and compare outside it.
        torch.cuda.synchronize()
        start = time.perf_counter()
        previous = native_attention_capture.copy_capture_tensor(value, self.destination,
            copy=self.copy_tensors, preserve_strides=self.preserve_strides, pinned_host=self.pinned_host)
        torch.cuda.synchronize()
        previous_seconds = time.perf_counter()-start
        start = time.perf_counter()
        original_retain_dense(self, name, value)
        torch.cuda.synchronize()
        current_seconds = time.perf_counter()-start
        actual = self.values['dense_'+name]
        assert torch.equal(previous, actual)
        assert previous.stride() == actual.stride()
        alias_records.append(dict(name=name, bytes=value.numel()*value.element_size(),
            previous_seconds=previous_seconds, current_seconds=current_seconds,
            exact_values_and_strides=True, alias=self.dense_aliases.get('dense_'+name)))
        print('DENSE_CAPTURE_COMPARE', json.dumps(alias_records[-1]), flush=True)

    def finish_alias_capture(self, *exc):
        original_alias_exit(self, *exc)
        if exc[0] is None:
            result = dict(scope='Same actual first-FA 32k replay operands, original actor and sleeping vLLM; capture-only comparison',
                status='observed_requested_copies', records=alias_records)
            Path(os.environ['DT_PROFILE_OUTPUT']).write_text(json.dumps(result, indent=2)+'\n')
            raise DiagnosticStop()

    native_dense_attention_capture.NativeDenseAttentionCapture.retain_dense = compare_dense_capture
    native_dense_attention_capture.NativeDenseAttentionCapture.__exit__ = finish_alias_capture


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
        # Older owners read the input solely for its shape. Retain the actual
        # tensor for this comparison, outside the timed propagation calls.
        native_init = qwen35_code_local_capture.NativeGDNCapture.__init__
        def comparison_capture_init(self,*a,**kw):
            kw['capture_input']=True
            native_init(self,*a,**kw)
        qwen35_code_local_capture.NativeGDNCapture.__init__=comparison_capture_init

    def observed_gdn(module, *args, **kwargs):
        if os.environ.get('DT_PROFILE_SELECTIVE_CAPTURE') == '1':
            # Compare the same captured native endpoint tensors, with the same
            # GPU tensors retained throughout both routes. Only their source
            # for the existing head-group restore changes. Copies and output
            # checks stay outside finite propagation timings.
            host_endpoints = dict(args[1])
            copied_bytes = 0
            torch.cuda.synchronize()
            copy_start = time.perf_counter()
            for name, tensor in args[1].items():
                if tensor.is_cuda:
                    host_endpoints[name] = native_attention_capture.copy_capture_tensor(
                        tensor, 'cpu', preserve_strides=True, pinned_host=True)
                    copied_bytes += tensor.numel()*tensor.element_size()
            torch.cuda.synchronize()
            copy_seconds = time.perf_counter()-copy_start
            if not copied_bytes:
                raise ValueError('Selective comparison requires actual GPU-retained endpoint captures.')
            reference = None
            routes = [('gpu_cold', args[1]), ('cpu_cold', host_endpoints),
                      ('cpu_warm_0', host_endpoints), ('gpu_warm_0', args[1]),
                      ('gpu_warm_1', args[1]), ('cpu_warm_1', host_endpoints)]
            for name, endpoints in routes:
                operands = (dict(args[0]), dict(endpoints), *args[2:])
                torch.cuda.synchronize()
                before = storage_report()
                start = time.perf_counter()
                value = original_gdn(module, *operands, **kwargs)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                actual = value[0].detach().cpu()
                if reference is None:
                    reference = actual
                assert torch.equal(reference, actual), name
                records.append(dict(layer=module.layer_idx, name=name, seconds=elapsed,
                    exact_output=True, before=before))
                print('GDN_SELECTIVE_COMPARE', json.dumps(records[-1]), flush=True)
                del value, actual, operands
            profile_output.write_text(json.dumps(dict(
                scope='Same actual 32k first GDN captures and upstream; retained GPU tensors remain alive for both routes; not full DT capacity or training acceptance',
                status='observed_requested_phases', copied_bytes=copied_bytes,
                d2h_copy_seconds=copy_seconds, records=records), indent=2)+'\n')
            raise DiagnosticStop()
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
                options=dict(kwargs)
                if function is previous_gdn:
                    options.pop('input_shape',None)
                value = function(module,*operands,**options)
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
    print('DIAGNOSTIC_STOP: requested phases recorded; remaining DT/PPO intentionally not run', flush=True)
