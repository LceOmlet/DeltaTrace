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
compact_copy_records = []
compact_copy_sources = {}
if os.environ.get('DT_PROFILE_COMPACT_COPIES') == '1':
    native_select_time=qwen35_code_local_capture.NativeGDNCapture.select_time
    def observe_selected_time(self,value,dimension=1,*,start=None):
        if value is None or self.module.layer_idx!=30:
            return native_select_time(self,value,dimension,start=start)
        torch.cuda.synchronize();tick=time.perf_counter()
        selected=native_select_time(self,value,dimension,start=start)
        torch.cuda.synchronize()
        compact_copy_sources[id(selected)]=(value,dimension,
            self.coefficient_start if start is None else start,time.perf_counter()-tick)
        return selected
    qwen35_code_local_capture.NativeGDNCapture.select_time=observe_selected_time


def timed_gdn_copy(self, value, **kwargs):
    if value is None or self.module.layer_idx != 30:
        return native_gdn_copy(self, value, **kwargs)
    source=compact_copy_sources.pop(id(value),None)
    if source is not None:
        full,dimension,cut,pack_seconds=source
        destination=kwargs.get('device',self.device)
        if destination is None:destination=self.device
        row=dict(source_shape=list(full.shape),selected_shape=list(value.shape),
            source_bytes=full.numel()*full.element_size(),selected_bytes=value.numel()*value.element_size(),
            destination=str(destination),pack_seconds=pack_seconds,records=[])
        reference=None
        if torch.device(destination).type=='cpu':
            for name,operand in [('full_cold',full),('compact_cold',value),('full_warm_0',full),
                                 ('compact_warm_0',value),('compact_warm_1',value),('full_warm_1',full)]:
                torch.cuda.synchronize();tick=time.perf_counter()
                copied=native_gdn_copy(self,operand,**kwargs)
                torch.cuda.synchronize();elapsed=time.perf_counter()-tick
                actual=copied.narrow(dimension,cut,copied.shape[dimension]-cut) if name.startswith('full') else copied
                if reference is None:reference=actual.clone()
                assert torch.equal(reference,actual)
                row['records'].append(dict(name=name,seconds=elapsed,selected_values_exact=True))
                del copied,actual
        compact_copy_records.append(row)
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


if os.environ.get('DT_PROFILE_PROGRESS')=='1':
    original_decoder_finite=qwen35_dense_finite_runner.decoder_finite_pullback
    def record_decoder_progress(layer,*args,**kwargs):
        mixer=layer.self_attn if layer.block_type=='full_attention' else layer.linear_attn
        print('FINITE_PHASE_ENQUEUE_START',mixer.layer_idx,time.time(),flush=True)
        result=original_decoder_finite(layer,*args,**kwargs)
        print('FINITE_PHASE_ENQUEUE_END',mixer.layer_idx,time.time(),flush=True)
        return result
    qwen35_dense_finite_runner.decoder_finite_pullback=record_decoder_progress


if os.environ.get('DT_PROFILE_ROOT_COPY') == '1':
    original_root_copy=qwen35_dense_finite_runner._copy
    root_copy_recorded=False
    def compare_root_copy(value,device,*,pinned_host=False):
        global root_copy_recorded
        if (not root_copy_recorded and isinstance(value,torch.Tensor) and value.is_cuda
                and str(device)=='cpu' and tuple(value.shape)==(8,32768,4096)):
            root_copy_recorded=True
            rows=[];reference=None
            for name,pinned in [('pageable_cold',False),('pinned_cold',True),('pageable_warm_0',False),
                                ('pinned_warm_0',True),('pinned_warm_1',True),('pageable_warm_1',False)]:
                torch.cuda.synchronize();tick=time.perf_counter()
                copied=original_root_copy(value,device,pinned_host=pinned)
                torch.cuda.synchronize();elapsed=time.perf_counter()-tick
                if reference is None:reference=copied
                assert copied.stride()==reference.stride() and torch.equal(copied,reference)
                rows.append(dict(name=name,seconds=elapsed,exact_values_and_strides=True))
                del copied
            result=dict(scope='Same actual first 2 GiB paired layer input, current actor and sleeping vLLM; capture transfer only',
                bytes=value.numel()*value.element_size(),records=rows)
            Path(os.environ['DT_PROFILE_ROOT_COPY_OUTPUT']).write_text(json.dumps(result,indent=2)+'\n')
            print('ROOT_COPY_COMPARE',json.dumps(result),flush=True)
            if os.environ.get('DT_PROFILE_STOP_AFTER_ROOT_COPY')=='1':raise DiagnosticStop()
        return original_root_copy(value,device,pinned_host=pinned_host)
    qwen35_dense_finite_runner._copy=compare_root_copy


if os.environ.get('DT_PROFILE_FA_SUFFIX') == '1':
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256, RightPaddedLengths
    original_finite_fa = VendorFAFiniteP1BF16D256.__call__

    def compare_fa_suffix(self, operands, scale, layout, activity=None):
        if layout.coefficient_starts is None:
            raise ValueError('Actual runner must supply its common-prefix coefficient bounds.')
        full = RightPaddedLengths(layout.lengths, layout.padded_length, layout._tensor.device)
        records = []
        reference = None
        for name, selected in [('full_cold', full), ('suffix_cold', layout),
                               ('full_warm_0', full), ('suffix_warm_0', layout),
                               ('suffix_warm_1', layout), ('full_warm_1', full)]:
            torch.cuda.synchronize()
            before = storage_report()
            start = time.perf_counter()
            value = original_finite_fa(self, operands, scale, selected, activity)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            actual = {k:v.detach().cpu() for k,v in value.items()}
            if reference is None:
                reference = actual
            for key in reference:
                for index, cut in enumerate(layout.coefficient_starts):
                    assert torch.equal(reference[key][index,:,cut:], actual[key][index,:,cut:]), key
            records.append(dict(name=name, seconds=elapsed, exact_suffix=True, before=before))
            print('FA_SUFFIX_IN_CONTEXT', json.dumps(records[-1]), flush=True)
            del value, actual
        result = dict(scope='Same actual first-FA 32k operands/upstream, original actor and sleeping vLLM; full K/V history retained; not complete DT/PPO acceptance',
            status='observed_requested_operator', coefficient_starts=layout.coefficient_starts,
            valid_lengths=layout.lengths, records=records)
        Path(os.environ['DT_PROFILE_OUTPUT']).write_text(json.dumps(result, indent=2)+'\n')
        raise DiagnosticStop()

    VendorFAFiniteP1BF16D256.__call__ = compare_fa_suffix


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
        if os.environ.get('DT_PROFILE_GDN_SUFFIX') == '1':
            cut=kwargs['fla_coefficient_start']
            if not cut:
                raise ValueError('GDN suffix comparison requires the actual common-prefix bound.')
            reference=None
            for name,selected in [('full_cold',0),('suffix_cold',cut),('full_warm_0',0),
                                  ('suffix_warm_0',cut),('suffix_warm_1',cut),('full_warm_1',0)]:
                operands=(dict(args[0]),dict(args[1]),*args[2:])
                options=dict(kwargs,fla_coefficient_start=selected)
                torch.cuda.synchronize()
                before=storage_report()
                start=time.perf_counter()
                value=original_gdn(module,*operands,**options)
                torch.cuda.synchronize()
                elapsed=time.perf_counter()-start
                actual=value[0].detach()[:,cut:].cpu()
                if reference is None:reference=actual
                difference=actual-reference
                record=dict(name=name,seconds=elapsed,before=before,exact_suffix=torch.equal(actual,reference),
                    suffix_max_abs=float(difference.abs().max()),
                    suffix_relative_l2=float(difference.norm()/reference.norm().clamp_min(1e-30)))
                records.append(record)
                print('GDN_SUFFIX_IN_CONTEXT',json.dumps(record),flush=True)
                del value,actual,operands,difference
            profile_output.write_text(json.dumps(dict(
                scope='Same actual first-GDN32k captures/upstream, original actor and sleeping vLLM; official operator tolerance is checked separately, no invented whole-GDN threshold',
                status='observed_requested_phases',coefficient_start=cut,records=records),indent=2)+'\n')
            raise DiagnosticStop()
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
            status='observed_requested_phases' if len(records)==limit else 'running', records=records,
            compact_capture_comparisons=compact_copy_records), indent=2)+'\n')
        print('GDN_IN_CONTEXT_PROFILE', json.dumps(record), flush=True)
        if len(records) == limit:
            raise DiagnosticStop()
        return value

    qwen35_dense_finite_runner.gdn_finite_pullback = observed_gdn

try:
    runpy.run_path(os.environ.get('DT_CAPACITY_SCRIPT',os.environ['DT_ROOT']+'/experiments/rl/verify_dt_context_capacity.py'), run_name='__main__')
except DiagnosticStop:
    print('DIAGNOSTIC_STOP: requested phases recorded; remaining DT/PPO intentionally not run', flush=True)
