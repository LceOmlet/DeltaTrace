"""Compare public cached-convolution APIs on one passive, real DT capture.

No model, checkpoint, environment, DT propagation, or production patch is
loaded. All operands and the existing finite backward seed come from rank*.pt.
Keep their dtypes. Compile the pinned owner's tolerance statements and output/
input-gradient assertions verbatim; do not implement another convolution or
invent tolerances. The actual frozen BF16 weights are outside the official
test's FP32-weight fixture, which is reported even when assertions pass.

Benchmarking is optional, runs after correctness, and excludes CPU transfers
and preparation of the projection's channel-last layout. It measures only the
stated convolution calls on these actual tensors, never an entire DT call.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace


INTERFACE_SHA = '7286f93996561532dac8452de1e75bd44a31baf5acbb3ad4fd48b87fbc7fdc94'
TEST_SHA = 'c15131c88911cf7e942fdd693cfbddd3e2b4d29b0acbfd28ca69b7fd6cb965bf'
HERE = Path(__file__).resolve().parent


def identity(path: Path) -> dict:
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def owner_statements(test_path: Path) -> tuple[dict, dict]:
    source = test_path.read_text(encoding='utf-8')
    if identity(test_path)['sha256'] != TEST_SHA:
        raise RuntimeError('Pinned v1.5.0 official test bytes changed')
    tree = ast.parse(source, filename=str(test_path))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name == 'test_causal_conv1d')
    tolerance = [node for node in fn.body
                 if (isinstance(node, ast.Assign)
                     and ast.unparse(node.targets[0]) == '(rtol, atol)')
                 or (isinstance(node, ast.If)
                     and ast.unparse(node.test) == 'itype == torch.bfloat16')]
    if len(tolerance) != 2:
        raise RuntimeError('Owner tolerance AST selection differs')
    nodes = {'tolerance': tolerance}
    for kind, expression in (
        ('output', 'torch.allclose(out, out_ref, rtol=rtol, atol=atol)'),
        ('dx', 'torch.allclose(x.grad, x_ref.grad.to(dtype=itype), rtol=rtol, atol=atol)'),
    ):
        selected = [node for node in ast.walk(fn) if isinstance(node, ast.Assert)
                    and ast.unparse(node.test) == expression]
        if len(selected) != 1:
            raise RuntimeError('Owner assertion AST selection differs: ' + kind)
        nodes[kind] = selected
    compiled = {key: compile(ast.fix_missing_locations(ast.Module(body=value,
                         type_ignores=[])), str(test_path), 'exec')
                for key, value in nodes.items()}
    provenance = dict(test=identity(test_path), owner_test='test_causal_conv1d',
        statements={key: [dict(line=n.lineno, end_line=n.end_lineno,
                    source=ast.get_source_segment(source, n)) for n in value]
                    for key, value in nodes.items()},
        official_fixture_weight_dtype='torch.float32',
        skipped_assertions='Frozen weight/bias and fixed initial state: no dweight/dbias/dinitial_states assertion')
    return compiled, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--official-test', type=Path, default=HERE.parent / 'test_causal_conv1d_v150.py')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--inspect-only', action='store_true', help='AST/source checks only; never import Torch')
    args = parser.parse_args()
    assertions, owner = owner_statements(args.official_test)
    report = dict(script=identity(Path(__file__)), owner=owner,
        scope='Real cached convolution output and frozen-weight input VJP only; no model/PPO/DT acceptance',
        operations=dict(model_loads=0, checkpoint_loads=0, production_edits=0,
                        installs=0, extra_dt_calls=0), phases=[], checks=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    if args.inspect_only:
        report['status'] = 'official_AST_prepared_only_no_torch_import'
        save()
        return 0
    if args.payload is None or args.warmup < 1 or args.repeats < 1:
        parser.error('A real --payload and positive warmup/repeats are required')

    import psutil
    import torch
    import causal_conv1d.causal_conv1d_interface as installed
    installed_path = Path(inspect.getsourcefile(installed.causal_conv1d_ref))
    report['installed_interface'] = identity(installed_path)
    if report['installed_interface']['sha256'] != INTERFACE_SHA:
        raise RuntimeError('Installed public interface differs from verified v1.5.0 source')
    report['torch'] = torch.__version__
    report['payload'] = identity(args.payload)
    device = torch.device('cuda', args.device)
    torch.cuda.set_device(device)
    process = psutil.Process()
    def phase(name: str):
        torch.cuda.synchronize(device)
        free, total = torch.cuda.mem_get_info(device)
        memory = process.memory_full_info()
        entry = dict(phase=name, unix=time.time(), rss=memory.rss,
                     pss=getattr(memory, 'pss', None), gpu_free=free, gpu_total=total,
                     allocated=torch.cuda.memory_allocated(device),
                     reserved=torch.cuda.memory_reserved(device),
                     peak_allocated=torch.cuda.max_memory_allocated(device))
        cgroup = Path('/sys/fs/cgroup/memory/memory.usage_in_bytes')
        if cgroup.exists():
            entry['cgroup_memory_bytes'] = int(cgroup.read_text())
        try:
            smi = subprocess.run(['mx-smi'], capture_output=True, text=True, timeout=10)
            entry['physical_mx_smi'] = smi.stdout[-25000:]
        except (OSError, subprocess.TimeoutExpired) as error:
            entry['physical_mx_smi_unavailable'] = repr(error)
        report['phases'].append(entry)
        save()
    phase('before_CPU_payload_load')
    saved = torch.load(args.payload, map_location='cpu', weights_only=False)
    record, tensors = saved['record'], saved['tensors']
    report['capture_record'] = record
    context = record['calls']['native']['left_context']
    if context < 1 or tensors['native_x'].shape != tensors['finite_x'].shape:
        raise RuntimeError('Capture does not expose the exact common native/finite left-context cut')
    if tensors['finite_seed'].shape != tensors['finite_x'].shape:
        raise RuntimeError('Actual finite seed shape is incompatible')
    report['paired_endpoint_rows'] = tensors['native_x'].shape[0]
    report['policy_minibatch'] = tensors['native_x'].shape[0] // 2
    report['left_context'] = context
    report['dtype_scope'] = dict(actual={k: str(v.dtype) for k, v in tensors.items()},
        weight_dtype_preserved=True, reference_weight_dtype_preserved=True,
        official_FP32_weight_fixture=(tensors['native_weight'].dtype == torch.float32
                                     and tensors['finite_weight'].dtype == torch.float32),
        statement='Original assertion formula is reused on actual dtypes; BF16 weights are outside the official FP32-weight parameterization')
    # Sequential suites keep live allocations bounded. This is an allocation
    # estimate, not a numerical or performance acceptance threshold.
    largest = max(t.numel() * t.element_size() for t in tensors.values())
    report['estimated_live_test_tensor_bytes'] = 12 * largest
    if torch.cuda.mem_get_info(device)[0] < report['estimated_live_test_tensor_bytes']:
        raise RuntimeError('Insufficient free VRAM for the bounded actual-tensor comparison')
    phase('CPU_payload_loaded_before_GPU_operands')
    conv, reference = installed.causal_conv1d_fn, installed.causal_conv1d_ref

    def check(name, kind, actual, expected):
        env = dict(torch=torch, itype=actual.dtype)
        exec(assertions['tolerance'], env)
        delta = (actual.float() - expected.float()).abs()
        item = dict(name=name, assertion=kind, rtol=env['rtol'], atol=env['atol'],
                    max_abs=delta.max().item(), mean_abs=delta.mean().item())
        if kind == 'output':
            env.update(out=actual, out_ref=expected)
        else:
            env.update(x=SimpleNamespace(grad=actual), x_ref=SimpleNamespace(grad=expected))
        try:
            exec(assertions[kind], env)
            item['passed'] = True
        except AssertionError:
            item['passed'] = False
        report['checks'].append(item)
        save()

    def channel_last(x):
        return x.transpose(1, 2).contiguous().transpose(1, 2)

    def original_layout(name):
        cpu = tensors[name]
        metadata = record['tensors'][name]
        if list(cpu.shape) != metadata['shape'] or str(cpu.dtype) != metadata['dtype']:
            raise RuntimeError('Actual operand metadata differs: ' + name)
        # Restore the captured original layout, not the post-cat slice layout.
        value = torch.empty_strided(cpu.shape, metadata['stride'], device=device, dtype=cpu.dtype)
        value.copy_(cpu)
        return value

    def suite(prefix, activation, backward):
        phase(prefix + '_prepare')
        x = tensors[prefix + '_x'].to(device).detach()
        weight = tensors[prefix + '_weight'].to(device).detach()
        bias = tensors.get(prefix + '_bias')
        bias = None if bias is None else bias.to(device).detach()
        width = weight.shape[-1]
        if context < width - 1:
            raise RuntimeError('Saved cache lacks the public interface left window')
        left, suffix = x[..., :context], channel_last(x[..., context:])
        original_cat_inputs = None
        if prefix == 'native':
            try:
                if 'native_conv_state' in tensors and 'native_projection_output' in tensors:
                    original_left = original_layout('native_conv_state')
                    original_projection = original_layout('native_projection_output')
                    original_mixed = original_projection.transpose(1, 2)
                    joined = torch.cat([original_left, original_mixed], dim=-1)
                    reproduced = torch.equal(joined, x)
                    report['actual_preconcat_reproduction'] = dict(equal=reproduced,
                        original_state_stride=list(original_left.stride()),
                        original_projection_stride=list(original_projection.stride()),
                        original_mixed_stride=list(original_mixed.stride()),
                        joined_stride=list(joined.stride()), recorded_cat_stride=record['tensors']['native_x']['stride'])
                    del joined
                    if reproduced:
                        original_cat_inputs = (original_left, original_mixed)
                        left, suffix = original_cat_inputs
                    else:
                        report['native_concat_benchmark_unavailable'] = 'Actual original inputs do not reproduce saved native_x; no substitute used'
                else:
                    report['native_concat_benchmark_unavailable'] = 'No actual state/projection operands; kernel-only comparison'
            except (RuntimeError, KeyError) as error:
                report['native_concat_benchmark_unavailable'] = repr(error)
        initial = channel_last(left[..., -(width - 1):]).detach()
        seed = tensors['finite_seed'].to(device).detach() if backward else None
        x = x.requires_grad_(backward)
        xr = x.detach().clone().requires_grad_(backward)
        suffix = suffix.detach().requires_grad_(backward)
        sr = suffix.detach().clone().requires_grad_(backward)
        out = conv(x, weight, bias, activation=activation)
        out_ref = reference(xr, weight, bias, activation=activation)
        candidate = conv(suffix, weight, bias, initial_states=initial, activation=activation)
        candidate_ref = reference(sr, weight, bias, initial_states=initial, activation=activation)
        check(prefix + '_existing_cat_vs_official_ref', 'output', out, out_ref)
        check(prefix + '_initial_states_vs_official_ref', 'output', candidate, candidate_ref)
        check(prefix + '_initial_states_vs_existing_cat_suffix', 'output', candidate, out[..., context:])
        captured = tensors['finite_pre' if backward else 'native_output'].to(device)
        report[prefix + '_saved_output_reproduction'] = dict(
            equal=torch.equal(out.detach(), captured),
            max_abs=(out.detach().float() - captured.float()).abs().max().item(),
            scope='Diagnostic only; not a different tolerance or acceptance gate')
        if backward:
            if torch.count_nonzero(seed[..., :context]).item():
                raise RuntimeError('Actual finite seed is not zero on the fixed cached history')
            out.backward(seed)
            out_ref.backward(seed)
            candidate.backward(seed[..., context:])
            candidate_ref.backward(seed[..., context:])
            check('finite_existing_cat_dx_vs_official_ref', 'dx', x.grad, xr.grad)
            check('finite_initial_states_dx_vs_official_ref', 'dx', suffix.grad, sr.grad)
            check('finite_initial_states_dx_vs_existing_cat_suffix', 'dx', suffix.grad, x.grad[..., context:])
            captured_dx = tensors['finite_dx'].to(device)
            report['finite_saved_dx_reproduction'] = dict(equal=torch.equal(x.grad, captured_dx),
                max_abs=(x.grad.float() - captured_dx.float()).abs().max().item(),
                scope='Diagnostic only; actual existing input VJP is preserved')
        phase(prefix + '_assertions_completed')
        if args.benchmark and all(c['passed'] for c in report['checks']):
            # Original native state/projection layouts are restored before timing.
            # A concat timer is emitted only when those actual inputs reproduce x.
            cat_x = x.detach().requires_grad_(backward)
            suffix_x = suffix.detach().requires_grad_(backward)
            def cat_call():
                value = conv(cat_x, weight, bias, activation=activation)
                return torch.autograd.grad(value, cat_x, seed)[0] if backward else value
            def initial_call():
                value = conv(suffix_x, weight, bias, initial_states=initial, activation=activation)
                return torch.autograd.grad(value, suffix_x, seed[..., context:])[0] if backward else value
            def concat_call():
                actual_state, actual_projection_view = original_cat_inputs
                return conv(torch.cat([actual_state, actual_projection_view], dim=-1), weight, bias, activation=activation)
            operations = [('existing_materialized_cat', cat_call), ('initial_states', initial_call)]
            if not backward and original_cat_inputs is not None:
                operations.append(('native_actual_concat_then_kernel', concat_call))
            timings = {}
            for name, operation in operations:
                for _ in range(args.warmup):
                    operation()
                torch.cuda.synchronize(device)
                elapsed, wall = [], []
                for _ in range(args.repeats):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    started = time.perf_counter()
                    begin.record()
                    value = operation()
                    end.record()
                    end.synchronize()
                    elapsed.append(begin.elapsed_time(end))
                    wall.append((time.perf_counter() - started) * 1000)
                    del value
                timings[name] = dict(cuda_ms=elapsed, wall_ms=wall,
                                     median_cuda_ms=statistics.median(elapsed),
                                     median_wall_ms=statistics.median(wall))
            report[prefix + '_benchmark'] = dict(warmup=args.warmup, repeats=args.repeats,
                input_stride=list(x.stride()), suffix_stride=list(suffix.stride()),
                initial_stride=list(initial.stride()), operations=timings,
                scope='One actual layer kernel or kernel+VJP; transfers/layout preparation excluded; not complete DT speedup')
            phase(prefix + '_benchmark_completed')

    try:
        with torch.no_grad():
            suite('native', record['calls']['native']['activation'], False)
        gc.collect()
        suite('finite', None, True)
        gc.collect()
        phase('finished')
        report['status'] = ('original_output_dx_assertions_pass_on_actual_operands'
                            if all(c['passed'] for c in report['checks']) else 'original_assertion_failure')
    except BaseException as error:
        report['status'] = 'operator_test_error'
        report['error'] = repr(error)
        save()
        raise
    save()
    print(json.dumps(dict(status=report['status'], output=str(args.output),
                          checks=len(report['checks']), dtype_scope=report['dtype_scope'])))
    return 0 if all(c['passed'] for c in report['checks']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
