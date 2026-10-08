"""Native convolution seam on saved real projected endpoints, all 199 sources.

These historical operator inputs do not replace the frozen method development
set or its author RISE/MAS evaluation. No model, DT or optimizer is executed.
"""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import time

import torch
from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn, causal_conv1d_ref
from conditional_conv_windows import conditional_conv_windows, native_qkv


def identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def official_tolerances(path, dtype):
    """Execute the exact rtol/atol assignments from original test function."""
    test = next(n for n in ast.parse(path.read_bytes()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'test_causal_conv1d')
    assignments = []
    for node in test.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
            names = [getattr(n, 'id', None) for n in node.targets[0].elts]
            if names == ['rtol', 'atol']:
                assignments.append(node)
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'itype == torch.bfloat16':
            assignments.append(node)
    assert len(assignments) == 2
    ns = dict(torch=torch, itype=dtype)
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), 'exec'), ns)
    return ns['rtol'], ns['atol']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('operands', 'official', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    assert identity(args.operands)['sha256'] == '1f557a5b4320d5ce6986ba9123c149fbd68274405e539b7e063a38006362c32a'
    assert identity(args.official)['sha256'] == 'c15131c88911cf7e942fdd693cfbddd3e2b4d29b0acbfd28ca69b7fd6cb965bf'
    torch.set_num_threads(8)
    saved = torch.load(args.operands, mmap=True, map_location='cpu', weights_only=False)
    raw = saved['tensors']
    assert saved['record']['calls']['native']['left_context'] == 4
    assert raw['native_weight'].shape == (8192, 4)
    assert torch.equal(raw['finite_x'], raw['native_x'])
    result = dict(scope=__doc__, cases=[], script=identity(__file__),
        seam=identity(inspect.getfile(conditional_conv_windows)),
        owner=identity(inspect.getfile(causal_conv1d_fn)), operands=identity(args.operands),
        official_test=identity(args.official), model_calls=0, DT_calls=0, optimizer=0,
        production_modified=False, whole_DT_credit_repair_accepted=False,
        boundary_rule='real last3 cached inputs; only one source changed, its next3 factual rows retained')
    def save(phase):
        result.update(phase=phase, observed_unix=time.time())
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    with torch.no_grad():
        for dtype in (torch.float16, torch.bfloat16):
            save('native_convolution_'+str(dtype))
            tick = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            T = 199
            factual = raw['native_x'][1::2, :, 4:4+T].cuda().to(dtype)
            replacement = raw['native_x'][0::2, :, 4:4+T].cuda().to(dtype)
            initial = raw['native_x'][1::2, :, 1:4].cuda().to(dtype)
            weight = raw['native_weight'].cuda().to(dtype)
            windows = conditional_conv_windows(causal_conv1d_fn, factual, replacement, weight, initial=initial)
            qkv = native_qkv(windows['output'], key_heads=16, value_heads=32, key_dim=128, value_dim=128)
            case = dict(dtype=str(dtype), B=4, source_count=T, source_selection='all positions in saved prefix span',
                original_tolerance=official_tolerances(args.official, dtype), reference_groups=[],
                composition_mismatches=dict(pre=0, output=0, q=0, k=0, v=0))
            B, D, _ = factual.shape
            for start in range(0, T, 16):
                stop = min(start+16, T)
                E = stop-start
                # Full original public calls are a diagnostic reference only;
                # this E*T work is never added to the training estimator.
                x = factual.repeat(E, 1, 1).transpose(1, 2).contiguous().transpose(1, 2)
                h = initial.repeat(E, 1, 1).transpose(1, 2).contiguous().transpose(1, 2)
                for row, source in enumerate(range(start, stop)):
                    x[row*B:(row+1)*B, :, source] = replacement[:, :, source]
                ref_pre = causal_conv1d_fn(x, weight, initial_states=h, activation=None)
                ref_out = causal_conv1d_fn(x, weight, initial_states=h, activation='silu')
                rtol, atol = case['original_tolerance']
                checks = {}
                for name, actual, activation in [('pre', ref_pre, None), ('output', ref_out, 'silu')]:
                    original_ref = causal_conv1d_ref(x, weight.float(), initial_states=h, activation=activation)
                    checks[name] = bool(torch.allclose(actual, original_ref, rtol=rtol, atol=atol))
                    assert checks[name], (str(dtype), start, name, 'original native/ref tolerance')
                gathered = {name:torch.zeros((4, B, E, D), device=x.device, dtype=dtype)
                            for name in ('pre', 'output')}
                for row, source in enumerate(range(start, stop)):
                    n = min(4, T-source)
                    for name, value in [('pre', ref_pre), ('output', ref_out)]:
                        gathered[name][:n, :, row] = value[row*B:(row+1)*B, :, source:source+n].permute(2, 0, 1)
                mask = windows['valid'][:, None, start:stop, None]
                for name in ('pre', 'output'):
                    case['composition_mismatches'][name] += int(((gathered[name] != windows[name][:, :, start:stop]) & mask).sum())
                expected_qkv = native_qkv(gathered['output'], key_heads=16, value_heads=32, key_dim=128, value_dim=128)
                for name in ('q', 'k', 'v'):
                    case['composition_mismatches'][name] += int(((expected_qkv[name] != qkv[name][:, :, start:stop]) & mask[..., None]).sum())
                case['reference_groups'].append(dict(start=start, stop=stop, owner_checks=checks))
                del x, h, ref_pre, ref_out, original_ref, gathered, expected_qkv
            assert not any(case['composition_mismatches'].values()), case['composition_mismatches']
            torch.cuda.synchronize()
            case.update(seconds=time.perf_counter()-tick,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                original_public_window_calls=2,
                zero_change_rows=int((factual==replacement).all(1).sum()))
            artifact=args.output.parent/('conv-windows-'+str(dtype).split('.')[-1]+'.pt')
            torch.save(dict(factual=factual.cpu(), replacement=replacement.cpu(), initial=initial.cpu(),
                weight=weight.cpu(), windows={k:v.cpu() for k,v in windows.items()},
                qkv={k:v.cpu() for k,v in qkv.items()}), artifact)
            case['exact_artifact'] = identity(artifact)
            result['cases'].append(case)
            save('case_complete')
            del factual, replacement, initial, weight, windows, qkv
    result['passed'] = True
    save('complete')
    print(json.dumps(dict(result=str(args.output), cases=[{k:c[k] for k in
        ('dtype','source_count','composition_mismatches','seconds','peak_allocated_bytes')}
        for c in result['cases']])), flush=True)


if __name__ == '__main__':
    main()
