"""Diagnose a saved nonzero finite-FA call by composing existing DT owners.

No model, rollout, backward, optimizer or kernel changes. The original DT
softmax/matmul finite rules and FA's original causal mask provide the dense
reference, evaluated in FP32 and FP64. This is not an invented FA tolerance:
official FA assertions cover ordinary attention/backward, not this finite map.
All rows and heads of the one previously saved B4 call are used. This operator
diagnostic does not substitute for the frozen trajectory faithfulness study.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import psutil
    import torch
    torch.set_num_threads(8)
    source = args.root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
    assert sha(source) == '58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
    environment = json.loads(source.read_bytes())['environment']
    dt = Path(environment['DT_ROOT'])
    paths = {
        'rules': (dt/'clean/qwen3/signed_secant_rules.py',
                  '056d576e31b7076e7a08c89a5f25fde86897cfdf9daa3988d0a20acd6401030d'),
        'finite': (dt/'clean/qwen35/vendor_fa_finite_bf16_d256.py',
                   '3e1d61037be22a1cc826b004d49f854e3c246a149642a4f9167c204181f34089'),
        'official_FA': (args.root/'receipts/training-setup/official-kernel-tests/test_flash_attn_v263.py',
                        'a290e11cbcb2e65fe7b8399d42eae3bb5c4113bbc12e6190cd7f710ad70abca9'),
        'operands': (args.root/'receipts/direct-target-extreme-operator-20261007-v1/results/rank0-decoder27-fa.pt',
                     'ed804b5cd725cd027d697b97897a076170a2b7df64bf1347cd1e6297f68bad92'),
    }
    for path, expected in paths.values():
        assert sha(path) == expected, str(path)
    sys.path[:0] = [str(dt/'clean/qwen35'), str(dt/'clean/qwen3')]
    rules = load('diagnostic_existing_secant_rules', paths['rules'][0])
    finite = load('vendor_fa_finite_bf16_d256', paths['finite'][0])
    official = load('diagnostic_original_FA_test', paths['official_FA'][0])
    saved = torch.load(paths['operands'][0], map_location='cpu', weights_only=False, mmap=True)
    config = json.loads(Path(environment['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
    operation = finite.VendorFAFiniteP1BF16D256(config['finite_library'], config['finite_library_sha256'])
    layout = saved['layout']
    starts = layout.query_starts or (layout.query_start,)*len(layout.lengths)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(scope=__doc__, unix=time.time(), pid=os.getpid(), birth=psutil.Process().create_time(),
        source_sha256=sha(source), script_sha256=sha(__file__),
        sources={name:dict(path=str(path), sha256=expected) for name,(path,expected) in paths.items()},
        library=dict(path=config['finite_library'], sha256=sha(config['finite_library'])),
        reference='Original softmax_secant_pullback + matmul_secant_pullback; content-P1 PV rule; original FA causal mask',
        upstream='Original FP32 U explicitly rounded to BF16, as the active finite wrapper does',
        chunk_query_rows=1024, rows=[], completed_heads=0,
        model_calls=0, DT_calls=0, optimizer_steps=0, acceptance_threshold=None)
    phase_file = (args.output/'phases.jsonl').open('x', buffering=1)
    started = time.perf_counter()

    def emit(phase, **data):
        phase_file.write(json.dumps(dict(phase=phase, unix=time.time(), elapsed=time.perf_counter()-started,
            allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
            pss=psutil.Process().memory_full_info().pss, **data))+'\n')

    def compare(actual, reference):
        # No threshold: preserve the actual signed extrema and descriptive error.
        a, b = actual.double(), reference.double()
        error = a-b
        flat = int(error.abs().argmax())
        coordinates = list(torch.unravel_index(torch.tensor(flat, device=error.device), error.shape))
        coordinates = [int(i) for i in coordinates]
        ref_l2 = float(torch.linalg.vector_norm(b))
        return dict(max_abs=float(error.abs().max()), error_l2=float(torch.linalg.vector_norm(error)),
            reference_l2=ref_l2, relative_l2=None if ref_l2 == 0 else float(torch.linalg.vector_norm(error))/ref_l2,
            max_at=coordinates, actual_at_max=float(a[tuple(coordinates)]),
            reference_at_max=float(b[tuple(coordinates)]), finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))

    try:
        with torch.no_grad():
            for row, (length, start, coefficient) in enumerate(zip(layout.lengths, starts, layout.coefficient_starts)):
                count, keep = length-start, coefficient-start
                ops = {name:value[row:row+1,:,:length if name in ('k0','k1','v0') else count].contiguous().cuda()
                       for name,value in saved['ops'].items()}
                actual_layout = finite.RightPaddedLengths([length], length, 'cuda',
                    coefficient_starts=[coefficient], query_starts=[start], query_padded_length=count)
                emit('original_finite_begin', row=row)
                actual = operation(ops, saved['scale'], actual_layout)
                torch.cuda.synchronize()
                replay = {name:torch.equal(value.cpu(),saved['coefficients'][name][row:row+1,:,:count])
                          for name,value in actual.items()}
                assert all(replay.values()), replay
                item = dict(row=row, length=length, query_start=start, coefficient_start=coefficient,
                    replay_exact=replay, heads=[])
                report['rows'].append(item)
                # Original FA bottom-right causal mask, restricted only after its construction.
                causal_mask = official.construct_local_mask(count, length, window_size=(-1,0), device='cuda')
                q_heads, kv_heads = ops['q0'].shape[1], ops['k0'].shape[1]
                for head in range(q_heads):
                    kv_head = head//(q_heads//kv_heads)
                    result = dict(head=head, kv_head=kv_head, precisions={})
                    ref32 = None
                    for dtype in (torch.float32, torch.float64):
                        emit('reference_head_begin', row=row, head=head, dtype=str(dtype))
                        tick = time.perf_counter()
                        q0, q1 = [ops[name][0,head].to(dtype) for name in ('q0','q1')]
                        k0, k1, v0 = [ops[name][0,kv_head].to(dtype) for name in ('k0','k1','v0')]
                        u = ops['u'][0,head].bfloat16().to(dtype)
                        dq = torch.zeros_like(q0)
                        dk, dv = torch.zeros_like(k0), torch.zeros_like(v0)
                        tau = torch.zeros(count, device='cuda', dtype=dtype)
                        center = torch.zeros_like(tau)
                        lse_discrepancy = {'lse0':0., 'lse1':0.}
                        for begin in range(keep, count, 1024):
                            end = min(count, begin+1024)
                            s0 = (q0[begin:end]@k0.T)*saved['scale']
                            s1 = (q1[begin:end]@k1.T)*saved['scale']
                            mask = causal_mask[begin:end]
                            s0.masked_fill_(mask, -torch.inf);s1.masked_fill_(mask, -torch.inf)
                            for name, scores in (('lse0',s0),('lse1',s1)):
                                lse_discrepancy[name] = max(lse_discrepancy[name],
                                    float((scores.logsumexp(-1)-ops[name][0,head,begin:end]).abs().max()))
                            mp = u[begin:end]@v0.T
                            weights = rules.softmax_logarithmic_mean(s0,s1)
                            tau[begin:end] = weights.sum(-1)
                            center[begin:end] = (weights*mp).sum(-1)/weights.sum(-1)
                            del weights
                            ms = rules.softmax_secant_pullback(s0,s1,mp)
                            q_part, k_part_transposed = rules.matmul_secant_pullback(
                                q0[begin:end], q1[begin:end], k0.T, k1.T, ms)
                            dq[begin:end] = q_part*saved['scale']
                            dk.add_(k_part_transposed.T*saved['scale'])
                            dv.add_(s1.softmax(-1).T@u[begin:end])
                            del s0,s1,mp,ms,q_part,k_part_transposed
                        values = dict(dq=dq[keep:],dk=dk[start+keep:],dv=dv[start+keep:],tau=tau[keep:],center=center[keep:])
                        checks = {name:compare(actual[name][0,head,keep:],value) for name,value in values.items()}
                        if ref32 is not None:
                            fp32_to_fp64 = {name:compare(ref32[name],value) for name,value in values.items()}
                        else:
                            fp32_to_fp64 = None
                            ref32 = {name:value.clone() for name,value in values.items()}
                        # Numerical contribution error at each actual retained token, not only vector norms.
                        contribution = {}
                        for name, delta in (('dq',q1[keep:]-q0[keep:]),('dk',k1[coefficient:]-k0[coefficient:])):
                            observed = (actual[name][0,head,keep:].double()*delta.double()).sum(-1)
                            expected = (values[name].double()*delta.double()).sum(-1)
                            contribution[name] = compare(observed,expected)
                        result['precisions'][str(dtype)] = dict(seconds=time.perf_counter()-tick,
                            checks=checks, fp32_to_fp64=fp32_to_fp64, supplied_lse_max_abs=lse_discrepancy,
                            token_delta_contribution=contribution,
                            tau_min=float(tau[keep:].min()),tau_max=float(tau[keep:].max()))
                        emit('reference_head_complete', row=row, head=head, dtype=str(dtype),
                             seconds=time.perf_counter()-tick)
                        del q0,q1,k0,k1,v0,u,dq,dk,dv,tau,center,values
                    item['heads'].append(result)
                    report['completed_heads'] += 1
                    report['elapsed_seconds'] = time.perf_counter()-started
                    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
                    del ref32
                del ops,actual,actual_layout,causal_mask
        report.update(status='completed_descriptive_nonzero_reference', completed_unix=time.time(),
            elapsed_seconds=time.perf_counter()-started, peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved())
    except BaseException:
        import traceback
        report.update(status='failed', traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter()-started)
        raise
    finally:
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        phase_file.close()
    print(json.dumps(dict(status=report['status'],heads=report['completed_heads'],seconds=report['elapsed_seconds'])))


if __name__ == '__main__':
    main()
