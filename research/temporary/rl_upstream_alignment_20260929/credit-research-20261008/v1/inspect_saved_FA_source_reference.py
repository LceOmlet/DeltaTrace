"""Controlled original finite-FA replay of the source-key reference value.

Use every saved TextCraft operator case, not selected extreme tokens. Compare
the original joint V reference with one source-key row replaced by the saved
native single-deletion V at that row. This is an oracle operand diagnostic,
not a computable training rule or replacement of token credit. Original Q/K,
upstream, finite library, dtype and causal coordinates remain fixed.
"""
import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import time

import psutil
import torch
from flash_attn import flash_attn_func


def ref(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8 << 20), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def contraction(coeff, native, row):
    start = native['starts'][row]
    length = native['actual_context_lengths'][row]-start
    values = {}
    for op, name in (('query', 'dq'), ('key', 'dk'), ('value', 'dv')):
        pair = native['endpoints'][op]
        total = 0.
        for first in range(0, length, 128):
            last = min(first+128, length)
            delta = pair[2*row+1, :, start+first:start+last].double()-pair[2*row, :, start+first:start+last].double()
            c = coeff[name][row, :, first:last].double().reshape(pair.shape[1], -1, last-first, pair.shape[-1]).sum(1)
            total += float((c*delta).sum())
        values[op] = total
    return values


def replay_difference(a, b):
    largest = 0.
    differing = 0
    for first in range(0, a.shape[2], 128):
        x, y = a[:, :, first:first+128], b[:, :, first:first+128]
        largest = max(largest, float((x.float()-y.float()).abs().max()))
        differing += int((x != y).sum())
    return dict(maxabs=largest, differing_elements=differing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--owner', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    begin = time.perf_counter()
    proc = psutil.Process()
    rank_paths = [args.directory/'results'/f'rank{r}.json' for r in (0, 1)]
    reports = [json.loads(p.read_bytes()) for p in rank_paths]
    assert all(r['phase'] == 'complete' for r in reports)
    points = {(p['traj_uid'], p['packed_slot']):p for r in reports for b in r['batches'] for p in b['points']}
    assert len(points) == 165
    protocol = json.loads(args.protocol.read_bytes())
    assert {(p['traj_uid'], p['packed_slot']) for p in protocol['points']} == set(points)
    assert [ref(p) for p in rank_paths] == protocol['original_rank_reports']
    owner_ref = ref(args.owner)
    library_ref = ref(args.library)
    assert owner_ref['sha256'] == '3e1d61037be22a1cc826b004d49f854e3c246a149642a4f9167c204181f34089'
    assert library_ref['sha256'] == '4f42c391055afec0a0fee9ee698c0820163ff413e42f1c4909b2961ce81e5157'
    spec = importlib.util.spec_from_file_location('saved_original_finite_FA_owner', args.owner)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    finite = owner.VendorFAFiniteP1BF16D256(args.library, library_ref['sha256'])
    public_ref = ref(inspect.getsourcefile(flash_attn_func))
    assert {p['final_FA_PV_ledger']['actual_owner']['sha256'] for p in points.values()} == {public_ref['sha256']}
    result = dict(scope=__doc__, pid=os.getpid(), birth=proc.create_time(), unix=time.time(), phase='initialized',
        sources=[ref(__file__), ref(args.protocol), *[ref(p) for p in rank_paths], owner_ref, library_ref, public_ref],
        operations=dict(model=0, DT=0, optimizer=0, rollout=0, checkpoint_restore=0,
                        original_public_FA=0, original_finite_FA=0),
        production_modified=False, training_candidate=False, points=[], batches=[],
        sampled_PSS_peak_bytes=0, GPU_peak_allocated_bytes=0, GPU_peak_reserved_bytes=0,
        limitation='The native source value is only available in the saved diagnostic. No single-delete model calls are introduced into training. '
            'A smaller local residual does not establish a more accurate composed token credit or a deployable repair.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    phases = args.output.with_suffix('.phases.jsonl').open('x', buffering=1)

    def save(phase, **extra):
        result.update(phase=phase, elapsed_seconds=time.perf_counter()-begin, **extra)
        result['sampled_PSS_peak_bytes'] = max(result['sampled_PSS_peak_bytes'], proc.memory_full_info().pss)
        if torch.cuda.is_initialized():
            result['GPU_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            result['GPU_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        phases.write(json.dumps(dict(phase=phase, elapsed_seconds=result['elapsed_seconds'], points=len(result['points']),
            operations=result['operations'], **extra))+'\n')
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

    save('begin')
    try:
        visited = set()
        with torch.no_grad():
            for report in reports:
                for batch in report['batches']:
                    artifacts = batch['attention_PV_readout']['original_operand_artifacts']
                    joint_ref = next(a for a in artifacts if '-joint.pt' in a['path'])
                    save('hash_joint_operands', artifact=joint_ref['path'])
                    assert ref(joint_ref['path']) == joint_ref
                    joint = torch.load(joint_ref['path'], map_location='cpu', mmap=True, weights_only=False)
                    bank = joint['bank']
                    q, k, v = (bank[n].to('cuda') for n in ('query', 'key', 'value'))
                    nrow, nheads, qlen, dim = q.shape
                    assert nrow == 8 and dim == 256
                    lengths, starts = joint['actual_context_lengths'], joint['starts']
                    lse = torch.full((nrow, nheads, qlen), float('inf'), device='cuda', dtype=torch.float32)
                    # Original public API, bottom-right causal alignment. Each
                    # paired row has its own logical cached suffix and length.
                    # dropout=0 means the native API does not allocate S_dmask.
                    for row in range(nrow):
                        length, start = lengths[row//2], starts[row//2]
                        n = length-start
                        out, norm, probabilities = flash_attn_func(
                            q[row:row+1, :, :n].transpose(1, 2),
                            k[row:row+1, :, :length].transpose(1, 2),
                            v[row:row+1, :, :length].transpose(1, 2),
                            dropout_p=0., softmax_scale=.0625, causal=True, return_attn_probs=True)
                        result['operations']['original_public_FA'] += 1
                        assert probabilities is None or probabilities.numel() == 0  # No NxN buffer at dropout=0.
                        lse[row, :, :n] = norm[0]
                        del out, norm, probabilities
                    layout = owner.RightPaddedLengths(lengths, k.shape[2], q.device,
                        coefficient_starts=starts, query_starts=starts, query_padded_length=qlen)
                    ops = dict(q0=q[0::2], q1=q[1::2], k0=k[0::2], k1=k[1::2], v0=v[0::2],
                        u=bank['mc'].transpose(1, 2).to('cuda'), lse0=lse[0::2], lse1=lse[1::2])
                    raw = finite(ops, .0625, layout)
                    result['operations']['original_finite_FA'] += 1
                    baseline = {n:raw[n].cpu() for n in ('dq', 'dk', 'dv')}
                    del raw
                    result['batches'].append(dict(joint_file=joint_ref,
                        baseline_coefficient_replay_difference={n:replay_difference(baseline[n], bank[n])
                            for n in baseline},
                        query_starts=starts, lengths=lengths, query_padded_length=qlen,
                        scope='Replay difference is recorded, not a new tolerance; controlled variants share exactly these replayed LSE values.'))
                    for native_ref in (a for a in artifacts if '-native-' in a['path']):
                        save('hash_native_operands', artifact=native_ref['path'])
                        assert ref(native_ref['path']) == native_ref
                        native = torch.load(native_ref['path'], map_location='cpu', mmap=True, weights_only=False)
                        assert native['starts'] == starts and native['actual_context_lengths'] == lengths
                        modified_v = ops['v0'].clone()
                        for row, query in enumerate(native['queries']):
                            if query is not None:
                                pos = query['packed_slot']
                                modified_v[row, :, pos] = native['endpoints']['value'][2*row, :, pos].to('cuda')
                        changed_ops = dict(ops, v0=modified_v)
                        raw = finite(changed_ops, .0625, layout)
                        result['operations']['original_finite_FA'] += 1
                        controlled = {n:raw[n].cpu() for n in ('dq', 'dk', 'dv')}
                        del raw, changed_ops, modified_v
                        for row, query in enumerate(native['queries']):
                            if query is None:
                                continue
                            key = native['trajectories'][row], query['packed_slot']
                            assert key in points and key not in visited
                            visited.add(key)
                            p = points[key]
                            a, b = contraction(baseline, native, row), contraction(controlled, native, row)
                            target = p['final_FA_PV_ledger']['native_content_contraction']
                            result['points'].append(dict(traj_uid=key[0], packed_slot=key[1], token_id=query['token_id'],
                                initial_state_sha256=p['initial_state_sha256'], previously_examined=p['previously_examined'],
                                native_file=native_ref, baseline_QKV_contractions=a, controlled_QKV_contractions=b,
                                native_output_contraction=target, baseline_core_residual=sum(a.values())-target,
                                controlled_core_residual=sum(b.values())-target,
                                historical_core_residual=p['final_FA_PV_ledger']['native_pair_core_residual'],
                                value_coefficient_contraction_difference=b['value']-a['value'],
                                interpretation='Controlled local operator result only; no new global token d or A is inferred.'))
                        del native, controlled
                        gc.collect()
                        save('native_source_reference_complete', artifact=native_ref['path'])
                    del baseline, ops, q, k, v, lse, layout, bank, joint
                    gc.collect()
        assert visited == set(points)
        assert result['operations']['original_public_FA'] == 96
        assert result['operations']['original_finite_FA'] == 72
        torch.cuda.synchronize()
        save('complete')
    except BaseException:
        import traceback
        save('failed', traceback=traceback.format_exc())
        raise
    finally:
        phases.close()
    print(json.dumps(dict(output=ref(args.output), points=len(visited), operations=result['operations'],
        elapsed_seconds=result['elapsed_seconds'], PSS=result['sampled_PSS_peak_bytes'],
        GPU_allocated=result['GPU_peak_allocated_bytes'], GPU_reserved=result['GPU_peak_reserved_bytes'])))


if __name__ == '__main__':
    main()
