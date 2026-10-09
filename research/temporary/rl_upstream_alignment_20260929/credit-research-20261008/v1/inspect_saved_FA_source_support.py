"""Measure the causal support of original single-deletion Q/K/V changes.

CPU inspection of all frozen native operator captures. No model/FA/DT call,
new reference, candidate coefficient, or token credit is computed. Ratios
describe operator changes; they are not attribution accuracy or causal shares.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import psutil
import torch


def ref(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            digest.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=digest.hexdigest())


def support(pair, row, position, length):
    # Use the saved native full-context layout [2B,heads,time,dimension].
    # 128 positions bound the FP64 temporary; padding is not included.
    energy = dict(before=0.,at=0.,after=0.)
    changed_rows = dict(before=0,at=0,after=0)
    elements = dict(before=0,at=0,after=0)
    for first in range(0,length,128):
        last = min(first+128,length)
        delta = pair[2*row+1,:,first:last].double()-pair[2*row,:,first:last].double()
        row_energy = delta.square().sum((0,2))
        row_changed = (delta != 0).any(0).any(-1)
        locations = torch.arange(first,last)
        for name,mask in (('before',locations<position),('at',locations==position),('after',locations>position)):
            energy[name] += float(row_energy[mask].sum())
            changed_rows[name] += int(row_changed[mask].sum())
            elements[name] += int(mask.sum())*pair.shape[1]*pair.shape[-1]
    total = sum(energy.values())
    return dict(squared_change_energy=energy,changed_rows=changed_rows,elements=elements,
        total_squared_change=total,
        energy_fraction={k:v/total if total else None for k,v in energy.items()},
        outside_source_row_fraction=(energy['before']+energy['after'])/total if total else None,
        interpretation='Exact saved BF16 operand differences accumulated in FP64; zero is literal, not a tolerance.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    begin = time.perf_counter()
    process = psutil.Process()
    ranks = [args.directory/'results'/f'rank{r}.json' for r in (0,1)]
    rank_data = [json.loads(p.read_bytes()) for p in ranks]
    assert all(r['phase']=='complete' for r in rank_data)
    points = {(p['traj_uid'],p['packed_slot']):p for r in rank_data for b in r['batches'] for p in b['points']}
    assert len(points)==165
    artifacts = {a['path']:a for r in rank_data for b in r['batches']
        for a in b['attention_PV_readout']['original_operand_artifacts'] if '-native-' in a['path']}
    result = dict(scope=__doc__,pid=os.getpid(),birth=process.create_time(),unix=time.time(),
        sources=[ref(__file__),*[ref(p) for p in ranks]],expected_points=165,
        operations=dict(model=0,FA=0,DT=0,optimizer=0,rollout=0,checkpoint_restore=0),
        production_modified=False,candidate=False,CUDA_initialized=False,
        phase='initialized',files=[],points=[],elapsed_seconds=0.,sampled_PSS_peak_bytes=0,
        RSS_scope='Sampled process PSS after each completed native artifact; not a whole-host peak',
        limitation='This checks whether original input-token deletion remains a one-hidden-row perturbation. '
            'Energy is not a credit magnitude or proof of an optimal propagation rule. Body/tail grouping uses existing frozen identities.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    phases = args.output.with_suffix('.phases.jsonl').open('x',buffering=1)

    def save(phase, **extra):
        result.update(phase=phase,elapsed_seconds=time.perf_counter()-begin,**extra)
        pss=process.memory_full_info().pss
        result['sampled_PSS_peak_bytes']=max(result['sampled_PSS_peak_bytes'],pss)
        phases.write(json.dumps(dict(phase=phase,elapsed_seconds=result['elapsed_seconds'],
            PSS_bytes=pss,completed_points=len(result['points']),**extra))+'\n')
        args.output.write_text(json.dumps(result,indent=2)+'\n')

    save('begin')
    visited=set()
    try:
        for path,expected in artifacts.items():
            save('hash_original_artifact',artifact=path)
            actual=ref(path)
            assert actual['sha256']==expected['sha256'] and actual['bytes']==expected['bytes']
            saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            save('read_original_operand_support',artifact=path)
            for row,query in enumerate(saved['queries']):
                if query is None:
                    continue
                key=(saved['trajectories'][row],query['packed_slot'])
                assert key in points and key not in visited
                point=points[key]
                assert point['token_id']==query['token_id'] and point['source_index']==query['source_index']
                length=saved['actual_context_lengths'][row]
                position=query['packed_slot']
                assert 0<=position<length
                result['points'].append(dict(traj_uid=key[0],packed_slot=position,token_id=query['token_id'],
                    initial_state_sha256=point['initial_state_sha256'],previously_examined=point['previously_examined'],
                    previous_comparisons=point['previous_comparisons'],
                    saved_d=point['saved_d'] if 'saved_d' in point else point['d'],
                    fresh_DT_d=point['fresh_DT_d'],native_single_d=point['native_single_d'],
                    context_tokens=length,actual_operator_file=actual,
                    actual_dtype={n:str(saved['endpoints'][n].dtype) for n in ('query','key','value')},
                    operand_support={n:support(saved['endpoints'][n],row,position,length) for n in ('query','key','value')}))
                visited.add(key)
            result['files'].append(actual)
            del saved
            gc.collect()
            save('original_artifact_complete',artifact=path)
        assert visited==set(points)
        assert not torch.cuda.is_initialized()
        save('complete')
    except BaseException:
        import traceback
        save('failed',traceback=traceback.format_exc())
        raise
    finally:
        phases.close()
    print(json.dumps(dict(output=ref(args.output),points=len(visited),files=len(artifacts),
        elapsed_seconds=result['elapsed_seconds'],sampled_PSS_peak_bytes=result['sampled_PSS_peak_bytes'])))


if __name__=='__main__':
    main()
