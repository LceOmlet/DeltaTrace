"""Read existing single-background stage ledgers on CPU; never replay DT.

Finite-rule and native layer-replay residuals remain separate. A B4 ledger is
a batch contraction and must not be assigned to one source token. No official
tolerance is invented and no saved credit is modified.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import psutil
import torch


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    start=time.perf_counter()
    process=psutil.Process()
    protocol=json.loads(args.protocol.read_bytes())
    output=dict(scope=__doc__,pid=process.pid,birth=process.create_time(),unix=time.time(),
        source=dict(path=__file__,sha256=sha(__file__)),protocol=dict(path=str(args.protocol),sha256=sha(args.protocol)),
        phase='reading',tasks={},operations=dict(model=0,DT=0,GPU=0,optimizer=0,rollout=0,checkpoint_restore=0),
        production_modified=False,candidate=False,sampled_PSS_peak_bytes=0)

    def save():
        output['elapsed_seconds']=time.perf_counter()-start
        output['sampled_PSS_peak_bytes']=max(output['sampled_PSS_peak_bytes'],process.memory_full_info().pss)
        output['cuda_initialized']=torch.cuda.is_initialized()
        assert not output['cuda_initialized']
        args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')

    save()
    for task,points in protocol['tasks'].items():
        artifacts={}
        for p in points:
            a=p['artifact']
            group=artifacts.setdefault(a['path'],dict(artifact=a,points=[]))
            assert group['artifact']==a
            group['points'].append(p)
        records=[]
        output['tasks'][task]=dict(points=len(points),artifacts=len(artifacts),batches=records)
        for path,spec in artifacts.items():
            assert sha(path)==spec['artifact']['sha256']
            data=torch.load(path,map_location='cpu',weights_only=False)
            detail=data['detail']
            assert len(detail['layers'])==32
            stages=[]
            total=detail['seed_effect']-detail['root_effect']
            for layer in reversed(range(32)):
                entry=detail['layers'][str(layer)]
                finite=entry['input_effect']-entry['replay_output_effect']
                replay=entry['replay_output_effect']-entry['root_output_effect']
                total+=finite+replay
                stages.append(dict(layer=layer,block_type=entry['block_type'],finite_rule_residual=finite,
                    native_replay_residual=replay,replay_relative_L2=entry['replay_relative_L2'],
                    root_output_effect=entry['root_output_effect'],replay_output_effect=entry['replay_output_effect'],
                    input_effect=entry['input_effect']))
            cohorts=set(c for p in spec['points'] for c in p['cohorts'])
            record=dict(artifact=spec['artifact'],points=spec['points'],cohorts=sorted(cohorts),
                cohort_composition='uniform_only' if cohorts=={'uniform'} else
                    'tail_only' if cohorts=={'predicted_tail_census'} else 'mixed',
                root_effect=detail['root_effect'],seed_effect=detail['seed_effect'],signed_sum=detail['signed_sum'],
                head_and_final_norm_residual=detail['seed_effect']-detail['root_effect'],
                compiled_head_endpoint_discrepancy=detail.get('compiled_seed_logprob_effect_minus_root'),
                reconstructed_signed_minus_root=total,
                saved_signed_minus_root=detail['signed_sum']-detail['root_effect'],
                closure_roundoff=total-(detail['signed_sum']-detail['root_effect']),stages=stages,
                original_DT_seconds=detail['complete_attribution_seconds_with_diagnostics'])
            assert all(math.isfinite(stage[key]) for stage in stages
                for key in ('finite_rule_residual','native_replay_residual'))
            records.append(record)
            del data
            save()
    output.update(phase='complete',interpretation='Original B4 scalar contractions only. Finite-rule residuals include '
        'the original arithmetic and endpoint captures; replay residuals compare the native replay and native root. '
        'The ledger is not a per-token causal error share, new tolerance, or complete accuracy repair.')
    save()
    print(json.dumps(dict(output=str(args.output),phase=output['phase'],seconds=output['elapsed_seconds'],
        PSS=output['sampled_PSS_peak_bytes'],cuda_initialized=output['cuda_initialized'],
        tasks={task:dict(points=r['points'],artifacts=r['artifacts']) for task,r in output['tasks'].items()})))


if __name__=='__main__':
    main()
