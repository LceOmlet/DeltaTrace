"""Call original native FLA on captured joint operands; no model or repair.

The saved Q/K are already normalized. The original public API therefore uses
use_qk_l2norm_in_kernel=False and the actual captured incoming chunk state.
Finite contractions are measurements, with no invented acceptance tolerance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ('source','launch','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    source=json.loads(a.source.read_bytes())
    launch=json.loads(a.launch.read_bytes())
    assert sha(a.source)==launch['source_sha256']
    os.environ.update(source['environment'])
    os.environ['CUDA_VISIBLE_DEVICES']='4'
    os.environ.pop('MACA_VISIBLE_DEVICES',None)
    candidate=launch['memory_candidate']
    root=Path(candidate['candidate_dt_root'])
    env=json.loads(Path(candidate['candidate_environment']).read_bytes())['qwen35']
    sys.path[:0]=[str(root),os.environ.get('DT_OFFICIAL_ROOT') or env['official_root'],
        str(root/'clean/qwen35'),*source['pythonpath'].split(':'),env['ft_extension_root']]
    import inspect
    import psutil
    import torch
    from finite_fla_gpu import verify_native_sources
    from qwen35_dense_finite_runner import _token_effect
    chunk=verify_native_sources(env['native_stage_source_sha256'])
    native=json.loads((a.launch.parent/'results/rank0.json').read_bytes())['gdn_subops']['30']
    a.output.mkdir(exist_ok=False)
    report=dict(pid=os.getpid(),birth=psutil.Process().create_time(),phase='running',groups=[],
        owner=dict(path=inspect.getfile(chunk),sha256=sha(inspect.getfile(chunk))),
        verified_native_source_sha256=env['native_stage_source_sha256'],
        public_signature=str(inspect.signature(chunk.chunk_gated_delta_rule)),
        scope='Original public native FLA forward on captured selected joint row, paired B2 and eight heads per call; no model/full-DT/actor/rollout/training-backward/update. Initial state is actual captured h[:,0], not zero. No single-deletion states are synthesized.',
        official_tolerance_claim=False,production_profile_changed=False,credit_repaired=False)
    def save():
        (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for artifact in native['operand_artifacts']:
            path=Path(artifact['path'])
            assert sha(path)==artifact['sha256']
            saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            ep={k:v.to('cuda') for k,v in saved['endpoints'].items()}
            initial=ep['h'][:,0].contiguous()
            do=saved['do'].to('cuda')
            torch.cuda.synchronize()
            tick=time.perf_counter()
            with torch.no_grad():
                output,final_state=chunk.chunk_gated_delta_rule(ep['q'],ep['k'],ep['v'],ep['raw_g'],ep['beta'],
                    scale=saved['scale'],initial_state=initial,output_final_state=False,
                    use_qk_l2norm_in_kernel=False)
            torch.cuda.synchronize()
            seconds=time.perf_counter()-tick
            assert final_state is None
            terms={key:float(_token_effect(saved['outputs'][key].to('cuda'),ep['raw_g' if key=='g' else key]).sum())
                for key in ('q','k','v','g','beta')}
            native_effect=float(_token_effect(do,output).sum())
            cast_effect=float(_token_effect(do,output.to(torch.bfloat16)).sum())
            report['groups'].append(dict(head_start=saved['head_start'],seconds=seconds,
                initial_state_shape=list(initial.shape),initial_state_dtype=str(initial.dtype),
                initial_state_pair_equal=torch.equal(initial[0],initial[1]),
                input_dtypes={k:str(ep[k].dtype) for k in ('q','k','v','raw_g','beta')},
                output_dtype=str(output.dtype),do_dtype=str(do.dtype),
                finite_joint_terms=terms,finite_joint_contraction=sum(terms.values()),
                native_output_effect=native_effect,native_output_cast_BF16_effect=cast_effect,
                finite_minus_native=sum(terms.values())-native_effect,
                group_end_allocated_bytes=torch.cuda.memory_allocated(),
                pss_bytes=psutil.Process().memory_full_info().pss,artifact=artifact))
            report['summary']=dict(finite_joint_contraction=sum(g['finite_joint_contraction'] for g in report['groups']),
                native_output_effect=sum(g['native_output_effect'] for g in report['groups']),
                native_output_cast_BF16_effect=sum(g['native_output_cast_BF16_effect'] for g in report['groups']))
            save()
            del ep,initial,do,output,saved
        report.update(phase='complete')
        save()
    except BaseException:
        import traceback
        report.update(phase='failed',traceback=traceback.format_exc())
        save()
        raise


if __name__=='__main__':
    main()
