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
    p.add_argument('--single-deletion',action='store_true')
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
    from finite_fla_gpu import verify_native_sources, finite_fla_pullback
    from profiles.qwen35_gdn_symmetric import average_memory_endpoint_orders
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
    if a.single_deletion:
        report['scope']='Original public native FLA and original symmetric finite callback on saved actual single-deletion operands. Both endpoints reuse the captured factual incoming state before the only changed token, by the original causal-prefix contract. No model/full-DT/actor/rollout/training-backward/update; no estimated or zero initial state.'
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
            offset=0
            if a.single_deletion:
                offset=saved['actual_single_time_start']-saved['time_start']
                assert offset>=0 and offset%64==0
                single={('raw_g' if k=='g' else k):v.to('cuda') for k,v in saved['actual_single'].items()}
                factual_equal={k:torch.equal(single[k][1],ep[k][1,offset:]) for k in single}
                assert all(factual_equal.values())
                # The saved single capture starts at the unchanged chunk boundary
                # preceding its only changed token. This is the actual factual
                # state at that boundary, shared by these two causal histories.
                initial=ep['h'][1:2,offset//64].expand(2,-1,-1,-1).contiguous()
                del ep
                ep=single
                do=do[:,offset:].contiguous()
            captured={}
            def observe(frame,event,value):
                if event=='return' and frame.f_code is chunk.chunk_gated_delta_rule_fwd.__code__:
                    f=frame.f_locals
                    captured.update({k:f[k] for k in ('q','k','v','g','beta','A','w','v_new','h')})
                    captured['raw_g']=ep['raw_g']
            torch.cuda.synchronize()
            tick=time.perf_counter()
            assert sys.getprofile() is None
            if a.single_deletion:sys.setprofile(observe)
            try:
                with torch.no_grad():
                    output,final_state=chunk.chunk_gated_delta_rule(ep['q'],ep['k'],ep['v'],ep['raw_g'],ep['beta'],
                        scale=saved['scale'],initial_state=initial,output_final_state=False,
                        use_qk_l2norm_in_kernel=False)
            finally:
                if a.single_deletion:sys.setprofile(None)
            torch.cuda.synchronize()
            seconds=time.perf_counter()-tick
            assert final_state is None
            terms={key:float(_token_effect(saved['outputs'][key][:,offset:].to('cuda'),ep['raw_g' if key=='g' else key]).sum())
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
            if a.single_deletion:
                assert captured
                with torch.no_grad():
                    actual_coeff=average_memory_endpoint_orders(finite_fla_pullback)(captured,do,saved['scale'])
                actual_terms={key:float(_token_effect(actual_coeff[key],ep['raw_g' if key=='g' else key]).sum())
                    for key in terms}
                report['groups'][-1].update(factual_operands_equal=factual_equal,
                    original_single_capture_offset=offset,
                    original_single_finite_terms=actual_terms,
                    original_single_finite_contraction=sum(actual_terms.values()),
                    original_single_finite_minus_native=sum(actual_terms.values())-native_effect)
                del actual_coeff,captured,single
            report['summary']=dict(finite_joint_contraction=sum(g['finite_joint_contraction'] for g in report['groups']),
                native_output_effect=sum(g['native_output_effect'] for g in report['groups']),
                native_output_cast_BF16_effect=sum(g['native_output_cast_BF16_effect'] for g in report['groups']))
            if a.single_deletion:
                report['summary']['original_single_finite_contraction']=sum(g['original_single_finite_contraction'] for g in report['groups'])
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
