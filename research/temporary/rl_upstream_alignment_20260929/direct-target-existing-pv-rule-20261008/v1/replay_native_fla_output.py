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
    p.add_argument('--observe-memory-orders',action='store_true')
    p.add_argument('--conditional-v-only',action='store_true')
    p.add_argument('--factual-jacobian',action='store_true')
    a=p.parse_args()
    assert sum((a.single_deletion,a.observe_memory_orders,a.conditional_v_only,a.factual_jacobian))<=1
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
    if a.observe_memory_orders:
        assert not a.single_deletion
        report['scope']+=' Passive observation of the two original memory callback orientations, returning every original coefficient unchanged; no profile replacement.'
    def save():
        (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        if a.conditional_v_only:
            report['scope']='Conditional V diagnostic only: saved actual single-deletion V pair, all other FLA inputs fixed to the original factual endpoint, actual shared factual incoming state. Original public FLA and existing forward callback only; no model, full DT, update, rollout, new credit rule or tolerance.'
            for artifact in native['operand_artifacts']:
                path=Path(artifact['path'])
                assert sha(path)==artifact['sha256']
                saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                ep={k:v.to('cuda') for k,v in saved['endpoints'].items()}
                offset=saved['actual_single_time_start']-saved['time_start']
                assert offset>=0 and offset%64==0
                actual_v=saved['actual_single']['v'].to('cuda')
                assert torch.equal(actual_v[1],ep['v'][1,offset:])
                operands={name:ep[name][1:2,offset:].expand(2,*ep[name][1:2,offset:].shape[1:]).contiguous()
                          for name in ('q','k','raw_g','beta')}
                assert all(torch.equal(value[0],value[1]) for value in operands.values())
                operands['v']=actual_v
                initial=ep['h'][1:2,offset//64].expand(2,*ep['h'][1:2,offset//64].shape[1:]).contiguous()
                do=saved['do'].to('cuda')
                torch.cuda.synchronize();tick=time.perf_counter()
                with torch.no_grad():
                    output,state=chunk.chunk_gated_delta_rule(operands['q'],operands['k'],operands['v'],operands['raw_g'],operands['beta'],scale=saved['scale'],initial_state=initial,output_final_state=False,use_qk_l2norm_in_kernel=False)
                    forward=finite_fla_pullback(ep,do,saved['scale'])
                torch.cuda.synchronize()
                assert state is None
                native_effect=float(_token_effect(do[:,offset:],output).sum())
                forward_effect=float(_token_effect(forward['v'][:,offset:],actual_v).sum())
                averaged_effect=float(_token_effect(saved['outputs']['v'][:,offset:].to('cuda'),actual_v).sum())
                report['groups'].append(dict(head_start=saved['head_start'],seconds=time.perf_counter()-tick,
                    native_conditional_V_effect=native_effect,
                    original_forward_V_times_actual_single_delta=forward_effect,
                    original_symmetric_V_times_actual_single_delta=averaged_effect,
                    forward_minus_native=forward_effect-native_effect,
                    symmetric_minus_native=averaged_effect-native_effect,
                    all_other_inputs_factual_pair_equal=True,factual_V_exact_equal=True,
                    actual_single_suffix_offset=offset,initial_state_pair_equal=torch.equal(initial[0],initial[1]),
                    input_dtypes={key:str(value.dtype) for key,value in operands.items()},
                    output_dtype=str(output.dtype),do_dtype=str(do.dtype),
                    peak_live_allocated_bytes=torch.cuda.max_memory_allocated(),pss_bytes=psutil.Process().memory_full_info().pss,
                    artifact=artifact))
                report['summary']={key:sum(g[key] for g in report['groups']) for key in ('native_conditional_V_effect','original_forward_V_times_actual_single_delta','original_symmetric_V_times_actual_single_delta','forward_minus_native','symmetric_minus_native')}
                save()
                del ep,saved,actual_v,operands,initial,do,output,forward
            report['phase']='complete';save();return
        if a.factual_jacobian:
            report['scope']='Original native FLA factual-endpoint VJP on the saved actual single-deletion operand pair, using the original saved output cotangent and shared factual incoming state. Compare its contraction with the complete native single-deletion effect. Operator-only diagnosis, no model/full DT/credit/profile/optimizer/rollout or tolerance change.'
            for artifact in native['operand_artifacts']:
                path=Path(artifact['path']);assert sha(path)==artifact['sha256']
                saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                offset=saved['actual_single_time_start']-saved['time_start']
                assert offset>=0 and offset%64==0
                ep={('raw_g' if key=='g' else key):value.to('cuda').requires_grad_(True)
                    for key,value in saved['actual_single'].items()}
                factual_equal={key:torch.equal(value[1],saved['endpoints'][key][1,offset:].to('cuda'))
                    for key,value in ep.items()}
                assert all(factual_equal.values())
                initial=saved['endpoints']['h'][1:2,offset//64].to('cuda').expand(2,-1,-1,-1).contiguous()
                assert torch.equal(initial[0],initial[1])
                do=saved['do'][:,offset:].to('cuda').contiguous()
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
                output,state=chunk.chunk_gated_delta_rule(ep['q'],ep['k'],ep['v'],ep['raw_g'],ep['beta'],scale=saved['scale'],initial_state=initial,output_final_state=False,use_qk_l2norm_in_kernel=False)
                assert state is None and output.requires_grad
                incoming=torch.cat((torch.zeros_like(do),do),dim=0)
                torch.autograd.backward(output,grad_tensors=incoming)
                torch.cuda.synchronize()
                terms={key:float(_token_effect(ep['raw_g' if key=='g' else key].grad[1:2],ep['raw_g' if key=='g' else key].detach()).sum())
                    for key in ('q','k','v','g','beta')}
                reference_gradient_maxabs={key:float(value.grad[0].abs().max()) for key,value in ep.items()}
                actual_native=float(_token_effect(do,output.detach()).sum())
                report['groups'].append(dict(head_start=saved['head_start'],seconds=time.perf_counter()-tick,
                    original_native_forward_calls=1,original_native_backward_calls=1,
                    factual_operands_equal=factual_equal,initial_state_pair_equal=True,
                    input_dtypes={key:str(value.dtype) for key,value in ep.items()},
                    gradient_dtypes={key:str(value.grad.dtype) for key,value in ep.items()},
                    gradients_all_finite=all(bool(torch.isfinite(value.grad).all()) for value in ep.values()),
                    reference_gradient_maxabs=reference_gradient_maxabs,
                    native_single_deletion_output_effect=actual_native,
                    factual_native_VJP_times_actual_delta_terms=terms,
                    factual_native_VJP_times_actual_delta=sum(terms.values()),
                    local_linearization_minus_native=sum(terms.values())-actual_native,
                    peak_live_allocated_bytes=torch.cuda.max_memory_allocated(),
                    pss_bytes=psutil.Process().memory_full_info().pss,artifact=artifact))
                report['summary']={key:sum(group[key] for group in report['groups']) for key in ('native_single_deletion_output_effect','factual_native_VJP_times_actual_delta','local_linearization_minus_native')}
                report['summary']['factual_native_VJP_terms']={key:sum(group['factual_native_VJP_times_actual_delta_terms'][key] for group in report['groups']) for key in terms}
                save()
                del ep,saved,initial,do,output,incoming
            report['phase']='complete';save();return
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
            if a.observe_memory_orders:
                single_offset=saved['actual_single_time_start']-saved['time_start']
                actual={k:v.to('cuda') for k,v in saved['actual_single'].items()}
                orders=[]
                def observe_order(endpoints,upstream,scale):
                    values=finite_fla_pullback(endpoints,upstream,scale)
                    ordered_terms={key:float(_token_effect(values[key][:,single_offset:],actual[key]).sum())
                        for key in ('q','k','v','g','beta')}
                    orders.append(dict(orientation='forward' if not orders else 'reversed',
                        terms=ordered_terms,coefficient_times_actual_single_delta=sum(ordered_terms.values())))
                    return values
                with torch.no_grad():
                    averaged=average_memory_endpoint_orders(observe_order)(ep,do,saved['scale'])
                assert len(orders)==2
                averaged_value=sum(float(_token_effect(averaged[key][:,single_offset:],actual[key]).sum())
                    for key in ('q','k','v','g','beta'))
                report['groups'][-1].update(memory_orders=orders,
                    original_averaged_coefficients_times_single_delta=averaged_value)
                del actual,averaged
            report['summary']=dict(finite_joint_contraction=sum(g['finite_joint_contraction'] for g in report['groups']),
                native_output_effect=sum(g['native_output_effect'] for g in report['groups']),
                native_output_cast_BF16_effect=sum(g['native_output_cast_BF16_effect'] for g in report['groups']))
            if a.single_deletion:
                report['summary']['original_single_finite_contraction']=sum(g['original_single_finite_contraction'] for g in report['groups'])
            if a.observe_memory_orders:
                report['summary']['memory_orders']={name:sum(g['memory_orders'][i]['coefficient_times_actual_single_delta'] for g in report['groups'])
                    for i,name in enumerate(('forward','reversed'))}
                report['summary']['original_averaged_coefficients_times_single_delta']=sum(g['original_averaged_coefficients_times_single_delta'] for g in report['groups'])
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
