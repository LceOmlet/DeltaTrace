"""Passively contract original joint coefficients with actual single-EOS states.

Two original producer DT calls on the same saved B4. No observer argument,
operator replacement, credit correction, update or model forward is inserted.
The diagnostic contractions are scalars only; original return objects pass
through unchanged. Native cached identity-row differences are also recorded.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import time

import torch

from inspect_extreme_endpoint import CASES, check_imports, geometry, load_request, sha


def pair_differences(value):
    if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != 8:
        return None
    result = []
    for row in range(4):
        a=value[2*row].detach();b=value[2*row+1].detach()
        delta=b.float()-a.float()
        result.append(dict(row=row, equal=torch.equal(a,b), maxabs=float(delta.abs().max()),
            l2=float(delta.double().square().sum().sqrt())))
    return dict(shape=list(value.shape),dtype=str(value.dtype),pairs=result)


def make_worker():
    import ray
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    @ray.remote
    class LayerEffectWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def inspect_layer_effect(self, source_path, output, case_path=None):
            import psutil
            import reward_readout
            from deltatrace_rollout import DeltaTraceRolloutProducer
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu,offload_fsdp_model_to_cpu

            binding=json.loads(Path(case_path).read_bytes()) if case_path is not None else None
            case=CASES['appworld'] if binding is None else binding['case']
            probe_layers=set(() if binding is None else binding.get('probe_decoder_layers',()))
            probe_gdn=set(() if binding is None else binding.get('probe_gdn_layers',()))
            stop_after=None if binding is None else binding.get('stop_after_decoder')
            class ProbeBoundaryReached(Exception):
                """Stop a diagnostic after the explicitly requested boundary."""
            source,native,rows=load_request(source_path,case['native'],case)
            out=Path(output);log=(out/f'rank{self.rank}-phases.jsonl').open('a',buffering=1)
            result=dict(rank=self.rank,pid=os.getpid(),birth=psutil.Process().create_time(),
                owners=check_imports(source),geometry=geometry(source,native,rows,case),
                phases={},native_root=[],operations=dict(DT=2,optimizer=0,backward=0,rollout=0,checkpoint_restore=0))
            def save(phase,**extra):
                result.update(phase=phase,unix=time.time(),**extra)
                (out/f'rank{self.rank}.json').write_text(json.dumps(result,indent=2)+'\n')
            def emit(phase,**extra):
                log.write(json.dumps(dict(phase=phase,unix=time.time(),pss_bytes=psutil.Process().memory_full_info().pss,
                    allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),**extra))+'\n')
            producer=None;globals_=None;handles=[]
            trace_owner=reward_readout.trace_token_attribution
            state=dict(mode='unset',root=False,token_effect_calls=0)
            single={};cross={};index={};native_sub={'single_EOS':{},'original_joint_EOS':{}};gdn_single={};mixer_index={}
            result['decoder_subops']={}
            result['gdn_subops']={}
            try:
                assert self.actor.config.ppo_micro_batch_size_per_gpu==4
                assert (self.config.model.lora_rank,self.config.model.lora_alpha)==(8,16)
                shards=[]
                for name,p in self.actor_module_fsdp.named_parameters():
                    if '.lora_B.' in name:
                        p=p.detach();p=p.to_local() if hasattr(p,'to_local') else p
                        shards.append(dict(name=name,elements=p.numel(),nonzero=int(torch.count_nonzero(p))))
                assert shards and not any(v['nonzero'] for v in shards)
                result['lora_B_local_shards']=shards
                if self._is_offload_param:load_fsdp_model_to_gpu(self.actor_module_fsdp)
                producer=DeltaTraceRolloutProducer(self.actor_module_fsdp,eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id)
                runner=producer.runner;model=runner.model
                globals_=runner.attribute.__func__.__globals__
                decoder_owner=globals_['decoder_finite_pullback'];effect_owner=globals_['_token_effect']
                gdn_owner=globals_['gdn_finite_pullback']
                forward_owner=model.forward_root
                result['finite_owner']=dict(path=inspect.getsourcefile(runner.attribute),sha256=sha(inspect.getsourcefile(runner.attribute)),
                    decoder_path=inspect.getsourcefile(decoder_owner),decoder_sha256=sha(inspect.getsourcefile(decoder_owner)),
                    offload_replay_mixer=runner.offload_replay_mixer)
                expected_runner='628006b637516f8d62e95583a9eb51fe9038ea7931798e2c1f42c28e154cf24f' if binding is None else binding['runner_sha256']
                assert result['finite_owner']['sha256']==expected_runner
                result['case_binding']=binding
                for i,layer in enumerate(model.model.language_model.layers):
                    index[id(layer)]=i
                    if hasattr(layer,'linear_attn'):mixer_index[id(layer.linear_attn)]=i
                    def before(module,args,kwargs,i=i):
                        if not state['root']:return
                        x=args[0] if args else kwargs['hidden_states']
                        comparison=pair_differences(x)
                        if comparison is None:return
                        cache=kwargs.get('past_key_values');cached={}
                        if cache is not None:
                            cached={name:pair_differences(v) for name,v in vars(cache.layers[i]).items() if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==8}
                        result['native_root'].append(dict(mode=state['mode'],layer=i,kind='input',comparison=comparison,cache=cached))
                    def after(module,args,output,i=i):
                        if not state['root']:return
                        comparison=pair_differences(output)
                        if comparison is not None:result['native_root'].append(dict(mode=state['mode'],layer=i,kind='output',comparison=comparison))
                    handles.extend([layer.register_forward_pre_hook(before,with_kwargs=True),layer.register_forward_hook(after)])
                    if i in probe_layers:
                        def retain(name,value,i=i):
                            if state['mode'] not in native_sub or not isinstance(value,torch.Tensor) or value.shape[0]!=8:return
                            row=case['row']
                            native_sub[state['mode']][i,name]=value[2*row:2*row+2].detach().to('cpu',copy=True)
                        for name,module in [('input_norm',layer.input_layernorm),('post_norm',layer.post_attention_layernorm)]:
                            def norm_capture(_module,args,value,name=name,retain=retain):
                                retain(name+'_input',args[0]);retain(name+'_output',value)
                            handles.append(module.register_forward_hook(norm_capture))
                        def decoder_capture(_module,_args,value,retain=retain):retain('decoder_output',value)
                        handles.append(layer.register_forward_hook(decoder_capture))
                def forward(*args,**kwargs):
                    previous=state['root'];state['root']=True
                    try:return forward_owner(*args,**kwargs)
                    finally:state['root']=previous
                model.forward_root=forward
                def boundary(layer,coefficients,endpoints):
                    with torch.no_grad():
                        row=case['row'];pair=endpoints[2*row:2*row+2]
                        if state['mode']=='single_EOS':
                            single[layer]=pair.detach().to('cpu',copy=True)
                        else:
                            actual=single.pop(layer).to(coefficients.device)
                            if pair.shape != actual.shape:raise ValueError('Single and joint owner boundary axes differ')
                            comparison=pair_differences(endpoints)
                            cross[str(layer)]=dict(boundary=layer,
                                joint_coefficient_times_single_deletion_delta=float(effect_owner(coefficients[row:row+1],actual).sum()),
                                joint_coefficient_times_joint_delta=float(effect_owner(coefficients[row:row+1],pair).sum()),
                                factual_endpoints_equal=torch.equal(actual[1],pair[1]),
                                factual_endpoint_maxabs=float((actual[1].float()-pair[1].float()).abs().max()),
                                coefficient_dtype=str(coefficients.dtype),endpoint_dtype=str(pair.dtype),shape=list(pair.shape),
                                native_joint_pair_differences=comparison)
                            del actual
                    emit('boundary',mode=state['mode'],layer=layer,remaining_single_endpoint_bytes=sum(v.numel()*v.element_size() for v in single.values()))
                def decoder(*args,**kwargs):
                    i=index[id(args[0])]
                    if i not in probe_layers or state['mode']!='original_joint_EOS':
                        returned=decoder_owner(*args,**kwargs)
                    else:
                        # All numerical operations remain the original owner
                        # calls. Only scalar contractions observe their inputs
                        # and return values on the actual single-delete states.
                        def dot(coefficients,key):
                            endpoints=native_sub['single_EOS'][i,key].to(coefficients.device)
                            row=case['row']
                            return float(effect_owner(coefficients[row:row+1],endpoints).sum())
                        probe=dict(points=[],factual_endpoint_checks={})
                        def point(name,value):
                            probe['points'].append(dict(name=name,coefficient_times_single_delta=value))
                            result['decoder_subops'][str(i)]=probe
                            save('decoder_subop_complete',active_mode=state['mode'],active_decoder=i)
                        point('before_MLP_residual',dot(args[2],'decoder_output'))
                        boundaries=args[4];norm_owner=boundaries.norm_residual;mixer_owner=args[3];calls=[0]
                        def norm(*operands,**options):
                            which=calls[0];calls[0]+=1
                            if which==0:
                                point('after_MLP_before_post_RMSNorm',dot(operands[3],'post_norm_output')+dot(operands[4],'post_norm_input'))
                            value=norm_owner(*operands,**options)
                            point('after_post_RMSNorm_residual' if which==0 else 'after_input_RMSNorm_residual',dot(value,'post_norm_input' if which==0 else 'input_norm_input'))
                            return value
                        def mixer(upstream):
                            value=mixer_owner(upstream)
                            point('after_mixer_before_input_RMSNorm',dot(value[0],'input_norm_output')+dot(upstream,'input_norm_input'))
                            return value
                        wrapped=list(args);wrapped[3]=mixer;boundaries.norm_residual=norm
                        try:returned=decoder_owner(*wrapped,**kwargs)
                        finally:boundaries.norm_residual=norm_owner
                        assert calls[0]==2
                        for key in ('input_norm_input','input_norm_output','post_norm_input','post_norm_output','decoder_output'):
                            a=native_sub['single_EOS'].pop((i,key));b=native_sub['original_joint_EOS'].pop((i,key))
                            probe['factual_endpoint_checks'][key]=dict(equal=torch.equal(a[1],b[1]),maxabs=float((a[1].float()-b[1].float()).abs().max()),shape=list(a.shape),dtype=str(a.dtype))
                        result['decoder_subops'][str(i)]=probe
                    boundary(i,returned[0],args[1]['input_norm_input'])
                    if stop_after is not None and i==stop_after:raise ProbeBoundaryReached()
                    return returned
                def effect(coefficients,endpoints):
                    returned=effect_owner(coefficients,endpoints)
                    if state['token_effect_calls']==0:boundary(32,coefficients,endpoints)
                    state['token_effect_calls']+=1
                    return returned
                globals_['decoder_finite_pullback']=decoder;globals_['_token_effect']=effect
                def gdn(*args,**kwargs):
                    module,c,e,upstream,scale,fla_owner,*rest=args
                    i=mixer_index[id(module)]
                    if i not in probe_gdn:return gdn_owner(*args,**kwargs)
                    row=case['row']
                    current={name:(c if name=='z' else e)[name][2*row:2*row+2].detach().to('cpu',copy=True)
                             for name in ('o','z','q','k','v','raw_g','beta')}
                    if state['mode']=='single_EOS':
                        gdn_single[i]=current
                        return gdn_owner(*args,**kwargs)
                    actual=gdn_single[i]
                    probe=dict(points=[],factual_endpoint_checks={},fla_groups=[],operand_artifacts=[],
                        owner=dict(path=inspect.getsourcefile(gdn_owner),sha256=sha(inspect.getsourcefile(gdn_owner))),
                        effective_options={k:v for k,v in kwargs.items() if k not in ('norm_gate_pullback','conv_silu_pullback','key_norm_pullback')})
                    for name,value in current.items():
                        a=actual[name]
                        probe['factual_endpoint_checks'][name]=dict(equal=torch.equal(a[1],value[1]),
                            maxabs=float((a[1].float()-value[1].float()).abs().max()),shape=list(a.shape),dtype=str(a.dtype))
                    del current
                    def dot(coefficients,endpoints):
                        return float(effect_owner(coefficients[row:row+1],endpoints.to(coefficients.device)).sum())
                    skip=dot(upstream,native_sub['single_EOS'][i,'input_norm_input'])
                    context=dict(z_credit=None,head_start=0,fla_credit=0.0)
                    def point(name,value):
                        probe['points'].append(dict(name=name,coefficient_times_single_delta=value))
                        result['gdn_subops'][str(i)]=probe
                        save('gdn_subop_complete',active_mode=state['mode'],active_decoder=i)
                    gate_owner=kwargs.get('norm_gate_pullback') or gdn_owner.__globals__['_norm_gate_finite_rule']
                    def gate(*operands,**options):
                        returned=gate_owner(*operands,**options)
                        context['z_credit']=dot(returned[1],actual['z'])
                        point('after_gdn_norm_and_silu_gate',skip+context['z_credit']+dot(returned[0],actual['o']))
                        return returned
                    def fla(endpoints,do,scale):
                        returned=fla_owner(endpoints,do,scale)
                        start=actual['q'].shape[1]-endpoints['q'].shape[1]
                        head=context['head_start'];heads=endpoints['q'].shape[2]
                        terms={name:dot(returned[name],actual['raw_g' if name=='g' else name][:,start:,head:head+heads])
                               for name in ('q','k','v','g','beta')}
                        context['fla_credit']+=sum(terms.values());context['head_start']+=heads
                        probe['fla_groups'].append(dict(head_start=head,heads=heads,time_start=start,
                            terms=terms,input_dtype=str(endpoints['q'].dtype),do_dtype=str(do.dtype),
                            output_dtypes={k:str(v.dtype) for k,v in returned.items()}))
                        if self.rank==0:
                            # Exact original finite-op operands, selected rows only,
                            # allow a later numerical replay without another model run.
                            path=out/f'rank0-decoder{i}-finite-fla-head{head}.pt'
                            payload=dict(endpoints={k:v[2*row:2*row+2].detach().to('cpu',copy=True) for k,v in endpoints.items()},
                                do=do[row:row+1].detach().to('cpu',copy=True),scale=scale,
                                outputs={k:v[row:row+1].detach().to('cpu',copy=True) for k,v in returned.items()},
                                actual_single={k:actual['raw_g' if k=='g' else k][:,start:,head:head+heads] for k in terms},
                                row=row,head_start=head,time_start=start,
                                scope='Selected original B4 finite operator inputs and outputs, not a new B1 execution')
                            torch.save(payload,path)
                            probe['operand_artifacts'].append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
                            del payload
                        save('gdn_finite_fla_group_complete',active_mode=state['mode'],active_decoder=i)
                        emit('gdn_finite_fla_group_complete',decoder=i,head_start=head,heads=heads)
                        return returned
                    wrapped=list(args);wrapped[5]=fla
                    returned=gdn_owner(*wrapped,**dict(kwargs,norm_gate_pullback=gate))
                    assert context['head_start']==actual['q'].shape[2]
                    point('after_finite_fla_before_qk_l2',skip+context['z_credit']+context['fla_credit'])
                    probe['residual_skip_single_delta']=skip
                    probe['z_branch_single_delta']=context['z_credit']
                    del gdn_single[i]
                    result['gdn_subops'][str(i)]=probe
                    return returned
                if probe_gdn:globals_['gdn_finite_pullback']=gdn
                def trace(*args,**kwargs):
                    if state['mode']=='single_EOS':
                        owner,reference,factual,*rest=args
                        reference=factual.clone();reference[case['row'],case['packed_slot']]=self.tokenizer.eos_token_id
                        args=(owner,reference,factual,*rest)
                    returned=trace_owner(*args,**kwargs)
                    result['phases'][state['mode']]=dict(candidate_signed=float(returned[0][case['row'],case['packed_slot']]),
                        detail=returned[2],full_signed_shape=list(returned[0].shape),full_signed_dtype=str(returned[0].dtype))
                    torch.save(returned[0].detach().cpu(),out/f'rank{self.rank}-{state["mode"]}-signed.pt')
                    return returned
                reward_readout.trace_token_attribution=trace
                save('original_owner_ready')
                for mode in ('single_EOS','original_joint_EOS'):
                    state.update(mode=mode,token_effect_calls=0)
                    emit('DT_begin',mode=mode);tick=time.perf_counter()
                    try:producer.attribute_episodes([[row['row'] for row in rows]],[0.0])
                    except ProbeBoundaryReached:
                        result['phases'][mode]=dict(partial=True,stopped_after_decoder=stop_after,
                            scope='Native root and original finite propagation through the requested decoder only; no full signed vector or advantages are produced.')
                        # The stop occurs before the runner's normal per-layer
                        # release. Use its own parameter-release API before the
                        # second diagnostic call; no cache rebuild is performed.
                        model.release_owner_params()
                    torch.cuda.synchronize();result['phases'][mode]['seconds']=time.perf_counter()-tick
                    result['phases'][mode]['remaining_snapshot_bytes']=sum(v.numel()*v.element_size() for v in single.values())
                    save('DT_complete',active_mode=mode,cross_boundary_contractions=cross)
                assert not single and len(cross)==(33 if stop_after is None else 33-stop_after)
                if probe_layers:assert not any(native_sub.values())
                assert not gdn_single
                save('complete',cross_boundary_contractions=cross,
                    scope='Passive diagnostics of original coefficients and actual single-deletion hidden differences. No new learning signal, acceptance tolerance, observer-mode path, credit change or update.')
                return dict(rank=self.rank,completed=True,optimizer_steps=0)
            except BaseException:
                import traceback
                save('failed',traceback=traceback.format_exc());raise
            finally:
                reward_readout.trace_token_attribution=trace_owner
                if globals_ is not None:
                    globals_['decoder_finite_pullback']=decoder_owner;globals_['_token_effect']=effect_owner
                    if probe_gdn:globals_['gdn_finite_pullback']=gdn_owner
                for handle in handles:handle.remove()
                single.clear()
                for values in native_sub.values():values.clear()
                gdn_single.clear()
                if producer is not None:
                    producer.runner.model.forward_root=forward_owner
                    producer.runner.model.release_owner_params()
                if self._is_offload_param:offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log.close()
    return LayerEffectWorker


def main():
    import ray
    from omegaconf import OmegaConf
    from verl.single_controller.ray import RayClassWithInitArgs,RayResourcePool,RayWorkerGroup
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--case',type=Path);a=parser.parse_args()
    source=json.loads(a.source.read_bytes());check_imports(source);assert sha(a.source)==CASES['appworld']['source_sha256']
    a.output.mkdir(exist_ok=False)
    config=OmegaConf.load(Path(source['verl_root'])/'verl/trainer/config/ppo_trainer.yaml')
    for key,value in source['startup_options'].items():OmegaConf.update(config,key.lstrip('+'),value,force_add=True)
    config.actor_rollout_ref.actor.optim.total_training_steps=config.trainer.total_training_steps
    (a.output/'effective-config.yaml').write_text(OmegaConf.to_yaml(config))
    ray.init(num_cpus=8,include_dashboard=False)
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),RayClassWithInitArgs(make_worker(),config.actor_rollout_ref,'actor'))
        group.init_model();v=group.inspect_layer_effect(str(a.source),str(a.output),str(a.case) if a.case is not None else None)
        (a.output/'completed.json').write_text(json.dumps(v,indent=2)+'\n')
    finally:ray.shutdown()


if __name__=='__main__':main()
