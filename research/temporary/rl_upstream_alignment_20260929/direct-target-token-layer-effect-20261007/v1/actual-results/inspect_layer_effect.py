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
        def inspect_layer_effect(self, source_path, output):
            import psutil
            import reward_readout
            from deltatrace_rollout import DeltaTraceRolloutProducer
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu,offload_fsdp_model_to_cpu

            case=CASES['appworld'];source,native,rows=load_request(source_path,case['native'],case)
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
            single={};cross={};index={}
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
                forward_owner=model.forward_root
                result['finite_owner']=dict(path=inspect.getsourcefile(runner.attribute),sha256=sha(inspect.getsourcefile(runner.attribute)),
                    decoder_path=inspect.getsourcefile(decoder_owner),decoder_sha256=sha(inspect.getsourcefile(decoder_owner)),
                    offload_replay_mixer=runner.offload_replay_mixer)
                assert result['finite_owner']['sha256']=='628006b637516f8d62e95583a9eb51fe9038ea7931798e2c1f42c28e154cf24f'
                for i,layer in enumerate(model.model.language_model.layers):
                    index[id(layer)]=i
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
                def forward(*args,**kwargs):
                    previous=state['root'];state['root']=True
                    try:return forward_owner(*args,**kwargs)
                    finally:state['root']=previous
                model.forward_root=forward
                def boundary(layer,coefficients,endpoints):
                    with torch.no_grad():
                        pair=endpoints[:2]
                        if state['mode']=='single_EOS':
                            single[layer]=pair.detach().to('cpu',copy=True)
                        else:
                            actual=single.pop(layer).to(coefficients.device)
                            if pair.shape != actual.shape:raise ValueError('Single and joint owner boundary axes differ')
                            comparison=pair_differences(endpoints)
                            cross[str(layer)]=dict(boundary=layer,
                                joint_coefficient_times_single_deletion_delta=float(effect_owner(coefficients[:1],actual).sum()),
                                joint_coefficient_times_joint_delta=float(effect_owner(coefficients[:1],pair).sum()),
                                factual_endpoints_equal=torch.equal(actual[1],pair[1]),
                                factual_endpoint_maxabs=float((actual[1].float()-pair[1].float()).abs().max()),
                                coefficient_dtype=str(coefficients.dtype),endpoint_dtype=str(pair.dtype),shape=list(pair.shape),
                                native_joint_pair_differences=comparison)
                            del actual
                    emit('boundary',mode=state['mode'],layer=layer,remaining_single_endpoint_bytes=sum(v.numel()*v.element_size() for v in single.values()))
                def decoder(*args,**kwargs):
                    returned=decoder_owner(*args,**kwargs)
                    boundary(index[id(args[0])],returned[0],args[1]['input_norm_input'])
                    return returned
                def effect(coefficients,endpoints):
                    returned=effect_owner(coefficients,endpoints)
                    if state['token_effect_calls']==0:boundary(32,coefficients,endpoints)
                    state['token_effect_calls']+=1
                    return returned
                globals_['decoder_finite_pullback']=decoder;globals_['_token_effect']=effect
                def trace(*args,**kwargs):
                    if state['mode']=='single_EOS':
                        owner,reference,factual,*rest=args
                        reference=factual.clone();reference[0,case['packed_slot']]=self.tokenizer.eos_token_id
                        args=(owner,reference,factual,*rest)
                    returned=trace_owner(*args,**kwargs)
                    result['phases'][state['mode']]=dict(candidate_signed=float(returned[0][0,case['packed_slot']]),
                        detail=returned[2],full_signed_shape=list(returned[0].shape),full_signed_dtype=str(returned[0].dtype))
                    torch.save(returned[0].detach().cpu(),out/f'rank{self.rank}-{state["mode"]}-signed.pt')
                    return returned
                reward_readout.trace_token_attribution=trace
                save('original_owner_ready')
                for mode in ('single_EOS','original_joint_EOS'):
                    state.update(mode=mode,token_effect_calls=0)
                    emit('DT_begin',mode=mode);tick=time.perf_counter()
                    producer.attribute_episodes([[row['row'] for row in rows]],[0.0])
                    torch.cuda.synchronize();result['phases'][mode]['seconds']=time.perf_counter()-tick
                    result['phases'][mode]['remaining_snapshot_bytes']=sum(v.numel()*v.element_size() for v in single.values())
                    save('DT_complete',active_mode=mode,cross_boundary_contractions=cross)
                assert not single and len(cross)==33
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
                for handle in handles:handle.remove()
                single.clear()
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
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    source=json.loads(a.source.read_bytes());check_imports(source);assert sha(a.source)==CASES['appworld']['source_sha256']
    a.output.mkdir(exist_ok=False)
    config=OmegaConf.load(Path(source['verl_root'])/'verl/trainer/config/ppo_trainer.yaml')
    for key,value in source['startup_options'].items():OmegaConf.update(config,key.lstrip('+'),value,force_add=True)
    config.actor_rollout_ref.actor.optim.total_training_steps=config.trainer.total_training_steps
    (a.output/'effective-config.yaml').write_text(OmegaConf.to_yaml(config))
    ray.init(num_cpus=8,include_dashboard=False)
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),RayClassWithInitArgs(make_worker(),config.actor_rollout_ref,'actor'))
        group.init_model();v=group.inspect_layer_effect(str(a.source),str(a.output))
        (a.output/'completed.json').write_text(json.dumps(v,indent=2)+'\n')
    finally:ray.shutdown()


if __name__=='__main__':main()
