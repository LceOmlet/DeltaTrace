"""Observe head and decoder27 original operands on two existing DT calls.
No observer-mode path, new forward, operator replacement or optimizer update.
"""
import inspect
import json
import os
from pathlib import Path
import time
import torch
import inspect_layer_effect as entry
from inspect_extreme_endpoint import CASES,check_imports,geometry,load_request,sha


def make_worker():
    import ray
    from verl.single_controller.base.decorator import Dispatch,register
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    @ray.remote
    class OperatorWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def inspect_layer_effect(self,source_path,output):
            import psutil,reward_readout,qwen35_answer_finite
            from deltatrace_rollout import DeltaTraceRolloutProducer
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu,offload_fsdp_model_to_cpu
            spec=CASES['appworld'];source,native,rows=load_request(source_path,spec['native'],spec)
            out=Path(output);log=(out/f'rank{self.rank}-phases.jsonl').open('a',buffering=1)
            r=dict(rank=self.rank,pid=os.getpid(),birth=psutil.Process().create_time(),owners=check_imports(source),geometry=geometry(source,native,rows,spec),modes={},operations=dict(DT=2,rollout=0,backward=0,optimizer=0,checkpoint_restore=0))
            state=dict(mode=None,decoder27=False,norm_calls=0,effect_calls=0)
            snapshots={};norm_native={};head_native={};hooks=[];producer=None;g=None
            trace_owner=reward_readout.trace_token_attribution
            answer_owner=qwen35_answer_finite.FiniteAnswerOps.__call__
            def emit(phase,**extra):
                log.write(json.dumps(dict(phase=phase,unix=time.time(),pss_bytes=psutil.Process().memory_full_info().pss,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),mode=state['mode'],**extra))+'\n')
            def save(phase):
                r.update(phase=phase,unix=time.time());(out/f'rank{self.rank}.json').write_text(json.dumps(r,indent=2)+'\n')
            def cp(v):return v.detach().to('cpu',copy=True)
            def contract(m,pair):
                with torch.no_grad():return float(effect_owner(m[:1],pair.to(m.device)).sum())
            try:
                assert self.actor.config.ppo_micro_batch_size_per_gpu==4
                assert (self.config.model.lora_rank,self.config.model.lora_alpha)==(8,16)
                lora=[]
                for name,p in self.actor_module_fsdp.named_parameters():
                    if '.lora_B.' in name:
                        p=p.detach();p=p.to_local() if hasattr(p,'to_local') else p
                        lora.append(dict(name=name,elements=p.numel(),nonzero=int(torch.count_nonzero(p))))
                assert lora and not any(v['nonzero'] for v in lora);r['lora_B_local_shards']=lora
                if self._is_offload_param:load_fsdp_model_to_gpu(self.actor_module_fsdp)
                producer=DeltaTraceRolloutProducer(self.actor_module_fsdp,eos_token_id=self.tokenizer.eos_token_id,pad_token_id=self.tokenizer.pad_token_id)
                runner=producer.runner;model=runner.model;g=runner.attribute.__func__.__globals__
                decoder_owner=g['decoder_finite_pullback'];effect_owner=g['_token_effect'];fa_owner=runner.finite_fa
                r['runner']=dict(path=inspect.getsourcefile(runner.attribute),sha256=sha(inspect.getsourcefile(runner.attribute)),offload_replay_mixer=runner.offload_replay_mixer)
                assert r['runner']['sha256']=='628006b637516f8d62e95583a9eb51fe9038ea7931798e2c1f42c28e154cf24f'
                norm_owner=runner.boundaries.norm_residual;mlp_owner=runner.boundaries.mlp
                layer=model.model.language_model.layers[27]
                def final_norm_hook(module,args,output):
                    if output.shape[0]==8:head_native['norm_output']=cp(output[:2])
                hooks.append(model.model.language_model.norm.register_forward_hook(final_norm_hook))
                for name,module in [('input_norm',layer.input_layernorm),('post_norm',layer.post_attention_layernorm)]:
                    def hook(module,args,output,name=name):
                        if state['mode']=='single_EOS' and output.shape[0]==8:
                            norm_native[name+'_input']=cp(args[0][:2]);norm_native[name+'_output']=cp(output[:2])
                    hooks.append(module.register_forward_hook(hook))
                def output_hook(module,args,value):
                    if state['mode']=='single_EOS' and value.shape[0]==8:norm_native['decoder_output']=cp(value[:2])
                hooks.append(layer.register_forward_hook(output_hook))
                def answer(instance,logits,head,selection,*args,**kwargs):
                    returned=answer_owner(instance,logits,head,selection,*args,**kwargs)
                    chosen=selection.samples.eq(0);indices=chosen.nonzero().flatten();paired=(2*indices[:,None]+torch.arange(2,device=indices.device)[None,:]).flatten()
                    h=head_native.pop('norm_output');positions=selection.positions[chosen].cpu()
                    selected_h=h[:,positions].transpose(0,1).reshape(-1,h.shape[-1]);del h
                    data=dict(logits=cp(logits.index_select(0,paired)),normalized_hidden=selected_h,packed_coefficients=cp(returned[1]['packed_hidden'][chosen]),labels=cp(selection.labels[chosen]),positions=positions,logp0=cp(returned[1]['logp0'][chosen]),logp1=cp(returned[1]['logp1'][chosen]),allocated=cp(returned[1]['allocated_logit_effect'][chosen]),head_weight_shape=list(head.weight.shape),head_weight_dtype=str(head.weight.dtype))
                    torch.save(data,out/f'rank{self.rank}-{state["mode"]}-head.pt');del data
                    emit('head_saved',target_count=int(chosen.sum()));return returned
                qwen35_answer_finite.FiniteAnswerOps.__call__=answer
                def effect(m,x):
                    returned=effect_owner(m,x)
                    if state['effect_calls']==0:
                        if state['mode']=='single_EOS':snapshots['final_norm_input']=cp(x[:2])
                        else:r['modes'][state['mode']]['after_final_norm']=contract(m,snapshots.pop('final_norm_input'))
                    state['effect_calls']+=1;return returned
                def mlp(*args,**kwargs):
                    returned=mlp_owner(*args,**kwargs)
                    if state['decoder27'] and state['mode']=='original_joint_EOS':
                        r['modes'][state['mode']]['decoder27']['mlp_coefficient_times_single_norm_output_delta']=contract(returned,norm_native['post_norm_output'])
                    return returned
                def norm(*args,**kwargs):
                    returned=norm_owner(*args,**kwargs)
                    if state['decoder27'] and state['mode']=='original_joint_EOS':
                        name=('post_norm','input_norm')[state['norm_calls']];state['norm_calls']+=1
                        rr=r['modes'][state['mode']]['decoder27']
                        rr[name]=dict(nonlinear_branch=contract(args[3],norm_native[name+'_output']),residual_branch=contract(args[4],norm_native[name+'_input']),returned_input=contract(returned,norm_native[name+'_input']))
                        if self.rank==0:
                            torch.save(dict(x0=cp(args[0][:1]),x1=cp(args[1][:1]),weight=cp(args[2]),upstream=cp(args[3][:1]),residual=cp(args[4][:1]),returned=cp(returned[:1]),eps=args[5],single_input=norm_native[name+'_input'],single_output=norm_native[name+'_output']),out/f'rank0-decoder27-{name}.pt')
                    return returned
                def fa(ops,*args,**kwargs):
                    returned=fa_owner(ops,*args,**kwargs)
                    if state['decoder27'] and state['mode']=='original_joint_EOS' and self.rank==0:
                        data=dict(ops={k:cp(v) for k,v in ops.items()},coefficients={k:cp(v) for k,v in returned.items() if isinstance(v,torch.Tensor)},scale=args[0],layout=args[1],owner_path=inspect.getsourcefile(fa_owner if inspect.isfunction(fa_owner) else type(fa_owner)),owner_sha256=sha(inspect.getsourcefile(fa_owner if inspect.isfunction(fa_owner) else type(fa_owner))))
                        torch.save(data,out/'rank0-decoder27-fa.pt');del data;emit('FA27_actual_operands_saved')
                    return returned
                def decoder(*args,**kwargs):
                    if args[0] is not layer:return decoder_owner(*args,**kwargs)
                    state.update(decoder27=True,norm_calls=0)
                    try:
                        if state['mode']=='single_EOS':
                            assert torch.equal(norm_native['input_norm_input'],args[1]['input_norm_input'][:2].cpu())
                            assert torch.equal(norm_native['post_norm_input'],args[1]['post_norm_input'][:2].cpu())
                        else:
                            r['modes'][state['mode']]['decoder27']=dict(output=contract(args[2],norm_native['decoder_output']))
                        old_mixer=args[3]
                        def mixer(upstream):
                            returned=old_mixer(upstream)
                            if state['mode']=='original_joint_EOS':r['modes'][state['mode']]['decoder27']['after_mixer']=contract(returned[0],norm_native['input_norm_output'])
                            return returned
                        returned=decoder_owner(*args[:3],mixer,*args[4:],**kwargs)
                        if state['mode']=='original_joint_EOS':
                            assert state['norm_calls']==2
                            r['modes'][state['mode']]['decoder27']['input']=contract(returned[0],norm_native['input_norm_input'])
                            norm_native.clear();save('decoder27_complete');emit('decoder27_complete')
                        return returned
                    finally:state['decoder27']=False
                g['decoder_finite_pullback']=decoder;g['_token_effect']=effect
                runner.boundaries.mlp=mlp;runner.boundaries.norm_residual=norm;runner.finite_fa=fa
                def trace(*args,**kwargs):
                    if state['mode']=='single_EOS':
                        owner,reference,factual,*rest=args;reference=factual.clone();reference[0,spec['packed_slot']]=self.tokenizer.eos_token_id;args=(owner,reference,factual,*rest)
                    returned=trace_owner(*args,**kwargs)
                    r['modes'][state['mode']].update(candidate_signed=float(returned[0][0,spec['packed_slot']]),detail=returned[2])
                    torch.save(returned[0].detach().cpu(),out/f'rank{self.rank}-{state["mode"]}-signed.pt');return returned
                reward_readout.trace_token_attribution=trace
                for mode in ('single_EOS','original_joint_EOS'):
                    state.update(mode=mode,effect_calls=0);r['modes'][mode]={};emit('DT_begin');tick=time.perf_counter()
                    producer.attribute_episodes([[row['row'] for row in rows]],[0.0]);torch.cuda.synchronize()
                    r['modes'][mode]['seconds']=time.perf_counter()-tick;save('DT_complete');emit('DT_complete')
                assert not norm_native and not snapshots
                save('complete');return dict(rank=self.rank,completed=True,optimizer_steps=0)
            except BaseException:
                import traceback
                r['traceback']=traceback.format_exc();save('failed');raise
            finally:
                reward_readout.trace_token_attribution=trace_owner;qwen35_answer_finite.FiniteAnswerOps.__call__=answer_owner
                for h in hooks:h.remove()
                norm_native.clear();snapshots.clear();head_native.clear()
                if producer is not None:
                    if g is not None:g['decoder_finite_pullback']=decoder_owner;g['_token_effect']=effect_owner
                    runner.boundaries.mlp=mlp_owner;runner.boundaries.norm_residual=norm_owner;runner.finite_fa=fa_owner
                    producer.runner.model.release_owner_params()
                if self._is_offload_param:offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log.close()
    return OperatorWorker


if __name__=='__main__':
    entry.make_worker=make_worker
    entry.main()
