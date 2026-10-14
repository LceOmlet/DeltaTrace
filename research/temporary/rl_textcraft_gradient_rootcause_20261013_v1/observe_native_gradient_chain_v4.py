"""Locate observed target derivative amplification through native Qwen boundaries.

Two already saved positive-row carriers, at the same original checkpoint5.
One native target PG pass per original global64. Passive tensor hooks record
adjoint norms and forward RMS; they return no replacement tensor/gradient.
No optimizer/scheduler update, generation, DT, validation or GRPO call.
"""
from contextlib import ExitStack
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import psutil
import ray
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/'native-gradient-chain-v4'
RUN=ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
OLD=OUT/'historical-target-derivatives-v1'
SHARED=Path('/mnt/si0021787ci2/default/lzq/deltatrace_backups/textcraft-20261011/runtime')/ROOT.relative_to('/')
ACTOR_SHA='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
CASE_MANIFEST=None
if os.environ.get('DT_NATIVE_DIAGNOSTIC_MANIFEST'):
    CASE_MANIFEST=json.loads(Path(os.environ['DT_NATIVE_DIAGNOSTIC_MANIFEST']).read_bytes())
    DEST=Path(CASE_MANIFEST['destination'])
CHECKPOINT_STEP=5 if CASE_MANIFEST is None else CASE_MANIFEST.get('checkpoint',5)
state={'scope':__doc__,'started_unix':time.time(),'observations':[],
       'optimizer_updates':0,'scheduler_updates':0,'DT_calls':0,'rollout_calls':0,'validation_calls':0,'GRPO_calls':0}
if CASE_MANIFEST is not None:state['scope']=CASE_MANIFEST['scope']

def phase(name,**values):
    state.update(phase=name,unix=time.time(),**values)
    (DEST/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps({'phase':name,'unix':state['unix'],**values}),flush=True)

def digest(actor):
    h=hashlib.sha256()
    for n,p in actor.actor_module.named_parameters():
        if p.requires_grad:
            v=p.to_local() if hasattr(p,'to_local') else p
            h.update(n.encode());h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def per_position_sumsq(tensor):
    # Bound temporary diagnostic memory. These reductions never enter the
    # model graph and do not replace native gradient norms or any kernel.
    with torch.no_grad():
        shape=tensor.shape
        flat=tensor.detach().reshape(-1,shape[-1])
        values=[]
        for start in range(0,len(flat),256):
            values.append(flat[start:start+256].float().square().sum(-1).cpu())
        return torch.cat(values).double()

class BoundaryReadout:
    def __init__(self,actor):
        self.actor=actor;self.handles=[];self.records=[];self.micro=-1
        self.forward=actor._forward_micro_batch
        self.extended=bool(CASE_MANIFEST and CASE_MANIFEST.get('native_ratio_chain_readout',False))
        self.previous_gradient=None;self.gradient_increments=[];self.parameter_layout=[]
        self.branch_forward=[];self.branch_backward=[];self.branch_owners=[]
    def forward_micro(self,*args,**kwargs):
        self.micro+=1
        return self.forward(*args,**kwargs)
    def hook(self,name,module,args,output):
        if not isinstance(output,torch.Tensor) or not output.requires_grad:
            return
        value=args[0] if args and isinstance(args[0],torch.Tensor) else None
        entry={'boundary':name,'micro':self.micro,'output_shape':list(output.shape),
               'dtype':str(output.dtype),'input_requires_grad':value is not None and value.requires_grad}
        self.records.append(entry)
        for which,t in [('output',output),('input',value)]:
            if t is None or not t.requires_grad:
                continue
            rms=None
            if not (name.endswith('.lm_head') and which=='output'):
                squared=per_position_sumsq(t)
                rms=(squared/t.shape[-1]).sqrt()
                entry[which+'_forward_RMS_median_all']=float(rms.median())
                entry[which+'_forward_RMS_range_all']=[float(rms.min()),float(rms.max())]
            def backward(gradient,which=which,rms=rms,entry=entry):
                sq=per_position_sumsq(gradient)
                active=sq>0
                finite=torch.isfinite(sq)
                entry[which+'_adjoint_norm']=float(sq.sum().sqrt())
                entry[which+'_adjoint_nonfinite_positions']=int((~finite).sum())
                entry[which+'_adjoint_nonzero_positions']=int(active.sum())
                if rms is not None and active.any():
                    selected=rms[active]
                    entry[which+'_forward_RMS_at_nonzero_adjoint_quantiles']=torch.quantile(selected,torch.tensor([0.,.1,.5,.9,1.],dtype=torch.float64)).tolist()
                # Returning None preserves the exact native gradient object.
                return None
            self.handles.append(t.register_hook(backward))
    def after_original_backward(self):
        # Observe the existing native accumulation only after its backward
        # returns. No gradient is assigned, scaled, clipped or replaced here.
        values=[];layout=[];offset=0
        for name,p in self.actor.actor_module.named_parameters():
            if not p.requires_grad:continue
            value=p.grad
            local=(p.to_local() if hasattr(p,'to_local') else p) if value is None else (value.to_local() if hasattr(value,'to_local') else value)
            copied=torch.zeros_like(local,device='cpu') if value is None else local.detach().to(device='cpu',copy=True)
            flat=copied.flatten();values.append(flat)
            layout.append(dict(name=name,start=offset,end=offset+flat.numel(),
                placements=[str(v) for v in value.placements] if value is not None and hasattr(value,'placements') else None))
            offset+=flat.numel()
        current=torch.cat(values)
        self.gradient_increments.append(current if self.previous_gradient is None else current-self.previous_gradient)
        self.previous_gradient=current;self.parameter_layout=layout
    def accumulation_report(self):
        if not self.extended:return None
        increments=torch.stack(self.gradient_increments)
        gram=torch.zeros((len(increments),len(increments)),dtype=torch.float64)
        for start in range(0,increments.shape[1],65536):
            block=increments[:,start:start+65536].double()
            gram+=block@block.T
        parameters=[]
        for p in self.parameter_layout:
            value=self.previous_gradient[p['start']:p['end']].double()
            parameters.append(dict(name=p['name'],placements=p['placements'],local_preclip_sumsq=float(value.square().sum())))
        return dict(original_backward_calls=len(increments),local_increment_Gram=gram.tolist(),
            local_final_preclip_sumsq=float(self.previous_gradient.double().square().sum()),
            parameters=parameters,
            scope='Differences of actual native accumulated FP32 gradient shards after each existing B4 backward, before clipping. Subtraction has FP32 rounding; this is not a replacement gradient reducer or independent-token variance estimate.')
    def install(self,stack):
        import observe_native_optimizer_minibatch as observer
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer,Qwen3_5RMSNorm
        found=[]
        for name,module in self.actor.actor_module.named_modules():
            if (isinstance(module,Qwen3_5DecoderLayer) or (isinstance(module,Qwen3_5RMSNorm) and name.endswith('.model.norm'))
                    or (self.extended and name.endswith('.lm_head'))):
                found.append(name)
                self.handles.append(module.register_forward_hook(
                    lambda module,args,output,name=name:self.hook(name,module,args,output)))
        assert len(found)==(34 if self.extended else 33),(len(found),found)
        branch_layers=[] if CASE_MANIFEST is None else CASE_MANIFEST.get('decoder_branch_layers',[])
        if branch_layers:
            # Native module backward hooks expose branch grad_input. A hook
            # on a shared residual-input Tensor would instead see the total
            # accumulated adjoint and cannot identify that branch alone.
            suffixes=[f'.layers.{layer}.{name}' for layer in branch_layers for name in
                ('input_layernorm','self_attn','post_attention_layernorm','mlp','self_attn.q_norm','self_attn.k_norm')]
            selected=[];latest={}
            for name,module in self.actor.actor_module.named_modules():
                if not any(name.endswith(suffix) for suffix in suffixes):continue
                selected.append(name)
                source=Path(inspect.getsourcefile(type(module)))
                self.branch_owners.append(dict(name=name,type=type(module).__name__,source=str(source),
                    SHA256=hashlib.sha256(source.read_bytes()).hexdigest()))
                def forward(module,args,output,name=name):
                    value=args[0] if args and isinstance(args[0],torch.Tensor) else None
                    rms=None if value is None else (per_position_sumsq(value)/value.shape[-1]).sqrt()
                    record=dict(name=name,micro=self.micro,input_shape=None if value is None else list(value.shape),
                        input_RMS_quantiles_all=None if rms is None else torch.quantile(rms,torch.tensor([0.,.1,.5,.9,1.],dtype=torch.float64)).tolist())
                    self.branch_forward.append(record);latest[name]=(self.micro,rms)
                def backward(module,grad_input,grad_output,name=name):
                    record=dict(name=name,micro=self.micro,native_module_grad_input=[],native_module_grad_output=[])
                    for which,values in [('input',grad_input),('output',grad_output)]:
                        for index,g in enumerate(values):
                            if not isinstance(g,torch.Tensor):continue
                            sq=per_position_sumsq(g)
                            row=dict(index=index,shape=list(g.shape),dtype=str(g.dtype),norm=float(sq.sum().sqrt()),
                                nonfinite_positions=int((~torch.isfinite(sq)).sum()),nonzero_positions=int((sq>0).sum()))
                            if which=='input' and name in latest:
                                micro,rms=latest[name]
                                if rms is not None and micro==self.micro and rms.shape==sq.shape and bool((sq>0).any()):
                                    row['forward_RMS_at_nonzero_branch_adjoint_quantiles']=torch.quantile(rms[sq>0],torch.tensor([0.,.1,.5,.9,1.],dtype=torch.float64)).tolist()
                            record['native_module_grad_'+which].append(row)
                    self.branch_backward.append(record)
                    return None
                self.handles.append(module.register_forward_hook(forward))
                self.handles.append(module.register_full_backward_hook(backward))
            assert len(selected)==len(suffixes),(selected,suffixes)
        stack.enter_context(observer._temporary_attribute(self.actor,'_forward_micro_batch',self.forward_micro))
        if self.extended:
            original_backward=torch.Tensor.backward
            def observed_backward(tensor,*args,**kwargs):
                result=original_backward(tensor,*args,**kwargs)
                self.after_original_backward()
                return result
            stack.enter_context(observer._temporary_attribute(torch.Tensor,'backward',observed_backward))
        stack.callback(lambda:[h.remove() for h in self.handles])
        return found

@ray.remote
class ObservedWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_case(self,iteration):
        self.case_iteration=iteration
        path=Path(inspect.getsourcefile(type(self.actor)))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==ACTOR_SHA
        if CASE_MANIFEST is not None and 'expected_trainable_SHA256_by_rank' in CASE_MANIFEST:
            assert digest(self.actor)==CASE_MANIFEST['expected_trainable_SHA256_by_rank'][str(self.rank)]
        return {'rank':self.rank,'pid':os.getpid(),'birth':psutil.Process().create_time(),
                'scheduler':self.actor_lr_scheduler.last_epoch,'actor_path':str(path),'actor_SHA256':ACTOR_SHA}
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def observe_chain(self,data):
        import observe_native_optimizer_minibatch as observer
        observer.ACTOR_SHA256=ACTOR_SHA
        before=digest(self.actor);scheduler=self.actor_lr_scheduler.last_epoch
        path=DEST/f'iteration{self.case_iteration}-rank{self.rank}-PG.json'
        readout=BoundaryReadout(self.actor)
        def observe(data):
            views=({'native_target':data.batch['diagnostic_target_legacy']} if CASE_MANIFEST is None or
                   'diagnostic_advantage_views' not in CASE_MANIFEST else
                   {name:data.batch[key] for name,key in CASE_MANIFEST['diagnostic_advantage_views'].items()})
            return observer.observe_native_optimizer_minibatch(self.actor,data,output_path=path,rank=self.rank,
                policy_advantages=views)
        with ExitStack() as stack:
            enabled=CASE_MANIFEST is None or CASE_MANIFEST.get('passive_boundary_readout_enabled',False)
            boundaries=readout.install(stack) if enabled else []
            stack.enter_context(observer._temporary_attribute(self.actor,'update_policy',observe))
            stack.enter_context(observer._temporary_attribute(self.actor_lr_scheduler,'step',lambda *a,**k:None))
            output=super().update_actor(data)
        after=digest(self.actor)
        assert before==after and self.actor_lr_scheduler.last_epoch==scheduler
        report={'rank':self.rank,'iteration':self.case_iteration,'checkpoint':CHECKPOINT_STEP,
                'source':state['scope'],'boundaries':boundaries,'records':readout.records,
                'passive_boundary_readout_enabled':enabled,
                'native_accumulation_readout':readout.accumulation_report(),
                'native_decoder_branch_forward':readout.branch_forward,
                'native_decoder_branch_backward':readout.branch_backward,
                'native_decoder_branch_owners':readout.branch_owners,
                'trainable_SHA256_before':before,'trainable_SHA256_after':after,
                'scheduler_before':scheduler,'scheduler_after':self.actor_lr_scheduler.last_epoch,
                'optimizer_steps':0,'scheduler_steps':0,
                'norm_scope':'Each rank actual local B4 hidden adjoints; no inferred FSDP sum or replacement reducer.',
                'forward_RMS_scope':'RMS at positions with a nonzero native adjoint. Recomputed gradient-checkpoint boundaries are retained as actual events; no full activation tensors persist.'}
        (DEST/f'iteration{self.case_iteration}-rank{self.rank}-chain.json').write_text(json.dumps(report,indent=2)+'\n')
        return output

def main():
    assert not (DEST/'result.json').exists(),'Do not repeat a running/completed diagnostic'
    launch=json.loads((RUN/'launch.json').read_bytes())
    cfg=OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    for k,v in launch['options'].items():OmegaConf.update(cfg,k.lstrip('+'),v,force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps=330
    env={k:v for k,v in launch['effective_environment'].items() if k not in ('CUDA_VISIBLE_DEVICES','RAY_ADDRESS','VERL_GLOBAL_TOKEN_MEAN')}
    env['PYTHONPATH']=str(OUT)+os.pathsep+env.get('PYTHONPATH','')
    env['DT_FSDP_EVENT_WAIT_RECEIPTS']=str(DEST/'original-eventwait')
    checkpoint=(SHARED/f'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/textcraft-dt/checkpoints/global_step_{CHECKPOINT_STEP}/actor'
                if CASE_MANIFEST is None or 'checkpoint_path' not in CASE_MANIFEST else Path(CASE_MANIFEST['checkpoint_path']))
    phase('initializing_original_actor_only',checkpoint=str(checkpoint))
    if CASE_MANIFEST is not None:
        env['DT_NATIVE_DIAGNOSTIC_MANIFEST']=os.environ['DT_NATIVE_DIAGNOSTIC_MANIFEST']
    ray.init(address='local',num_cpus=8,num_gpus=2,include_dashboard=False,
        _temp_dir='/tmp/ray-tc-chain-v4' if CASE_MANIFEST is None else '/tmp/ray-tc-matched-v11',runtime_env={'env_vars':env})
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
            RayClassWithInitArgs(ObservedWorker,cfg.actor_rollout_ref,'actor'))
        group.init_model()
        cases=([{'name':i,'path':str(OLD/f'iteration{i}-target-derivative-carrier.pkl')} for i in (6,81)]
               if CASE_MANIFEST is None else CASE_MANIFEST['cases'])
        for case in cases:
            iteration=case['name']
            full=DataProto.load_from_disk(case['path'])
            if CASE_MANIFEST is None:
                idx=torch.cat((torch.arange(32),torch.arange(128,160)))
                batch=full.select_idxs(idx)
            else:
                assert len(full)==64 and hashlib.sha256(Path(case['path']).read_bytes()).hexdigest()==case['SHA256']
                batch=full
            batch.meta_info['global_token_num']=batch.batch['attention_mask'].sum(-1).tolist()
            batch.meta_info.pop('global_batch_info',None)
            phase('loading_same_checkpoint',iteration=iteration)
            group.load_checkpoint(local_path=str(checkpoint),del_local_after_load=False)
            bindings=group.set_case(iteration)
            assert all(b['scheduler']==CHECKPOINT_STEP for b in bindings)
            phase('native_old_policy_point',iteration=iteration,bindings=bindings)
            probabilities=group.compute_log_prob(batch)
            batch.batch['old_log_probs']=probabilities.batch['old_log_probs']
            batch.batch['ref_log_prob']=batch.batch['old_log_probs'].clone()
            probability_artifact={}
            if CASE_MANIFEST is not None and (CASE_MANIFEST.get('score_readout_only',False) or
                    CASE_MANIFEST.get('save_native_probability_carrier',False)):
                # Preserve the actual native owner output. Saving this data
                # adds no model, softmax, backward or policy update call.
                path=DEST/f'{iteration}-native-logp.pkl'
                assert not path.exists()
                batch.save_to_disk(str(path))
                after_bindings=group.set_case(iteration)
                assert bindings==after_bindings
                probability_artifact=dict(
                    native_probability_carrier=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    before_bindings=bindings,after_bindings=after_bindings)
            if CASE_MANIFEST is not None and CASE_MANIFEST.get('score_readout_only',False):
                state['observations'].append(dict(iteration=iteration,checkpoint=CHECKPOINT_STEP,
                    **probability_artifact,parameter_backward=0))
                phase('native_target_probability_readout_complete',iteration=iteration)
                continue
            phase('native_backward_with_passive_boundary_readout',iteration=iteration)
            group.observe_chain(batch)
            records=[json.loads((DEST/f'iteration{iteration}-rank{rank}-PG.json').read_bytes()) for rank in (0,1)]
            state['observations'].append({'iteration':iteration,'checkpoint':CHECKPOINT_STEP,
                'gradient_statistics':records[0]['gradient_statistics'],**probability_artifact})
            phase('case_complete',iteration=iteration)
        phase('complete_native_target_score_readout_not_repair' if CASE_MANIFEST is not None and
              CASE_MANIFEST.get('score_readout_only',False) else 'complete_native_gradient_chain_localization_not_repair')
    except BaseException as exc:
        phase('failed',exception=repr(exc));raise
    finally:ray.shutdown()

if __name__=='__main__':main()
