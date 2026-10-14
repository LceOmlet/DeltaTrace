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
state={'scope':__doc__,'started_unix':time.time(),'observations':[],
       'optimizer_updates':0,'scheduler_updates':0,'DT_calls':0,'rollout_calls':0,'validation_calls':0,'GRPO_calls':0}

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
                if active.any():
                    selected=rms[active]
                    entry[which+'_forward_RMS_at_nonzero_adjoint_quantiles']=torch.quantile(selected,torch.tensor([0.,.1,.5,.9,1.],dtype=torch.float64)).tolist()
                # Returning None preserves the exact native gradient object.
                return None
            self.handles.append(t.register_hook(backward))
    def install(self,stack):
        import observe_native_optimizer_minibatch as observer
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer,Qwen3_5RMSNorm
        found=[]
        for name,module in self.actor.actor_module.named_modules():
            if isinstance(module,Qwen3_5DecoderLayer) or (isinstance(module,Qwen3_5RMSNorm) and name.endswith('.model.norm')):
                found.append(name)
                self.handles.append(module.register_forward_hook(
                    lambda module,args,output,name=name:self.hook(name,module,args,output)))
        assert len(found)==33,(len(found),found)
        stack.enter_context(observer._temporary_attribute(self.actor,'_forward_micro_batch',self.forward_micro))
        stack.callback(lambda:[h.remove() for h in self.handles])
        return found

@ray.remote
class ObservedWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_case(self,iteration):
        self.case_iteration=iteration
        path=Path(inspect.getsourcefile(type(self.actor)))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==ACTOR_SHA
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
            return observer.observe_native_optimizer_minibatch(self.actor,data,output_path=path,rank=self.rank,
                policy_advantages={'native_target':data.batch['diagnostic_target_legacy']})
        with ExitStack() as stack:
            boundaries=readout.install(stack)
            stack.enter_context(observer._temporary_attribute(self.actor,'update_policy',observe))
            stack.enter_context(observer._temporary_attribute(self.actor_lr_scheduler,'step',lambda *a,**k:None))
            output=super().update_actor(data)
        after=digest(self.actor)
        assert before==after and self.actor_lr_scheduler.last_epoch==scheduler
        report={'rank':self.rank,'iteration':self.case_iteration,'checkpoint':5,
                'source':__doc__,'boundaries':boundaries,'records':readout.records,
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
    checkpoint=SHARED/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/textcraft-dt/checkpoints/global_step_5/actor'
    phase('initializing_original_actor_only',checkpoint=str(checkpoint))
    ray.init(address='local',num_cpus=8,num_gpus=2,include_dashboard=False,
        _temp_dir='/tmp/ray-tc-chain-v4',runtime_env={'env_vars':env})
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
            RayClassWithInitArgs(ObservedWorker,cfg.actor_rollout_ref,'actor'))
        group.init_model()
        for iteration in (6,81):
            full=DataProto.load_from_disk(str(OLD/f'iteration{iteration}-target-derivative-carrier.pkl'))
            idx=torch.cat((torch.arange(32),torch.arange(128,160)))
            batch=full.select_idxs(idx)
            batch.meta_info['global_token_num']=batch.batch['attention_mask'].sum(-1).tolist()
            batch.meta_info.pop('global_batch_info',None)
            phase('loading_same_checkpoint',iteration=iteration)
            group.load_checkpoint(local_path=str(checkpoint),del_local_after_load=False)
            bindings=group.set_case(iteration)
            assert all(b['scheduler']==5 for b in bindings)
            phase('native_old_policy_point',iteration=iteration,bindings=bindings)
            probabilities=group.compute_log_prob(batch)
            batch.batch['old_log_probs']=probabilities.batch['old_log_probs']
            batch.batch['ref_log_prob']=batch.batch['old_log_probs'].clone()
            phase('native_backward_with_passive_boundary_readout',iteration=iteration)
            group.observe_chain(batch)
            records=[json.loads((DEST/f'iteration{iteration}-rank{rank}-PG.json').read_bytes()) for rank in (0,1)]
            state['observations'].append({'iteration':iteration,'checkpoint':5,
                'gradient_statistics':records[0]['gradient_statistics']})
            phase('case_complete',iteration=iteration)
        phase('complete_native_gradient_chain_localization_not_repair')
    except BaseException as exc:
        phase('failed',exception=repr(exc));raise
    finally:ray.shutdown()

if __name__=='__main__':main()
