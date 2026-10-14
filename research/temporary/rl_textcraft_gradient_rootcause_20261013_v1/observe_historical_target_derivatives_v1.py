"""Measure saved target derivatives at original checkpoints5/40/80.

Uses exact historical positive IDs/masks and original B4 placement through
explicit diagnostic derivative carriers. Failure inputs are zero-derivative
placeholders, not a fabricated historical full batch. Native actor owns HF,
PPO, B4 accumulation, FSDP and backward. No optimizer/scheduler update, DT,
generation, validation or GRPO is called. Four minibatches are measured at
each entry checkpoint with frozen weights, not sequential training updates.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from contextlib import ExitStack

import psutil
import ray
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch,register
from verl.single_controller.ray import RayClassWithInitArgs,RayResourcePool,RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/'historical-target-derivatives-v1'
RUN=ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
SHARED=Path('/mnt/si0021787ci2/default/lzq/deltatrace_backups/textcraft-20261011/runtime')/ROOT.relative_to('/')
ACTOR_SHA='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
state={'scope':__doc__,'started_unix':time.time(),'observations':[],
    'optimizer_updates':0,'scheduler_updates':0,'rollout_calls':0,'DT_calls':0,'GRPO_calls':0,'validation_calls':0}

def phase(name,**values):
    state.update(phase=name,unix=time.time(),**values)
    (DEST/'result.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps({'phase':name,'unix':state['unix'],**values}),flush=True)

def digest_trainable(actor):
    h=hashlib.sha256()
    for name,p in actor.actor_module.named_parameters():
        if p.requires_grad:
            x=p.to_local() if hasattr(p,'to_local') else p
            h.update(name.encode());h.update(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

@ray.remote
class ObservedWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_observation(self,iteration,mini):
        self.observation_iteration=iteration;self.observation_mini=mini
        return {'rank':self.rank,'pid':os.getpid(),'birth':psutil.Process().create_time(),
            'iteration':iteration,'mini':mini,'scheduler':self.actor_lr_scheduler.last_epoch,
            'actor_file':inspect.getsourcefile(type(self.actor)),
            'actor_SHA256':hashlib.sha256(Path(inspect.getsourcefile(type(self.actor))).read_bytes()).hexdigest()}

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def observe_target_derivatives(self,data):
        import observe_native_optimizer_minibatch as observer
        observer.ACTOR_SHA256=ACTOR_SHA
        iteration,mini=self.observation_iteration,self.observation_mini
        path=DEST/f'iteration{iteration}-mini{mini}-rank{self.rank}.json'
        before=digest_trainable(self.actor)
        scheduler=self.actor_lr_scheduler.last_epoch
        original_capture=observer.capture_native_gradient
        def capture(p,groups):
            value,record,bucket=original_capture(p,groups)
            record['local_gradient_sumsq']=None if value is None else float(value.double().square().sum())
            return value,record,bucket
        def observed_update_policy(data):
            return observer.observe_native_optimizer_minibatch(self.actor,data,output_path=path,rank=self.rank,
                policy_advantages={'historical_target_B4':data.batch['diagnostic_target_legacy'],
                    'historical_target_global64':data.batch['diagnostic_target_common']})
        with ExitStack() as stack:
            stack.enter_context(observer._temporary_attribute(observer,'capture_native_gradient',capture))
            stack.enter_context(observer._temporary_attribute(self.actor,'update_policy',observed_update_policy))
            stack.enter_context(observer._temporary_attribute(self.actor_lr_scheduler,'step',lambda *a,**k:None))
            output=super().update_actor(data)
        after=digest_trainable(self.actor)
        assert before==after and self.actor_lr_scheduler.last_epoch==scheduler
        (DEST/f'iteration{iteration}-mini{mini}-rank{self.rank}-state.json').write_text(json.dumps({
            'rank':self.rank,'trainable_SHA256_before':before,'trainable_SHA256_after':after,
            'scheduler_before':scheduler,'scheduler_after':self.actor_lr_scheduler.last_epoch,
            'optimizer_steps':0,'scheduler_steps':0},indent=2)+'\n')
        return output

def output_score(data):
    mask=data.batch['diagnostic_target_mask'].bool()
    lp=data.batch['old_log_probs'].double()
    policy=data.batch['response_mask'].bool()
    groups={}
    for label,key in [('B4','diagnostic_target_legacy'),('global64','diagnostic_target_common')]:
        a=data.batch[key].double()
        c=torch.zeros_like(a)
        for start in range(0,64,4):
            c[start:start+4]=a[start:start+4]/(16*int(policy[start:start+4].sum()))
        v=(c*(-lp.expm1()))[mask]
        lower=float(v.square().sum().sqrt())
        groups[label]={'output_logit_L2_bounds':[lower,2**.5*lower],
            'output_logit_L1':2*float(v.abs().sum()),
            'target_coefficient_L2':float(c[mask].square().sum().sqrt())}
    return {'target_tokens':int(mask.sum()),'native_target_NLL_sum':float(-lp[mask].sum()),
        'native_target_logp_gt_zero':int((lp[mask]>0).sum()),'groups':groups}

def main():
    assert not (DEST/'result.json').exists(),'Do not duplicate this bounded diagnostic'
    receipt=json.loads((DEST/'carrier-receipt.json').read_bytes())
    checkpoints={s:(ROOT if s==80 else SHARED)/Path('runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/textcraft-dt/checkpoints')/f'global_step_{s}/actor' for s in (5,40,80)}
    for cp in checkpoints.values():
        for rank in (0,1):
            for prefix in ('model','optim','extra_state'):
                assert (cp/f'{prefix}_world_size_2_rank_{rank}.pt').is_file()
    launch=json.loads((RUN/'launch.json').read_bytes())
    cfg=OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    for k,v in launch['options'].items():OmegaConf.update(cfg,k.lstrip('+'),v,force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps=330
    env={k:v for k,v in launch['effective_environment'].items() if k not in ('CUDA_VISIBLE_DEVICES','RAY_ADDRESS','VERL_GLOBAL_TOKEN_MEAN')}
    env['PYTHONPATH']=str(OUT)+os.pathsep+env.get('PYTHONPATH','')
    env['DT_FSDP_EVENT_WAIT_RECEIPTS']=str(DEST/'original-eventwait')
    phase('initializing_original_actor_only',carriers=receipt['carriers'],checkpoints={s:str(p) for s,p in checkpoints.items()},
        neutral_unselected_KL='ref_log_prob is set to freshly recomputed old_log_prob on these derivative carriers; KL backward is unselected. This prevents unrelated mismatched placeholder references and is not a full-objective/training comparison.')
    ray.init(address='local',num_cpus=8,num_gpus=2,include_dashboard=False,
        _temp_dir='/tmp/ray-tc-historical-target-v1',runtime_env={'env_vars':env})
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
            RayClassWithInitArgs(ObservedWorker,cfg.actor_rollout_ref,'actor'))
        group.init_model()
        for carrier in receipt['carriers']:
            iteration=carrier['iteration'];checkpoint=carrier['checkpoint']
            full=DataProto.load_from_disk(carrier['path'])
            assert hashlib.sha256(Path(carrier['path']).read_bytes()).hexdigest()==carrier['SHA256']
            phase('loading_original_historical_checkpoint',iteration=iteration,checkpoint=checkpoint)
            group.load_checkpoint(local_path=str(checkpoints[checkpoint]),del_local_after_load=False)
            bindings=group.set_observation(iteration,-1)
            assert all(b['scheduler']==checkpoint and b['actor_SHA256']==ACTOR_SHA for b in bindings)
            full.meta_info['global_token_num']=full.batch['attention_mask'].sum(-1).tolist()
            phase('native_HF_old_policy_point',iteration=iteration,checkpoint=checkpoint,worker_bindings=bindings)
            probabilities=group.compute_log_prob(full)
            full.batch['old_log_probs']=probabilities.batch['old_log_probs']
            full.batch['ref_log_prob']=full.batch['old_log_probs'].clone()
            for mini in range(4):
                indices=torch.cat((torch.arange(mini*32,mini*32+32),torch.arange(128+mini*32,128+mini*32+32)))
                batch=full.select_idxs(indices)
                batch.meta_info['global_token_num']=batch.batch['attention_mask'].sum(-1).tolist()
                batch.meta_info.pop('global_batch_info',None)
                group.set_observation(iteration,mini)
                phase('native_two_target_backward',iteration=iteration,checkpoint=checkpoint,mini=mini)
                group.observe_target_derivatives(batch)
                records=[json.loads((DEST/f'iteration{iteration}-mini{mini}-rank{r}.json').read_bytes()) for r in (0,1)]
                assert records[0]['gradient_statistics']==records[1]['gradient_statistics']
                state['observations'].append({'iteration':iteration,'checkpoint':checkpoint,'mini':mini,
                    'native_gradient_statistics':records[0]['gradient_statistics'],
                    'native_clip_returns_by_rank':[{k:v['optimizer_boundary']['native_clip_return'] for k,v in r['passes'].items()} for r in records],
                    'output_score':output_score(batch),
                    'native_forward_diagnostics_by_rank':[{k:v['microbatch_losses'] for k,v in r['passes'].items()} for r in records]})
                phase('native_minibatch_derivative_complete',iteration=iteration,checkpoint=checkpoint,mini=mini)
        phase('complete_historical_target_derivatives_not_repair')
    except BaseException as exc:
        phase('failed',exception=repr(exc));raise
    finally:ray.shutdown()

if __name__=='__main__':main()
