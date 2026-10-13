"""Locate temporal gradient amplification using a fixed real DT minibatch.

The same original iteration81 IDs/mask/whitened coefficients are held fixed at
native checkpoints5/40/80. At each checkpoint, native HF recomputes the old
policy point, then the existing native PG gradient observer executes one
complete global64/local32/B4 backward with optimizer and scheduler no-ops.
This isolates checkpoint sensitivity; it is not on-policy historical training,
a validation-score experiment, a GRPO comparison, or repair acceptance.
"""
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
from verl.single_controller.base.decorator import Dispatch,register
from verl.single_controller.ray import RayClassWithInitArgs,RayResourcePool,RayWorkerGroup
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.workers.fsdp_workers import ActorRolloutRefWorker

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/'fixed-batch-historical-checkpoints-v1'
RUN=ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
SHARED=Path('/mnt/si0021787ci2/default/lzq/deltatrace_backups/textcraft-20261011/runtime')/ROOT.relative_to('/')
ACTOR_SHA='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
report={'scope':__doc__,'started_unix':time.time(),'observations':[],
        'optimizer_updates':0,'scheduler_updates':0,'GRPO_calls':0,'validation_calls':0,'DT_calls':0}
def phase(name,**values):
    report.update(phase=name,unix=time.time(),**values)
    DEST.mkdir(exist_ok=True)
    (DEST/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'phase':name,'unix':report['unix'],**values}),flush=True)
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
    def set_observation_step(self,step):
        self.observation_step=step
        return {'rank':self.rank,'pid':os.getpid(),'birth':psutil.Process().create_time(),
                'step':step,'scheduler':self.actor_lr_scheduler.last_epoch}
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def observe_fixed_gradient(self,data):
        import observe_native_optimizer_minibatch as observer
        from contextlib import ExitStack
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        actor_path=Path(inspect.getsourcefile(DataParallelPPOActor))
        assert hashlib.sha256(actor_path.read_bytes()).hexdigest()==ACTOR_SHA
        observer.ACTOR_SHA256=ACTOR_SHA
        step=self.observation_step
        path=DEST/f'checkpoint{step}-rank{self.rank}-native.json'
        parameter_stats={}
        original_capture=observer.capture_native_gradient
        names={id(p):n for n,p in self.actor.actor_module.named_parameters() if p.requires_grad}
        before=digest_trainable(self.actor)
        def capture(p,groups):
            value,record,bucket=original_capture(p,groups)
            name=names[id(p)]
            w=p.to_local() if hasattr(p,'to_local') else p
            parameter_stats[name]={'bucket':list(bucket),'shape':record.get('local_grad_shape'),
                 'gradient_sumsq':None if value is None else value.double().square().sum().item(),
                 'weight_sumsq':w.detach().cpu().double().square().sum().item()}
            return value,record,bucket
        def observed_update_policy(data):
            return observer.observe_native_optimizer_minibatch(self.actor,data,output_path=path,rank=self.rank,
                       policy_advantages={'fixed_DT_PG':data.batch['advantages']})
        with ExitStack() as stack:
            stack.enter_context(observer._temporary_attribute(observer,'capture_native_gradient',capture))
            stack.enter_context(observer._temporary_attribute(self.actor,'update_policy',observed_update_policy))
            stack.enter_context(observer._temporary_attribute(self.actor_lr_scheduler,'step',lambda *a,**k:None))
            output=super().update_actor(data)
        after=digest_trainable(self.actor)
        assert before==after
        (DEST/f'checkpoint{step}-rank{self.rank}-parameter-statistics.json').write_text(json.dumps({
          'step':step,'rank':self.rank,'parameters':parameter_stats,'trainable_SHA256_before':before,
          'trainable_SHA256_after':after,'actor_path':str(actor_path),'actor_SHA256':ACTOR_SHA,
          'scheduler':self.actor_lr_scheduler.last_epoch},indent=2)+'\n')
        return output

def main():
    receipts=[json.loads((OUT/f'dt/rank{r}-capture.json').read_bytes()) for r in (0,1)]
    batches=[DataProto.load_from_disk(r['input']) for r in receipts]
    full=DataProto.concat(batches)
    check=full.select(deepcopy=True);check.batch=full.batch.clone(recurse=False)
    check=compute_advantage(check,adv_estimator='deltatrace',multi_turn=True)
    assert torch.equal(check.batch['advantages'],full.batch['advantages'])
    indices=torch.cat([torch.arange(32),torch.arange(128,160)])
    fixed=full.select_idxs(indices)
    fixed.meta_info['global_token_num']=fixed.batch['attention_mask'].sum(-1).tolist()
    fixed.meta_info.pop('global_batch_info',None)
    fixed.meta_info['multi_turn']=True
    checkpoints={s:(ROOT if s==80 else SHARED)/Path('runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/textcraft-dt/checkpoints')/f'global_step_{s}/actor' for s in (5,40,80)}
    for cp in checkpoints.values():
        for rank in (0,1):
            for prefix in ('model','optim','extra_state'):
                assert (cp/f'{prefix}_world_size_2_rank_{rank}.pt').is_file()
    launch=json.loads((RUN/'launch.json').read_bytes())
    cfg=OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    for k,v in launch['options'].items():OmegaConf.update(cfg,k.lstrip('+'),v,force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps=330
    env={k:v for k,v in launch['effective_environment'].items() if k not in ('CUDA_VISIBLE_DEVICES','RAY_ADDRESS')}
    env['PYTHONPATH']=str(OUT)+os.pathsep+env.get('PYTHONPATH','')
    env['DT_FSDP_EVENT_WAIT_RECEIPTS']=str(DEST/'eventwait')
    env.pop('VERL_GLOBAL_TOKEN_MEAN',None)
    phase('initializing_original_actor_only',checkpoints={s:str(p) for s,p in checkpoints.items()},
          saved_input_files=[r['input'] for r in receipts],original_global_rows=256,fixed_global_rows=64,
          coefficients_SHA256=hashlib.sha256(fixed.batch['advantages'].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
    ray.init(address='local',num_cpus=8,num_gpus=2,include_dashboard=False,
             _temp_dir='/tmp/ray-tc-fixed-history-v1',runtime_env={'env_vars':env})
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
                            RayClassWithInitArgs(ObservedWorker,cfg.actor_rollout_ref,'actor'))
        group.init_model()
        for step,cp in checkpoints.items():
            phase('loading_native_historical_checkpoint',current_checkpoint=step)
            group.load_checkpoint(local_path=str(cp),del_local_after_load=False)
            bindings=group.set_observation_step(step)
            assert all(b['scheduler']==step for b in bindings)
            phase('native_HF_old_policy_point',current_checkpoint=step,worker_bindings=bindings)
            probabilities=group.compute_log_prob(fixed)
            fixed.batch['old_log_probs']=probabilities.batch['old_log_probs']
            phase('fixed_DT_PG_original_backward',current_checkpoint=step)
            group.observe_fixed_gradient(fixed)
            results=[json.loads((DEST/f'checkpoint{step}-rank{r}-native.json').read_bytes()) for r in (0,1)]
            report['observations'].append({'checkpoint':step,'gradient_statistics_by_rank':[r['gradient_statistics'] for r in results],
                  'native_clip_return_by_rank':[r['passes']['fixed_DT_PG']['optimizer_boundary']['native_clip_return'] for r in results],
                  'microbatch_PPO_diagnostics':[r['passes']['fixed_DT_PG']['microbatch_losses'] for r in results]})
            phase('checkpoint_observation_complete',current_checkpoint=step)
        phase('completed_fixed_batch_gradient_observations')
    except BaseException as exc:
        phase('failed',exception=repr(exc));raise
    finally:ray.shutdown()
if __name__=='__main__':main()
