"""Locate the large saved PG using disjoint token groups through the native owner.

Three backwards on one original global64 minibatch, no optimizer/scheduler
updates. Group values retain original selected advantages and the entire
original loss mask/denominator. Their gradients partition the original PG.
This is gradient localization, never a single-update quality or repair gate.
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
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/'native-PG-groups-v1'
RUN=ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1'
ACTOR_SHA='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
state={'scope':__doc__,'started_unix':time.time(),'optimizer_updates':0,
    'scheduler_updates':0,'rollout_calls':0,'DT_calls':0,'GRPO_calls':0,'validation_calls':0}

def phase(name,**values):
    state.update(phase=name,unix=time.time(),**values)
    (DEST/'result.json').write_text(json.dumps(state,indent=2,default=str)+'\n')
    print(json.dumps({'phase':name,'unix':state['unix'],**values},default=str),flush=True)

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
    def bind_saved_weights_rng(self):
        source=json.loads((OUT/f'dt/rank{self.rank}-capture.json').read_bytes())
        saved=torch.load(source['lora_local'],map_location='cpu',weights_only=False)
        current={n:p for n,p in self.actor_module_fsdp.named_parameters() if p.requires_grad}
        assert set(saved)==set(current)
        for n,p in current.items():
            v=p.to_local() if hasattr(p,'to_local') else p
            assert torch.equal(v.detach().cpu(),saved[n]),n
        self.checkpoint_manager.load_rng_state(torch.load(source['rng'],map_location='cpu',weights_only=False))
        return {'rank':self.rank,'pid':os.getpid(),'birth':psutil.Process().create_time(),
            'saved_trainable_tensors_exact':len(saved),'scheduler':self.actor_lr_scheduler.last_epoch,
            'actor_file':inspect.getsourcefile(type(self.actor)),
            'actor_SHA256':hashlib.sha256(Path(inspect.getsourcefile(type(self.actor))).read_bytes()).hexdigest()}

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def observe_PG_groups(self,data):
        import observe_native_optimizer_minibatch as observer
        from contextlib import ExitStack
        actor_path=Path(inspect.getsourcefile(type(self.actor)))
        assert hashlib.sha256(actor_path.read_bytes()).hexdigest()==ACTOR_SHA
        observer.ACTOR_SHA256=ACTOR_SHA
        before=digest_trainable(self.actor)
        scheduler_before=self.actor_lr_scheduler.last_epoch
        capture=observer.capture_native_gradient
        def capture_with_local_sumsq(parameter,groups):
            value,record,bucket=capture(parameter,groups)
            record['local_gradient_sumsq']=None if value is None else float(value.double().square().sum())
            return value,record,bucket
        def observed_update_policy(data):
            a=data.batch['advantages']
            nonzero=data.batch['diagnostic_nonzero_reward'].bool()
            target=data.batch['diagnostic_target'].bool()
            groups={
                'successful_target':a*target*nonzero[:,None],
                'successful_source':a*(~target)*nonzero[:,None],
                'zero_reward_all_policy':a*(~nonzero[:,None]),
            }
            assert torch.equal(sum(groups.values()),a)
            return observer.observe_native_optimizer_minibatch(self.actor,data,
                output_path=DEST/f'rank{self.rank}-groups.json',rank=self.rank,policy_advantages=groups)
        with ExitStack() as stack:
            stack.enter_context(observer._temporary_attribute(observer,'capture_native_gradient',capture_with_local_sumsq))
            stack.enter_context(observer._temporary_attribute(self.actor,'update_policy',observed_update_policy))
            stack.enter_context(observer._temporary_attribute(self.actor_lr_scheduler,'step',lambda *a,**k:None))
            result=super().update_actor(data)
        after=digest_trainable(self.actor)
        assert before==after and self.actor_lr_scheduler.last_epoch==scheduler_before
        (DEST/f'rank{self.rank}-state.json').write_text(json.dumps({
            'rank':self.rank,'trainable_SHA256_before':before,'trainable_SHA256_after':after,
            'scheduler_before':scheduler_before,'scheduler_after':self.actor_lr_scheduler.last_epoch,
            'optimizer_steps':0,'scheduler_steps':0},indent=2)+'\n')
        return result

def main():
    assert DEST.is_dir() and not (DEST/'result.json').exists()
    sources=[json.loads((OUT/f'dt/rank{r}-capture.json').read_bytes()) for r in (0,1)]
    full=DataProto.concat([DataProto.load_from_disk(s['input']) for s in sources])
    target=torch.zeros_like(full.batch['responses'],dtype=torch.bool)
    for i,art in enumerate(full.non_tensor_batch['dt_direct_target_artifact']):
        pos=torch.as_tensor(art['retained_response_positions']);keep=pos>=0
        target[i,keep]=torch.as_tensor(art['target_mask'],dtype=torch.bool)[pos[keep]]
    full.batch['diagnostic_target']=target
    full.batch['diagnostic_nonzero_reward']=full.batch['token_level_rewards'].sum(-1)!=0
    idx=torch.cat((torch.arange(32),torch.arange(128,160)))
    batch=full.select_idxs(idx)
    batch.meta_info['global_token_num']=batch.batch['attention_mask'].sum(-1).tolist()
    assert 'global_batch_info' not in batch.meta_info
    launch=json.loads((RUN/'launch.json').read_bytes())
    cfg=OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    for key,value in launch['options'].items():OmegaConf.update(cfg,key.lstrip('+'),value,force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps=330
    env={k:v for k,v in launch['effective_environment'].items() if k not in ('CUDA_VISIBLE_DEVICES','RAY_ADDRESS','VERL_GLOBAL_TOKEN_MEAN')}
    env['PYTHONPATH']=str(OUT)+os.pathsep+env.get('PYTHONPATH','')
    env['DT_FSDP_EVENT_WAIT_RECEIPTS']=str(DEST/'original-eventwait')
    phase('initializing_original_actor_only',input_sources=sources,rows=64,
        original_checkpoint=str(RUN/'textcraft-dt/checkpoints/global_step_80/actor'))
    ray.init(address='local',num_cpus=8,num_gpus=2,include_dashboard=False,
        _temp_dir='/tmp/ray-tc-pggroups-v1',runtime_env={'env_vars':env})
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
            RayClassWithInitArgs(ObservedWorker,cfg.actor_rollout_ref,'actor'))
        group.init_model()
        group.load_checkpoint(local_path=str(RUN/'textcraft-dt/checkpoints/global_step_80/actor'),del_local_after_load=False)
        bindings=group.bind_saved_weights_rng()
        phase('native_three_group_backward',actual_bindings=bindings)
        group.observe_PG_groups(batch)
        records=[json.loads((DEST/f'rank{r}-groups.json').read_bytes()) for r in (0,1)]
        phase('complete_gradient_localization_not_repair',gradient_statistics=[r['gradient_statistics'] for r in records])
    finally:ray.shutdown()

if __name__=='__main__':main()
