"""Submit one B8 x 32768 batch through original VERL DP dispatch (B4/rank)."""
import json,os,time
from pathlib import Path
import ray,torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch,register
from verl.single_controller.ray import RayClassWithInitArgs,RayResourcePool,RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from launch_textcraft_native import options_for

ROOT=Path(os.environ['DT_RUNTIME_ROOT'])
OUT=ROOT/'receipts/owner-b8-dispatch-20260930'
OUT.mkdir(exist_ok=True)

@ray.remote
class ObservedWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def begin_observation(self):
        assert self.config.model.lora_rank==8 and self.config.model.lora_alpha==16
        assert self.actor.config.ppo_micro_batch_size_per_gpu==4
        assert self.config.rollout.log_prob_micro_batch_size_per_gpu==4
        assert self.actor.use_fused_kernels
        self.observations=dict(rank=self.rank,config=OmegaConf.to_container(self.config,resolve=True),calls=[])
        def local(p):return p.to_local() if hasattr(p,'to_local') else p
        self.original_parameters={n:local(p).detach().cpu().clone() for n,p in self.actor_module_fsdp.named_parameters() if p.requires_grad}
        original=self.actor._forward_micro_batch
        def observed(*args,**kwargs):
            batch=args[0] if args else kwargs['micro_batch']
            shape=list(batch['input_ids'].shape)
            assert shape==[4,32768],shape
            start=time.perf_counter()
            outputs=original(*args,**kwargs)
            torch.cuda.synchronize()
            self.observations['calls'].append(dict(shape=shape,seconds=time.perf_counter()-start,
                calculate_entropy=kwargs.get('calculate_entropy')))
            (OUT/f'rank{self.rank}.json').write_text(json.dumps(self.observations,indent=2)+'\n')
            return outputs
        self.actor._forward_micro_batch=observed
        with self.rollout_sharding_manager:pass
        torch.cuda.reset_peak_memory_stats()
        return dict(rank=self.rank,ready=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def finish_observation(self):
        changed=0
        for n,p in self.actor_module_fsdp.named_parameters():
            if p.requires_grad:
                v=(p.to_local() if hasattr(p,'to_local') else p).detach().cpu()
                assert torch.isfinite(v).all(),n
                changed+=int(not torch.equal(v,self.original_parameters[n]))
        assert changed>0
        assert len(self.observations['calls'])==2,self.observations['calls']
        self.observations.update(changed_parameter_tensors=changed,phase='updated',
            allocator_peak_gib=torch.cuda.max_memory_allocated()/2**30)
        t=time.perf_counter()
        with self.rollout_sharding_manager:pass
        self.observations.update(post_update_sync_sleep_seconds=time.perf_counter()-t)
        (OUT/f'rank{self.rank}.json').write_text(json.dumps(self.observations,indent=2)+'\n')
        return self.observations

if __name__=='__main__':
    cfg=OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    opts,_=options_for(ROOT/'datasets/textcraft-native-20260930',OUT/'unused-formal-output')
    for k,v in opts.items():OmegaConf.update(cfg,k.lstrip('+'),v,force_add=True)
    import pyarrow.parquet as pq
    rows=pq.read_metadata(ROOT/'datasets/textcraft-native-20260930/train.parquet').num_rows
    cfg.actor_rollout_ref.actor.optim.total_training_steps=rows//cfg.data.train_batch_size*cfg.trainer.total_epochs
    (OUT/'launch-config.yaml').write_text(OmegaConf.to_yaml(cfg))
    tokenizer=AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'],local_files_only=True)
    base=tokenizer.encode('Read the inventory, collect the required wood, and craft the requested item. ',add_special_tokens=False)
    ids=torch.tensor([(base[offset:]+base[:offset])*3000 for offset in range(8)])[:,:32768]
    mask=torch.ones_like(ids)
    data=DataProto.from_single_dict(dict(input_ids=ids,attention_mask=mask,
        position_ids=torch.arange(32768).unsqueeze(0).expand(8,-1).clone(),
        responses=ids[:,-32256:],loss_mask=mask.clone()))
    data.meta_info.update(temperature=cfg.actor_rollout_ref.rollout.temperature,multi_turn=True,global_token_num=[32768]*8)
    state=dict(scope=__doc__,submitted_shape=list(ids.shape),rank=8,alpha=16,microbatch_per_gpu=4)
    def record(phase,**kw):
        state.update(phase=phase,**kw)
        (OUT/'result.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(dict(phase=phase,**kw)),flush=True)
    ray.init(num_cpus=8,include_dashboard=False)
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
            RayClassWithInitArgs(ObservedWorker,cfg.actor_rollout_ref,'actor_rollout'))
        group.init_model();group.begin_observation()
        record('native_B8_old_logprob_start')
        t=time.perf_counter();old=group.compute_log_prob(data).batch['old_log_probs']
        assert old.shape==(8,32256) and torch.isfinite(old).all()
        record('native_B8_old_logprob_complete',old_logprob_seconds=time.perf_counter()-t)
        data.batch['old_log_probs']=old;data.batch['ref_log_prob']=old+.01
        data.batch['advantages']=torch.linspace(-.2,.3,32256).unsqueeze(0).expand(8,-1).clone()
        record('native_B8_update_start');t=time.perf_counter()
        metrics=group.update_actor(data).meta_info['metrics']
        record('native_B8_update_complete',update_seconds=time.perf_counter()-t,metrics=metrics)
        workers=group.finish_observation()
        record('passed',workers=workers)
    finally:ray.shutdown()
