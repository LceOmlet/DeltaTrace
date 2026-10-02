"""Same-weight padding diagnostics and one native update, not a task pilot.

Model-output differences are reported, without inventing a whole-network
tolerance. The unmodified owner loss/optimizer/update and B4/rank are used.
The padded fixture does not replace the prior fully occupied B8 x 32768 proof.
"""
import importlib.util
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from types import MethodType

import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

OUT = Path(os.environ['PADDING_DIAGNOSTIC_DIR'])
_source = json.loads((OUT/'source.json').read_text())
_launch = json.loads(Path(_source['formal_launch']).read_text())['options']
_rollout_mode = _launch.get('actor_rollout_ref.rollout.mode', 'sync')
# Same public class selection as the original main_ppo TaskRunner.
_OWNER_WORKER = AsyncActorRolloutRefWorker if _rollout_mode == 'async' else ActorRolloutRefWorker


@ray.remote
class ObservedWorker(_OWNER_WORKER):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def select_padding_owner(self, mode):
        assert self.config.model.lora_rank == 8 and self.config.model.lora_alpha == 16
        assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
        assert self.config.rollout.log_prob_micro_batch_size_per_gpu == 4
        if not hasattr(self, 'padding_candidate_forward'):
            self.padding_candidate_forward = self.actor._forward_micro_batch
            path = Path(os.environ['PADDING_REFERENCE_ROOT'])/'verl/workers/actor/dp_actor.py'
            spec = importlib.util.spec_from_file_location('padding_reference_owner', path)
            module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self.padding_reference_forward = MethodType(module.DataParallelPPOActor._forward_micro_batch, self.actor)
            self.padding_observations = []
            self.padding_before = {n:(p.to_local() if hasattr(p,'to_local') else p).detach().cpu().clone()
                for n,p in self.actor_module_fsdp.named_parameters() if p.requires_grad}
            if self.config.rollout.mode != 'async':
                with self.rollout_sharding_manager:
                    pass
        original = self.padding_reference_forward if mode == 'reference' else self.padding_candidate_forward
        def observed(*args, **kwargs):
            batch = args[0] if args else kwargs['micro_batch']
            assert list(batch['input_ids'].shape) == [4,32768]
            shapes = []
            def shape_hook(_module, _args, model_kwargs):
                shapes.append(list(model_kwargs['input_ids'].shape))
            hook = self.actor.actor_module.register_forward_pre_hook(shape_hook, with_kwargs=True)
            start = time.perf_counter()
            try:
                outputs = original(*args, **kwargs)
                torch.cuda.synchronize()
            finally:
                hook.remove()
            self.padding_observations.append(dict(mode=mode,model_shapes=shapes,seconds=time.perf_counter()-start))
            (OUT/f'rank{self.rank}.json').write_text(json.dumps(self.padding_observations,indent=2)+'\n')
            return outputs
        self.actor._forward_micro_batch = observed
        actor_source = Path(inspect.getsourcefile(self.actor.__class__))
        return dict(rank=self.rank, mode=mode, actor_source=str(actor_source),
                    actor_sha256=hashlib.sha256(actor_source.read_bytes()).hexdigest(),
                    lora_rank=self.config.model.lora_rank, lora_alpha=self.config.model.lora_alpha,
                    microbatch_per_gpu=self.actor.config.ppo_micro_batch_size_per_gpu,
                    use_fused_kernels=self.actor.use_fused_kernels,
                    entropy_coeff=self.actor.config.entropy_coeff,
                    clip_ratio_c=self.actor.config.clip_ratio_c)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def finish_padding_observation(self):
        changed = 0
        for name,p in self.actor_module_fsdp.named_parameters():
            if p.requires_grad:
                value=(p.to_local() if hasattr(p,'to_local') else p).detach().cpu()
                assert torch.isfinite(value).all(),name
                changed += int(not torch.equal(value,self.padding_before[name]))
        steps = sorted({float(s['step'].item()) for s in self.actor.actor_optimizer.state.values() if 'step' in s})
        assert changed > 0
        if self.config.rollout.mode != 'async':
            with self.rollout_sharding_manager:
                pass
        return dict(rank=self.rank,calls=self.padding_observations,changed_parameter_tensors=changed,
                    native_optimizer_steps=steps,
                    post_update_native_sync_sleep=self.config.rollout.mode != 'async')


if __name__ == '__main__':
    source=json.loads((OUT/'source.json').read_text())
    cfg=OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    options=json.loads(Path(source['formal_launch']).read_text())['options']
    for key,value in options.items():OmegaConf.update(cfg,key.lstrip('+'),value,force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps=source['original_total_training_steps']
    (OUT/'launch-config.yaml').write_text(OmegaConf.to_yaml(cfg))
    tokenizer=AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'],local_files_only=True)
    text=tokenizer.encode('Read the database schema and select the matching rows. ',add_special_tokens=False)
    ids=torch.full((8,32768),tokenizer.pad_token_id,dtype=torch.long)
    mask=torch.zeros_like(ids);loss_mask=torch.zeros_like(ids)
    lengths=[512,1024,2048,4096]*2
    for row,length in enumerate(lengths):
        ids[row,-length:]=torch.tensor((text*(length//len(text)+1))[:length])
        mask[row,-length:]=1;loss_mask[row,-(length-256):]=1
    width=32256
    data=DataProto.from_single_dict(dict(input_ids=ids,attention_mask=mask,loss_mask=loss_mask,
        position_ids=(mask.cumsum(-1)-1).clamp_min(0),responses=ids[:,-width:]))
    data.meta_info.update(temperature=cfg.actor_rollout_ref.rollout.temperature,multi_turn=True,global_token_num=lengths)
    selected=loss_mask[:,-width:].bool()
    state=dict(scope=__doc__,phase='initializing',readouts=[],source=source)
    def record(phase,**values):
        state.update(phase=phase,**values)
        (OUT/'result.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(dict(phase=phase,**values)),flush=True)
    ray.init(num_cpus=8,include_dashboard=False)
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
            RayClassWithInitArgs(ObservedWorker,cfg.actor_rollout_ref,'actor_rollout'))
        group.init_model()
        scores={}
        for mode in ('reference','candidate','reference','candidate'):
            group.select_padding_owner(mode);record(mode+'_logprob_start')
            start=time.perf_counter();lp=group.compute_log_prob(data).batch['old_log_probs']
            assert lp.shape==(8,width) and torch.isfinite(lp).all()
            scores[mode]=lp
            state['readouts'].append(dict(mode=mode,seconds=time.perf_counter()-start))
            record(mode+'_logprob_complete')
        error=(scores['candidate']-scores['reference'])[selected]
        differences=dict(max_abs=float(error.abs().max()),mean_abs=float(error.abs().mean()),
                         rms=float(error.square().mean().sqrt()),tokens=int(selected.sum()),
                         scope='Observed end-to-end differences only; no invented pass threshold')
        torch.save(dict(batch=data.batch.to_dict(),reference=scores['reference'],candidate=scores['candidate']),OUT/'same-input-readouts.pt')
        data.batch['old_log_probs']=scores['candidate'];data.batch['ref_log_prob']=scores['candidate']
        data.batch['advantages']=torch.linspace(-.2,.3,width).expand(8,-1).clone()
        record('candidate_native_update_start',differences=differences)
        start=time.perf_counter();metrics=group.update_actor(data).meta_info['metrics']
        record('candidate_native_update_complete',update_seconds=time.perf_counter()-start,metrics=metrics)
        workers=group.finish_padding_observation()
        record('diagnostic_complete',workers=workers)
    finally:
        ray.shutdown()
