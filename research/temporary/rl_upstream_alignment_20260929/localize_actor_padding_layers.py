"""Observe original actor layers on the saved padding diagnostic, without updates.

Uses the owner's standalone actor role. No rollout engine, task sampler, model
operation, numerical tolerance, or formal training configuration is replaced.
"""
import importlib.util
import json
import os
from pathlib import Path
import time
from types import MethodType

import ray
import torch
from omegaconf import OmegaConf
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker

OUT = Path(os.environ['PADDING_DIAGNOSTIC_DIR'])
LENGTHS = [512, 1024, 2048, 4096]


def valid_sample(value):
    assert isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[0] == 4
    return torch.stack([value[i, value.shape[1]-n:value.shape[1]-n+512]
                        for i, n in enumerate(LENGTHS)]).detach().cpu().clone()


@ray.remote
class ObservedWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def select_padding_owner(self, mode):
        assert self.role == 'actor' and not self._is_rollout
        assert self.config.model.lora_rank == 8 and self.config.model.lora_alpha == 16
        assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
        if not hasattr(self, 'candidate_forward'):
            self.candidate_forward = self.actor._forward_micro_batch
            path = Path(os.environ['PADDING_REFERENCE_ROOT'])/'verl/workers/actor/dp_actor.py'
            spec = importlib.util.spec_from_file_location('padding_reference_owner', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.reference_forward = MethodType(module.DataParallelPPOActor._forward_micro_batch, self.actor)
            self.samples = {}
        self.mode = mode
        self.samples[mode] = {}
        self.hooks = []
        self.model_shapes = []
        def model_shape(_module, _args, kwargs):
            self.model_shapes.append(list(kwargs['input_ids'].shape))
        self.hooks.append(self.actor.actor_module.register_forward_pre_hook(model_shape, with_kwargs=True))
        layers = [(n,m) for n,m in self.actor.actor_module.named_modules()
                  if isinstance(m, Qwen3_5DecoderLayer)]
        (OUT/f'layer-inventory-rank{self.rank}.json').write_text(json.dumps(
            [dict(name=n,actual_type=type(m).__name__) for n,m in layers],indent=2)+'\n')
        text_config = getattr(self.actor_model_config, 'text_config', self.actor_model_config)
        assert len(layers) == text_config.num_hidden_layers
        for index, (name, layer) in enumerate(layers):
            def before(_module, args, kwargs, index=index, name=name):
                if index == 0:
                    value = kwargs.get('hidden_states', args[0] if args else None)
                    self.samples[mode]['input'] = valid_sample(value)
            def after(_module, _args, output, index=index, name=name):
                self.samples[mode][name] = valid_sample(output)
                print(json.dumps(dict(phase='layer_captured',rank=self.rank,mode=mode,layer=index)),flush=True)
            self.hooks.append(layer.register_forward_pre_hook(before, with_kwargs=True))
            self.hooks.append(layer.register_forward_hook(after))
        self.actor._forward_micro_batch = self.reference_forward if mode == 'reference' else self.candidate_forward
        return dict(rank=self.rank,mode=mode,layers=[n for n,_ in layers],role=self.role)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def finish_readout(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        torch.save(self.samples[self.mode],OUT/f'{self.mode}-layers-rank{self.rank}.pt')
        result=dict(rank=self.rank,mode=self.mode,model_shapes=self.model_shapes,layers=[])
        if self.mode == 'candidate':
            assert self.samples['reference'].keys() == self.samples['candidate'].keys()
            for name, value in self.samples['candidate'].items():
                base=self.samples['reference'][name]
                diff=value.float()-base.float()
                result['layers'].append(dict(name=name,dtype=str(value.dtype),shape=list(value.shape),
                    max_abs=float(diff.abs().max()),rms=float(diff.square().mean().sqrt()),
                    reference_rms=float(base.float().square().mean().sqrt()),
                    exact_elements=int((value==base).sum()),elements=value.numel()))
        (OUT/f'{self.mode}-rank{self.rank}.json').write_text(json.dumps(result,indent=2)+'\n')
        return result


if __name__ == '__main__':
    source=json.loads((OUT/'source.json').read_text())
    cfg=OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    options=json.loads(Path(source['formal_launch']).read_text())['options']
    for key,value in options.items():OmegaConf.update(cfg,key.lstrip('+'),value,force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps=source['original_total_training_steps']
    (OUT/'launch-config.yaml').write_text(OmegaConf.to_yaml(cfg))
    saved=torch.load(source['saved_inputs'],map_location='cpu',weights_only=False)
    data=DataProto.from_single_dict(saved['batch'])
    data.meta_info.update(temperature=cfg.actor_rollout_ref.rollout.temperature,multi_turn=True,global_token_num=LENGTHS*2)
    width=data.batch['responses'].shape[1]
    selected=data.batch['loss_mask'][:,-width:].bool()
    state=dict(scope=__doc__,phase='initializing',source=source,readouts=[])
    def record(phase,**values):
        state.update(phase=phase,**values)
        (OUT/'result.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(dict(phase=phase,**values)),flush=True)
    ray.init(num_cpus=8,include_dashboard=False)
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
            RayClassWithInitArgs(ObservedWorker,cfg.actor_rollout_ref,'actor'))
        group.init_model()
        scores={}
        for mode in ('reference','candidate'):
            group.select_padding_owner(mode)
            record(mode+'_logprob_start')
            start=time.perf_counter()
            scores[mode]=group.compute_log_prob(data).batch['old_log_probs']
            elapsed=time.perf_counter()-start
            records=group.finish_readout()
            state['readouts'].append(dict(mode=mode,seconds=elapsed,workers=records))
            record(mode+'_logprob_complete')
        error=(scores['candidate']-scores['reference'])[selected]
        torch.save(scores,OUT/'readouts.pt')
        record('localization_complete',differences=dict(max_abs=float(error.abs().max()),
            rms=float(error.square().mean().sqrt()),tokens=int(selected.sum())),
            scope_of_result='Layer localization only; no new pass threshold, optimizer update or deployment')
    finally:
        ray.shutdown()
