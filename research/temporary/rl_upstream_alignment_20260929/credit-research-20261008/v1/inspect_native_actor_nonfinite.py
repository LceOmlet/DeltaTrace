"""Observe the original VERL update on its saved first-iteration actor inputs.

Fresh actor initialization, native old/ref log-prob calls, native sequential
optimizer updates and PyTorch's anomaly detector. No credit, loss, mask,
optimizer, dtype, minibatch or numerical tolerance is replaced. The original
old/ref outputs and initial random LoRA A were not saved, so this is a fresh
reproduction, not a bitwise replay of the destroyed historical process.
"""
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

import psutil
import ray
import torch
from omegaconf import OmegaConf
from verl.protocol import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from inspect_extreme_endpoint import check_imports, actor_initialization_steps

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = Path(os.environ['DT_ACTOR_INCIDENT_OUT'])
SOURCE = ROOT/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json'
CAPTURE = ROOT/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
ACTOR_SHA = '3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    path = inspect.getsourcefile(inspect.unwrap(value))
    return dict(path=path, sha256=sha(path))


def stats(value, mask=None):
    value = value.detach()
    if hasattr(value, 'to_local'):
        value = value.to_local()
    if mask is not None:
        value = value[mask.bool()]
    finite = torch.isfinite(value)
    selected = value[finite]
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                nonfinite=int((~finite).sum()),
                min=float(selected.min()) if selected.numel() else None,
                max=float(selected.max()) if selected.numel() else None)


def write_phase(name, **kwargs):
    record = dict(phase=name, unix=time.time(), pid=os.getpid(), **kwargs)
    (OUT/'phase.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


@ray.remote
class ActorIncidentWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def diagnose_native_update(self, data):
        from contextlib import ExitStack
        from unittest.mock import patch
        owner = importlib.import_module(type(self.actor).__module__)
        assert identity(type(self.actor))['sha256'] == ACTOR_SHA
        forward = self.actor._forward_micro_batch
        policy = owner.compute_policy_loss
        penalty = owner.kl_penalty
        step = self.actor._optimizer_step
        report = dict(scope=__doc__, rank=self.rank, pid=os.getpid(),
                      actor=identity(type(self.actor)), policy=identity(policy),
                      penalty=identity(penalty), script_sha256=sha(__file__),
                      microbatches=[], optimizer_steps=[])
        path = OUT/f'rank{self.rank}.json'

        def save():
            report.update(unix=time.time(), pss_bytes=psutil.Process().memory_full_info().pss,
                          allocated=torch.cuda.memory_allocated(),
                          reserved=torch.cuda.memory_reserved())
            path.write_text(json.dumps(report, indent=2)+'\n')

        def observed_forward(micro_batch, temperature, calculate_entropy=False):
            row = dict(index=len(report['microbatches']), phase='forward_begin',
                       unix=time.time(), input_shape=list(micro_batch['input_ids'].shape))
            report['microbatches'].append(row)
            save()
            result = forward(micro_batch, temperature, calculate_entropy)
            row.update(phase='forward_complete', entropy=None if result[0] is None else stats(result[0]),
                       log_prob=stats(result[1]), forward_completed_unix=time.time())
            for label, value in zip(('entropy', 'log_prob'), result):
                if value is not None and value.requires_grad:
                    def observe_output_gradient(gradient, name=label):
                        row[name+'_gradient'] = stats(gradient)
                        save()
                        return gradient
                    value.register_hook(observe_output_gradient)
            save()
            return result

        def observed_policy(*args, **kwargs):
            row = report['microbatches'][-1]
            row['policy_inputs'] = {name: stats(kwargs[name], kwargs['response_mask'])
                                    for name in ('old_log_prob', 'log_prob', 'advantages')}
            row['log_prob_minus_old_all_slots'] = stats(kwargs['log_prob']-kwargs['old_log_prob'])
            result = policy(*args, **kwargs)
            row['policy_outputs'] = [stats(value) for value in result]
            row['phase'] = 'policy_complete'
            save()
            return result

        def observed_penalty(*args, **kwargs):
            result = penalty(*args, **kwargs)
            row = report['microbatches'][-1]
            row['ref_minus_log_prob_all_slots'] = stats(kwargs['ref_logprob']-kwargs['logprob'])
            row['kl_penalty'] = stats(result)
            row['phase'] = 'kl_complete_before_backward'
            save()
            return result

        def observed_step():
            records = {name: stats(parameter.grad) for name, parameter
                       in self.actor_module_fsdp.named_parameters()
                       if parameter.requires_grad and parameter.grad is not None}
            item = dict(index=len(report['optimizer_steps']),
                        gradients=records, before_native_step_unix=time.time())
            report['optimizer_steps'].append(item)
            save()
            result = step()
            item.update(native_grad_norm=float(result), completed_unix=time.time())
            save()
            return result

        save()
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(self.actor, '_forward_micro_batch', observed_forward))
                stack.enter_context(patch.object(owner, 'compute_policy_loss', observed_policy))
                stack.enter_context(patch.object(owner, 'kl_penalty', observed_penalty))
                stack.enter_context(patch.object(self.actor, '_optimizer_step', observed_step))
                stack.enter_context(torch.autograd.detect_anomaly(check_nan=True))
                output = super().update_actor(data)
            report.update(phase='complete', metrics=output.meta_info['metrics'])
            save()
            return output
        except Exception as error:
            report.update(phase='failed', error=repr(error))
            save()
            raise


def prepare():
    torch.set_num_threads(4)
    assert sha(SOURCE) == '2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52'
    source = json.loads(SOURCE.read_bytes())
    check_imports(source)
    paths = [CAPTURE/f'rank{rank}-pre-update.pt' for rank in (0, 1)]
    expected = ['1563ce74f298769893b360398fd1bacf16c376439b1d462e56ef9dc6f905be59',
                'a1970da0bbf462b203314cbf29cc8ecd8c81e526fe1b6ee944978a76bbadd34e']
    assert [sha(p) for p in paths] == expected
    saved = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
    tensors = {key: torch.cat([s['tensors'][key] for s in saved]) for key in saved[0]['tensors']}
    tensors['position_ids'] = compute_position_id_with_mask(tensors['attention_mask'])
    config = OmegaConf.load(Path(source['verl_root'])/'verl/trainer/config/ppo_trainer.yaml')
    for key, value in source['startup_options'].items():
        OmegaConf.update(config, key.lstrip('+'), value, force_add=True)
    config.actor_rollout_ref.actor.optim.total_training_steps = actor_initialization_steps(
        config.trainer.total_training_steps, 330)
    data = DataProto.from_dict(tensors=tensors,
        non_tensors={'traj_uid': [str(u) for s in saved for u in s['non_tensors']['traj_uid']]},
        meta_info=dict(temperature=config.actor_rollout_ref.rollout.temperature, multi_turn=True,
                       global_token_num=tensors['attention_mask'].sum(-1).tolist()))
    receipt = dict(scope=__doc__, source_sha256=sha(SOURCE), inputs=[dict(path=str(p), sha256=sha(p)) for p in paths],
                   rows=len(data), local_original_minibatch=32, microbatch=4,
                   lora_rank=config.actor_rollout_ref.model.lora_rank,
                   lora_alpha=config.actor_rollout_ref.model.lora_alpha,
                   cuda_initialized=torch.cuda.is_initialized(), script_sha256=sha(__file__))
    (OUT/'input-inspection.json').write_text(json.dumps(receipt, indent=2)+'\n')
    (OUT/'effective-config.yaml').write_text(OmegaConf.to_yaml(config))
    return config, data


def main():
    config, data = prepare()
    if '--inspect-only' in sys.argv:
        return
    ray.init(num_cpus=8, include_dashboard=False)
    try:
        write_phase('native_actor_init_begin')
        group = RayWorkerGroup(RayResourcePool([2], use_gpu=True, max_colocate_count=1),
            RayClassWithInitArgs(ActorIncidentWorker, config.actor_rollout_ref, 'actor'))
        group.init_model()
        write_phase('native_old_log_prob_begin')
        data.union(group.compute_log_prob(data.select(deepcopy=True)))
        write_phase('native_ref_log_prob_begin')
        data.union(group.compute_ref_log_prob(data.select(deepcopy=True)))
        write_phase('native_update_begin')
        group.diagnose_native_update(data)
        write_phase('complete')
    except Exception as error:
        write_phase('failed', error=repr(error))
        raise
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
