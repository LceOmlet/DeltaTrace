"""Replay saved real DT inputs before the unchanged original VERL actor update.

This adds passive observations and native PyTorch anomaly detection only.
Saved actor advantages stay unchanged to isolate lifecycle effects. The lost
historical LoRA initialization is not reconstructed or called an exact replay.
"""
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import time
import traceback
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import psutil
import ray
import torch
from verl.protocol import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker
import inspect_native_actor_nonfinite as original

OUT = Path(os.environ['DT_ACTOR_INCIDENT_OUT'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(value):
    value = value.detach()
    if hasattr(value, 'to_local'):
        value = value.to_local()
    finite = torch.isfinite(value)
    selected = value[finite]
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                nonfinite=int((~finite).sum()),
                min=float(selected.min()) if selected.numel() else None,
                max=float(selected.max()) if selected.numel() else None)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


@ray.remote
class IncidentWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def observe_owner(self, phase, save_initial=False):
        states = []
        for name, module in self.actor_module_fsdp.named_modules():
            if hasattr(module, '_get_fsdp_state'):
                state = module._get_fsdp_state()
                group = state._fsdp_param_group
                states.append(dict(name=name, training=module.training,
                    state_training=str(getattr(state, '_training_state', None)),
                    group_training=str(getattr(group, '_training_state', None)),
                    sharded_state=str(getattr(group, '_sharded_state', None)),
                    auto_reshard=str(getattr(state, '_auto_reshard_after_forward', None)),
                    post_forward_mesh=str(getattr(group, 'post_forward_mesh_info', None))))
        record = dict(phase=phase, unix=time.time(), rank=self.rank,
                      actor=original.identity(type(self.actor)),
                      parameter_grad_present=[name for name, p in self.actor_module_fsdp.named_parameters()
                                              if p.grad is not None],
                      fsdp_states=states, TF32=torch.backends.cuda.matmul.allow_tf32,
                      matmul_precision=torch.get_float32_matmul_precision(),
                      root_training=self.actor_module_fsdp.training,
                      allocated=torch.cuda.memory_allocated(),
                      reserved=torch.cuda.memory_reserved(),
                      pss_bytes=psutil.Process().memory_full_info().pss)
        if save_initial:
            values = {}
            for name, p in self.actor_module_fsdp.named_parameters():
                if p.requires_grad:
                    value = p.detach()
                    value = value.to_local() if hasattr(value, 'to_local') else value
                    values[name] = value.cpu().clone()
            path = OUT / f'rank{self.rank}-initial-local-LoRA-and-rng.pt'
            torch.save(dict(parameters=values, cpu_rng=torch.get_rng_state(),
                            device_rng=torch.cuda.get_rng_state(),
                            purpose='Diagnostic tensor capture only; no checkpoint restore.'), path)
            record['initial_tensor_capture'] = dict(path=str(path), sha256=sha(path),
                                                    tensors=len(values), bytes=path.stat().st_size)
        write(OUT / f'rank{self.rank}-{phase}.json', record)
        return record

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def diagnose_update(self, data):
        owner = importlib.import_module(type(self.actor).__module__)
        assert original.identity(type(self.actor))['sha256'] == original.ACTOR_SHA
        forward, policy, penalty, step = (self.actor._forward_micro_batch,
            owner.compute_policy_loss, owner.kl_penalty, self.actor._optimizer_step)
        record = dict(scope=__doc__, rank=self.rank, pid=os.getpid(),
                      birth=psutil.Process().create_time(), script_sha256=sha(__file__),
                      actor=original.identity(type(self.actor)),
                      policy=original.identity(policy), penalty=original.identity(penalty),
                      microbatches=[], optimizer_steps=[])
        path = OUT / f'rank{self.rank}.json'
        events = (OUT / f'rank{self.rank}-events.jsonl').open('a', buffering=1)

        def save(phase):
            record.update(phase=phase, unix=time.time(),
                          pss_bytes=psutil.Process().memory_full_info().pss,
                          allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
            write(path, record)
            events.write(json.dumps(dict(phase=phase, unix=record['unix'],
                         microbatches=len(record['microbatches']),
                         optimizer_steps=len(record['optimizer_steps']))) + '\n')

        def observed_forward(micro_batch, temperature, calculate_entropy=False):
            row = dict(index=len(record['microbatches']), phase='forward_begin',
                       input_shape=list(micro_batch['input_ids'].shape), unix=time.time())
            record['microbatches'].append(row)
            save('forward_begin')
            result = forward(micro_batch, temperature, calculate_entropy)
            row.update(phase='forward_complete',
                       entropy=None if result[0] is None else stats(result[0]),
                       log_prob=stats(result[1]))
            for label, value in zip(('entropy', 'log_prob'), result):
                if value is not None and value.requires_grad:
                    def observe(gradient, label=label, row=row):
                        row[label + '_gradient'] = stats(gradient)
                        save('native_output_gradient')
                        return gradient
                    value.register_hook(observe)
            save('forward_complete')
            return result

        def observed_policy(*args, **kwargs):
            row = record['microbatches'][-1]
            result = policy(*args, **kwargs)
            mask = kwargs['response_mask'].bool()
            row.update(policy_outputs=[stats(x) for x in result],
                       valid_action_tokens=int(mask.sum()),
                       log_prob_minus_old_all=stats(kwargs['log_prob'] - kwargs['old_log_prob']),
                       log_prob_minus_old_action=stats((kwargs['log_prob'] - kwargs['old_log_prob'])[mask]),
                       advantages=stats(kwargs['advantages'][mask]))
            save('policy_complete')
            return result

        def observed_penalty(*args, **kwargs):
            result = penalty(*args, **kwargs)
            record['microbatches'][-1].update(
                kl_penalty=stats(result),
                ref_minus_log_prob=stats(kwargs['ref_logprob'] - kwargs['logprob']),
                phase='before_backward')
            save('before_backward')
            return result

        def observed_step():
            # Inspect the exact local gradients before native clipping; never
            # replace the norm, clip, optimizer or skip-on-nonfinite behavior.
            bad = []
            count = 0
            for name, parameter in self.actor_module_fsdp.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    count += 1
                    value = parameter.grad.detach()
                    value = value.to_local() if hasattr(value, 'to_local') else value
                    if not bool(torch.isfinite(value).all()):
                        bad.append(dict(name=name, stats=stats(value)))
            item = dict(index=len(record['optimizer_steps']),
                        observed_gradient_tensors=count, nonfinite_gradients=bad,
                        before_native_step_unix=time.time())
            record['optimizer_steps'].append(item)
            save('before_native_optimizer_step')
            result = step()
            item.update(native_grad_norm=float(result), completed_unix=time.time())
            save('native_optimizer_step_complete')
            return result

        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(self.actor, '_forward_micro_batch', observed_forward))
                stack.enter_context(patch.object(owner, 'compute_policy_loss', observed_policy))
                stack.enter_context(patch.object(owner, 'kl_penalty', observed_penalty))
                stack.enter_context(patch.object(self.actor, '_optimizer_step', observed_step))
                stack.enter_context(torch.autograd.detect_anomaly(check_nan=True))
                output = super().update_actor(data)
            record['metrics'] = output.meta_info['metrics']
            save('complete')
            return output
        except BaseException as error:
            record.update(error=repr(error), traceback=traceback.format_exc())
            save('failed')
            raise
        finally:
            events.close()


def prepare():
    config, data = original.prepare()
    paths = [original.CAPTURE / f'rank{rank}-input-prepared.pt' for rank in (0, 1)]
    expected = ['6193ca3bc8ee1699084eb4792ad8a6926d7b4fda93af0b288113c2f740469780',
                'aeec0e248ca6ef328131260bd0785f83d77cd5917c19303543f9f71bb12d1fa1']
    assert [sha(p) for p in paths] == expected
    saved = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
    tensors = {key: torch.cat([s['batch'][key] for s in saved]) for key in saved[0]['batch']}
    non_tensors = {key: np.concatenate([s['non_tensor_batch'][key] for s in saved])
                   for key in saved[0]['non_tensor_batch']}
    assert saved[0]['meta_info'] == saved[1]['meta_info']
    dt_data = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors,
                                  meta_info=saved[0]['meta_info'])
    assert len(data) == len(dt_data) == 256
    assert list(data.non_tensor_batch['traj_uid']) == list(dt_data.non_tensor_batch['traj_uid'])
    for key in ('input_ids', 'attention_mask', 'responses'):
        assert torch.equal(data.batch[key], dt_data.batch[key])
    write(OUT / 'DT-input-inspection.json', dict(
        source_sha256=sha(original.SOURCE), inputs=[dict(path=str(p), sha256=sha(p)) for p in paths],
        rows=len(dt_data), meta_info=dt_data.meta_info, script_sha256=sha(__file__),
        actor_advantages='Original saved whole-batch whitened values; recomputed DT output is observed, not substituted.',
        CUDA_initialized=torch.cuda.is_initialized()))
    return config, data, dt_data


def main():
    import sys
    config, data, dt_data = prepare()
    if '--inspect-only' in sys.argv:
        return
    ray.init(num_cpus=8, include_dashboard=False)
    try:
        original.write_phase('native_actor_init_begin')
        group = RayWorkerGroup(RayResourcePool([2], use_gpu=True, max_colocate_count=1),
            RayClassWithInitArgs(IncidentWorker, config.actor_rollout_ref, 'actor'))
        group.init_model()
        group.observe_owner('initial', save_initial=True)
        original.write_phase('native_old_log_prob_begin')
        data.union(group.compute_log_prob(data.select(deepcopy=True)))
        original.write_phase('native_ref_log_prob_begin')
        data.union(group.compute_ref_log_prob(data.select(deepcopy=True)))
        # Retain the values that were missing from the historical incident.
        # This is an input capture, never a training checkpoint or a restore.
        input_path = OUT / 'actor-input-with-native-old-ref.pt'
        torch.save(dict(batch={k: v.cpu() for k, v in data.batch.items()},
                        non_tensor_batch=data.non_tensor_batch, meta_info=data.meta_info), input_path)
        write(OUT / 'actor-input-with-native-old-ref.json',
              dict(path=str(input_path), sha256=sha(input_path), bytes=input_path.stat().st_size))
        group.observe_owner('before_DT')
        original.write_phase('original_DT_begin')
        dt_result = group.compute_dt_token_advantages(dt_data)
        write(OUT / 'DT-output-observation.json', {k: stats(v) for k, v in dt_result.batch.items()})
        del dt_result, dt_data
        group.observe_owner('after_DT')
        original.write_phase('native_update_begin')
        group.diagnose_update(data)
        original.write_phase('complete')
    except BaseException as error:
        original.write_phase('failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
