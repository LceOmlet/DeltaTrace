"""Bounded diagnostic of original vLLM handoff, DT and unchanged PPO.

Uses 32 original saved initial prompts for one native generation/context cycle.
It is not a new environment rollout, training budget or throughput benchmark.
Original saved DT and actor carriers then retain their own exact row orders.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import ray
import torch
from verl.protocol import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.base.decorator import Dispatch, register
import diagnose_dt_to_actor as previous

OUT = Path(os.environ['DT_ACTOR_INCIDENT_OUT'])
BaseIncidentWorker = previous.IncidentWorker.__ray_metadata__.modified_class


@ray.remote
class RolloutIncidentWorker(BaseIncidentWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def observe_handoff(self, phase):
        result = self.observe_owner(phase)
        result.update(default_dtype=str(torch.get_default_dtype()),
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                      bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                      fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                      torch_grad_enabled=torch.is_grad_enabled(),
                      lora={}, parameter_dtype_device={})
        for name, parameter in self.actor_module_fsdp.named_parameters():
            key = (str(parameter.dtype), str(parameter.device), str(parameter.requires_grad))
            key = '/'.join(key)
            result['parameter_dtype_device'][key] = result['parameter_dtype_device'].get(key, 0) + 1
            if parameter.requires_grad:
                value = parameter.detach()
                value = value.to_local() if hasattr(value, 'to_local') else value
                value = value.cpu().contiguous()
                result['lora'][name] = dict(shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
                    nonfinite=int((~torch.isfinite(value)).sum()))
        previous.write(OUT / f'rank{self.rank}-{phase}-handoff.json', result)
        return result

    def _build_rollout(self, *args, **kwargs):
        # Passive observations around the original owner's construction.
        self.observe_handoff('before_vllm_init')
        result = super()._build_rollout(*args, **kwargs)
        self.observe_handoff('after_vllm_init')
        return result


def main():
    config, data, dt_data = previous.prepare()
    # These are the original saved prompt tokens, before its response carrier.
    prompt_length = data.batch['input_ids'].shape[1] - data.batch['responses'].shape[1]
    prompts = DataProto.from_dict(tensors={
        key: data.batch[key][:32, :prompt_length].clone()
        for key in ('input_ids', 'attention_mask', 'position_ids')},
        meta_info={'temperature': config.actor_rollout_ref.rollout.temperature})
    previous.write(OUT / 'handoff-input-inspection.json', dict(
        prompt_rows=len(prompts), prompt_width=prompt_length,
        native_generation_response_limit=config.actor_rollout_ref.rollout.response_length,
        sampling='Unchanged original VERL/vLLM sampling parameters; generated output is diagnostic only.',
        actor_rows=len(data), DT_rows=len(dt_data),
        CUDA_initialized=torch.cuda.is_initialized(), script_sha256=previous.sha(__file__)))
    if '--inspect-only' in sys.argv:
        return
    ray.init(num_cpus=8, include_dashboard=False)
    try:
        previous.original.write_phase('native_actor_rollout_init_begin')
        group = RayWorkerGroup(RayResourcePool([2], use_gpu=True, max_colocate_count=1),
            RayClassWithInitArgs(RolloutIncidentWorker, config.actor_rollout_ref, 'actor_rollout'))
        group.init_model()
        group.observe_owner('initial', save_initial=True)
        previous.original.write_phase('native_rollout_context_begin')
        group.begin_rollout_context()
        try:
            previous.original.write_phase('native_generate_begin')
            generated = group.generate_sequences(prompts)
            previous.write(OUT / 'generated-observation.json', {
                k: previous.stats(v) for k, v in generated.batch.items()})
            del generated, prompts
        finally:
            previous.original.write_phase('native_rollout_context_end')
            group.end_rollout_context()
        group.observe_handoff('after_vllm_generation_sleep')
        previous.original.write_phase('native_old_log_prob_begin')
        data.union(group.compute_log_prob(data.select(deepcopy=True)))
        previous.original.write_phase('native_ref_log_prob_begin')
        data.union(group.compute_ref_log_prob(data.select(deepcopy=True)))
        path = OUT / 'actor-input-with-native-old-ref.pt'
        torch.save(dict(batch={k:v.cpu() for k,v in data.batch.items()},
                        non_tensor_batch=data.non_tensor_batch, meta_info=data.meta_info), path)
        previous.write(OUT / 'actor-input-with-native-old-ref.json',
                       dict(path=str(path), sha256=previous.sha(path), bytes=path.stat().st_size))
        group.observe_owner('before_DT')
        previous.original.write_phase('original_DT_begin')
        result = group.compute_dt_token_advantages(dt_data)
        previous.write(OUT / 'DT-output-observation.json', {k:previous.stats(v) for k,v in result.batch.items()})
        del result, dt_data
        group.observe_owner('after_DT')
        previous.original.write_phase('native_update_begin')
        group.diagnose_update(data)
        previous.original.write_phase('complete')
    except BaseException as error:
        previous.original.write_phase('failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
