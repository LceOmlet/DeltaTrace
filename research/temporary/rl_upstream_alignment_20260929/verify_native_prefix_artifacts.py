"""Bounded original-VERL B4/rank prefix artifact composition on saved real IDs.

One native prefill, owner Cache composition, and one native shorter-prefix
comparison. No optimization step, new advantage or invented numeric tolerance.
The candidate is not deployed to the existing formal jobs.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import ray
import torch
from omegaconf import OmegaConf
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker

OUT = Path(os.environ['DT_PREFIX_PROBE_ROOT'])


def tensors(cache):
    from transformers.cache_utils import LinearAttentionCacheLayerMixin
    return {(i, name): getattr(layer, name).detach().cpu()
            for i, layer in enumerate(cache.layers)
            for name in (('conv_states', 'recurrent_states')
                         if isinstance(layer, LinearAttentionCacheLayerMixin)
                         else ('keys', 'values'))}


@ray.remote
class PrefixWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def check_prefix(self):
        from contextlib import nullcontext
        from deltatrace_rollout import DeltaTraceRolloutProducer
        from native_prefix_artifacts_candidate import NativePrefixArtifacts
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

        result = dict(rank=self.rank, pid=os.getpid(), scope=__doc__, phase='started',
                      numerical_acceptance='No custom tolerance or complete-DT acceptance is defined by this API probe.')
        path = OUT / f'rank{self.rank}.json'
        def save(phase, **values):
            result.update(phase=phase, observed_unix=time.time(), **values)
            path.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(dict(rank=self.rank, phase=phase, **values)), flush=True)
        assert self.config.model.lora_rank == 8 and self.config.model.lora_alpha == 16
        assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
        result['imported_sources'] = {
            name: dict(path=inspect.getsourcefile(owner),
                       sha256=hashlib.sha256(Path(inspect.getsourcefile(owner)).read_bytes()).hexdigest())
            for name, owner in [('verl_worker', ActorRolloutRefWorker),
                                ('dt_producer', DeltaTraceRolloutProducer),
                                ('prepared_artifact_candidate', NativePrefixArtifacts)]}
        if os.environ.get('DT_PREFIX_DT_LEASE_DIAGNOSTIC') == '1':
            save('original_request_lease_setup', rank_lora=8,alpha=16,local_microbatch=4)
        else:
            inputs = json.loads((OUT/'actual-minimum-inputs.json').read_bytes())
            samples = inputs['samples']
            assert len(samples) == 4
            # Factual prompts only: queries, outcome labels and current actions
            # are excluded from the shared capture in this composition probe.
            longest = (max(s['source_start'] for s in samples)+63)//64*64
            prefix = min(s['source_start'] for s in samples)//64*64
            ids = torch.full((4, longest), inputs['eos_token_id'], device='cuda', dtype=torch.long)
            for i, sample in enumerate(samples):
                ids[i, :sample['source_start']] = torch.tensor(
                    sample['selected_input_ids'][:sample['source_start']], device='cuda')
            save('model_prefix_start', shape=list(ids.shape), requested_prefix=prefix,
                 rank_lora=8, alpha=16, local_microbatch=4,
                 source_sha256=hashlib.sha256((OUT/'actual-minimum-inputs.json').read_bytes()).hexdigest())
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        training = self.actor_module_fsdp.training
        self.actor_module_fsdp.eval()
        producer = DeltaTraceRolloutProducer(self.actor_module_fsdp,
            eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
            invalid_action_penalty_coef=self.config.actor.invalid_action_penalty_coef)
        runner = producer.runner
        text = runner.model.model.language_model
        previous_attention = text.config._attn_implementation
        text.set_attn_implementation('flash_attention_2')
        if producer.native_fla_fp16:
            from accelerated.qwen35.native_fla_precision import native_fla_fp16
            precision = native_fla_fp16(self.actor_module_fsdp)
        else:
            precision = nullcontext()
        try:
            with torch.no_grad(), precision:
                torch.cuda.reset_peak_memory_stats()
                if os.environ.get('DT_PREFIX_DT_LEASE_DIAGNOSTIC') == '1':
                    from diagnose_native_prefix_leases import diagnose
                    diagnose(runner, producer, OUT, save)
                    return result
                if os.environ.get('DT_PREFIX_DT_SEAM_DIAGNOSTIC') == '1':
                    from diagnose_native_prefix_dt_seam import diagnose
                    diagnose(runner, producer, inputs, OUT, save, cache_tensors=tensors)
                    return result
                if os.environ.get('DT_PREFIX_COMPONENT_DIAGNOSTIC') == '1':
                    from diagnose_native_prefix_components import diagnose
                    diagnose(runner, ids, prefix, save)
                    return result
                full_reference = {}
                def observe(cache):
                    full_reference.update(tensors(cache))
                start = time.perf_counter()
                shared = NativePrefixArtifacts.capture(runner.model, ids, [prefix, longest],
                    config=runner.model._conditional.config, observe_native_cache=observe)
                torch.cuda.synchronize()
                save('shared_capture_complete', capture_seconds=time.perf_counter()-start)
                start = time.perf_counter()
                cache = shared.materialize(longest, device='cuda')
                composed = tensors(cache)
                same_call = []
                for key, expected in full_reference.items():
                    actual = composed[key]
                    same_call.append(dict(layer=key[0], field=key[1], dtype=str(actual.dtype),
                        shape=list(actual.shape), equal=bool(torch.equal(actual, expected)),
                        max_absolute_difference=float((actual.float()-expected.float()).abs().max())))
                # Exact data copying and the same owner dtype conversion are
                # representation contracts, not floating-point tolerances.
                assert all(row['equal'] for row in same_call), same_call
                del cache, composed, full_reference
                save('same_forward_composition_exact', full_composition_seconds=time.perf_counter()-start,
                     full_prefix_fields=same_call)
                start = time.perf_counter()
                cache = shared.materialize(prefix, device='cuda')
                composed = tensors(cache)
                del cache
                shorter = runner.forward_prefix(ids[:, :prefix])
                native = tensors(shorter.past_key_values)
                torch.cuda.synchronize()
                comparison=[]
                for key, expected in native.items():
                    actual = composed[key]
                    comparison.append(dict(layer=key[0], field=key[1], dtype=str(actual.dtype),
                        shape=list(actual.shape), equal=bool(torch.equal(actual, expected)),
                        max_absolute_difference=float((actual.float()-expected.float()).abs().max())))
                save('shorter_prefix_observed', shorter_comparison_seconds=time.perf_counter()-start,
                     shorter_prefix_fields=comparison, physical_free_bytes=torch.cuda.mem_get_info()[0],
                     peak_torch_allocated_bytes=torch.cuda.max_memory_allocated())
            return result
        finally:
            text.set_attn_implementation(previous_attention)
            self.actor_module_fsdp.train(training)
            runner.model.release_owner_params()
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)


if __name__ == '__main__':
    cfg = OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    for key, value in json.loads((OUT/'native-launch-options.json').read_bytes()).items():
        OmegaConf.update(cfg, key.lstrip('+'), value, force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps = cfg.trainer.total_training_steps
    (OUT/'effective-config.yaml').write_text(OmegaConf.to_yaml(cfg))
    ray.init(num_cpus=8, include_dashboard=False)
    try:
        group = RayWorkerGroup(RayResourcePool([2], use_gpu=True, max_colocate_count=1),
            RayClassWithInitArgs(PrefixWorker, cfg.actor_rollout_ref, 'actor'))
        group.init_model()
        results = group.check_prefix()
        (OUT/'result.json').write_text(json.dumps(results, indent=2)+'\n')
    finally:
        ray.shutdown()
