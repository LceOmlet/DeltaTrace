"""Profile the original DT mixer lifetime and its existing offload option.

Actual saved B4 rows, original VERL actor initialization and original producer.
No model, parser, credit, attention, optimizer or tolerance is reconstructed.
This diagnostic does not restart a formal job or release TextCraft's update.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from types import MethodType
import weakref

import torch

from inspect_extreme_endpoint import check_imports, sha, CASES


def tensors(value, seen=None):
    seen = set() if seen is None else seen
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cuda':
            return []
        storage = value.untyped_storage()
        ptr = storage.data_ptr()
        if ptr in seen:
            return []
        seen.add(ptr)
        return [dict(shape=list(value.shape), dtype=str(value.dtype), ptr=ptr,
                     storage_bytes=storage.nbytes(), logical_bytes=value.numel()*value.element_size())]
    if isinstance(value, dict):
        return [item for child in value.values() for item in tensors(child, seen)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in tensors(child, seen)]
    return []


def cache_metadata(cache):
    return [] if cache is None else [
        dict(layer=i, type=type(layer).__name__, tensors=tensors(vars(layer)))
        for i, layer in enumerate(cache.layers)]


def closure_owners(callback):
    """Inspect the existing callback, without populating frame.f_locals.

    Frame-local dictionaries can themselves prolong old tensor lifetimes.
    This callback already owns its current captures; only metadata escapes.
    """
    local = inspect.getclosurevars(callback).nonlocals
    categories = {key: tensors(local.get(key)) for key in ('x', 'kw')}
    for key in ('dc', 'mc'):
        obj = local.get(key)
        categories[key] = tensors([getattr(obj, 'values', None), getattr(obj, 'endpoints', None)])
    categories['replay_cache'] = cache_metadata(local.get('kw', {}).get('past_key_values'))
    return categories


def make_worker(with_vllm=False):
    import ray
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
    owner = AsyncActorRolloutRefWorker if with_vllm else ActorRolloutRefWorker

    @ray.remote
    class MemoryWorker(owner):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def profile_existing(self, source_path, native_path, output, failed_batch):
            import psutil
            import reward_readout
            from deltatrace_rollout import DeltaTraceRolloutProducer
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

            source = json.loads(Path(source_path).read_bytes())
            check_imports(source)
            assert sha(source_path) == CASES['appworld']['source_sha256']
            native = torch.load(native_path, map_location='cpu', weights_only=False)
            rows = [row['row'] for row in sorted(native['rows'], key=lambda row: row['batch_row'])]
            assert len(rows) == 4
            assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
            assert (self.config.model.lora_rank, self.config.model.lora_alpha) == (8, 16)
            root = Path(output)
            log = (root/f'rank{self.rank}-phases.jsonl').open('a', buffering=1)
            def emit(phase, **extra):
                log.write(json.dumps(dict(phase=phase, unix=time.time(), rank=self.rank,
                    allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    device_free=torch.cuda.mem_get_info()[0],
                    pss_bytes=psutil.Process().memory_full_info().pss, **extra))+'\n')
            producer = None
            trace_owner = reward_readout.trace_token_attribution
            try:
                if self._is_offload_param:
                    load_fsdp_model_to_gpu(self.actor_module_fsdp)
                producer = DeltaTraceRolloutProducer(self.actor_module_fsdp,
                    eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)
                assert producer.direct_readout is not None
                runner_path = Path(inspect.getsourcefile(type(producer.runner)))
                emit('producer_ready', runner_path=str(runner_path), runner_sha256=sha(runner_path),
                     existing_offload=producer.runner.offload_replay_mixer,
                     official_rollout_present=self._is_rollout)
                replay_owner = producer.runner.model.replay_finite_layer
                release_owner = producer.runner.model.release_finite_layer
                phase_name = ['unset']
                traces = []
                details = []
                def trace(*args, **kwargs):
                    if phase_name[0] == 'single_source_eos':
                        # One diagnostic intervention at the existing DT input
                        # boundary. Original target objects, prefix leases and
                        # all finite operators remain owned by the runner.
                        runner, reference, factual, *rest = args
                        reference = factual.clone()
                        reference[0, CASES['appworld']['packed_slot']] = self.tokenizer.eos_token_id
                        args = (runner, reference, factual, *rest)
                    result = trace_owner(*args, **kwargs)
                    traces.append(result[0].detach().cpu())
                    details.append(result[2])
                    return result
                reward_readout.trace_token_attribution = trace
                def replay(_self, layer, callback):
                    index = next(i for i, item in enumerate(producer.runner.model.model.language_model.layers) if item is layer)
                    cache = inspect.getclosurevars(callback).nonlocals.get('kw', {}).get('past_key_values')
                    cache_reference[0] = None if cache is None else weakref.ref(cache)
                    del cache
                    def before_mlp(_module, _args):
                        emit('before_native_mlp', mode=phase_name[0], layer=index, owners=closure_owners(callback))
                    handle = layer.mlp.register_forward_pre_hook(before_mlp)
                    emit('before_native_layer', mode=phase_name[0], layer=index, owners=closure_owners(callback))
                    try:
                        return replay_owner(layer, callback)
                    finally:
                        handle.remove()
                def release(_self, layer):
                    result = release_owner(layer)
                    index = next(i for i, item in enumerate(producer.runner.model.model.language_model.layers) if item is layer)
                    cache = None if cache_reference[0] is None else cache_reference[0]()
                    emit('after_finite_layer', mode=phase_name[0], layer=index,
                         owners={'replay_cache': cache_metadata(cache)})
                    return result
                cache_reference = [None]
                producer.runner.model.replay_finite_layer = MethodType(replay, producer.runner.model)
                producer.runner.model.release_finite_layer = MethodType(release, producer.runner.model)
                outputs = []
                for enabled in (False, True):
                    phase_name[0] = 'original_gpu_captures' if not enabled else 'existing_offload_mixer'
                    producer.runner.offload_replay_mixer = enabled
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    emit('DT_begin', mode=phase_name[0], effective_offload=producer.runner.offload_replay_mixer)
                    result = producer.attribute_episodes([rows], [0.0])[0]
                    torch.cuda.synchronize()
                    outputs.append(result)
                    torch.save(dict(signed=traces[-1], values=result, report=producer.direct_readout.last_report),
                               root/f'rank{self.rank}-{phase_name[0]}.pt')
                    emit('DT_complete', mode=phase_name[0], seconds=time.perf_counter()-started,
                         peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
                # The owner Torch dtype-default check is used on unchanged
                # captured values, not presented as whole-DT FA certification.
                torch.testing.assert_close(traces[0], traces[1])
                for before, after in zip(outputs[0], outputs[1]):
                    for name in before:
                        torch.testing.assert_close(before[name], after[name])
                comparison = dict(signed_equal=torch.equal(traces[0], traces[1]),
                    signed_maxabs=float((traces[0]-traces[1]).abs().max()),
                    assertion='torch.testing.assert_close original dtype defaults; no local thresholds',
                    scope='Same actual B4 and unchanged math; existing offload scheduling only; not whole-DT FA/FLA certification')
                (root/f'rank{self.rank}-comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
                phase_name[0] = 'single_source_eos'
                emit('DT_begin', mode=phase_name[0], source_row=0,
                     packed_slot=CASES['appworld']['packed_slot'])
                started = time.perf_counter()
                result = producer.attribute_episodes([rows], [0.0])[0]
                torch.cuda.synchronize()
                torch.save(dict(signed=traces[-1], detail=details[-1],
                    values=result, report=producer.direct_readout.last_report,
                    scope='One-source EOS diagnostic through the unchanged formal producer and runner; not a production credit replacement'),
                    root/f'rank{self.rank}-single-source-eos.pt')
                emit('DT_complete', mode=phase_name[0], seconds=time.perf_counter()-started,
                     candidate_signed=float(traces[-1][0, CASES['appworld']['packed_slot']]))
                if failed_batch:
                    extra = torch.load(Path(failed_batch)/f'rank{self.rank}-failed-batch.pt', map_location='cpu', weights_only=False)
                    assert len(extra['rows']) == 4
                    phase_name[0] = 'failed_actual_B4_existing_offload'
                    emit('DT_begin', mode=phase_name[0], input_receipt=extra['provenance'])
                    started = time.perf_counter()
                    result = producer.attribute_episodes([extra['rows']], [0.0])[0]
                    torch.cuda.synchronize()
                    torch.save(dict(signed=traces[-1], values=result, report=producer.direct_readout.last_report),
                               root/f'rank{self.rank}-failed-actual-complete.pt')
                    emit('DT_complete', mode=phase_name[0], seconds=time.perf_counter()-started)
                emit('complete', optimizer_steps=0)
                return dict(rank=self.rank, completed=True, optimizer_steps=0)
            except BaseException:
                import traceback
                emit('failed', traceback=traceback.format_exc())
                raise
            finally:
                reward_readout.trace_token_attribution = trace_owner
                if producer is not None:
                    producer.runner.model.release_owner_params()
                if self._is_offload_param:
                    offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log.close()
    return MemoryWorker


def main():
    import ray
    from omegaconf import OmegaConf
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--failed-batch')
    parser.add_argument('--with-vllm', action='store_true',
        help='Use the original actor_rollout role and native rollout context sleep boundary')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.source.read_bytes())
    check_imports(source)
    cfg = OmegaConf.load(Path(source['verl_root'])/'verl/trainer/config/ppo_trainer.yaml')
    for key, value in source['startup_options'].items():
        OmegaConf.update(cfg, key.lstrip('+'), value, force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps = cfg.trainer.total_training_steps
    stopped = threading.Event()
    def sample():
        with (args.output/'physical-mx-smi.jsonl').open('a', buffering=1) as stream:
            while not stopped.is_set():
                result = subprocess.run(['mx-smi'], capture_output=True, timeout=20)
                stream.write(json.dumps(dict(unix=time.time(), returncode=result.returncode,
                    stdout=result.stdout.decode(errors='replace'), stderr=result.stderr.decode(errors='replace')))+'\n')
                stopped.wait(2)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    ray.init(num_cpus=8, include_dashboard=False)
    try:
        pool = RayResourcePool([2], use_gpu=True, max_colocate_count=1)
        actor = RayClassWithInitArgs(make_worker(args.with_vllm), cfg.actor_rollout_ref,
                                    'actor_rollout' if args.with_vllm else 'actor')
        if args.with_vllm:
            # Use the owning trainer's colocation/spawn interface. The async
            # executor reserves WorkerDict names; no alternate actor registry.
            from verl.single_controller.ray.base import create_colocated_worker_cls
            owner_group = RayWorkerGroup(resource_pool=pool,
                ray_cls_with_init=create_colocated_worker_cls(class_dict={'actor_rollout': actor}),
                device_name=cfg.trainer.device)
            group = owner_group.spawn(prefix_set={'actor_rollout'})['actor_rollout']
        else:
            group = RayWorkerGroup(pool, actor)
        group.init_model()
        if args.with_vllm:
            # Same async actor and server manager constructed by the pinned
            # RayPPOTrainer.init_workers. The synchronous context does not own
            # this task's deferred engine. No request/sampling config change.
            from verl.workers.rollout.async_server import AsyncLLMServerManager
            assert cfg.actor_rollout_ref.rollout.mode == 'async'
            async_manager = AsyncLLMServerManager(config=cfg.actor_rollout_ref, worker_group=group)
            async_manager.wake_up()
            async_manager.sleep()
        result = group.profile_existing(str(args.source), str(args.native), str(args.output), args.failed_batch)
        (args.output/'completed.json').write_text(json.dumps(result, indent=2)+'\n')
    finally:
        stopped.set()
        sampler.join(timeout=25)
        ray.shutdown()


if __name__ == '__main__':
    main()
