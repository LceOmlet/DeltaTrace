"""Bounded DT storage check through the original VERL async actor.

The saved failed B4 remains a real regression case.  A separate exact-32768
storage stress case extends only an existing observation interval.  Its reward
is a capacity coefficient, not a new environment result or accuracy test.
No rollout, optimizer, restore, credit rule, model forward or scorer is copied.
"""
import argparse
import inspect
import json
from pathlib import Path
import time
from types import MethodType
import weakref

import torch

import profile_existing_offload as profile_owner
from inspect_extreme_endpoint import check_imports, sha, CASES


def capacity_row(row, prepare, length=32768):
    original = prepare(row, 0)
    delta = length - original['selected'].numel()
    assert delta > 0
    prompt = original['prompt_length']
    suffix = original['selected'][prompt:]
    policy, target = original['valid_policy'], original['valid_target']
    assert target.any() and policy.any()
    # Use an actual observation run after a policy action and before the last
    # executed target. No token is decoded or replaced with an invented label.
    start = int(policy.nonzero()[0]) + 1
    end = int(target.nonzero()[-1])
    runs = []
    i = start
    while i < end:
        if policy[i]:
            i += 1
            continue
        j = i + 1
        while j < end and not policy[j]:
            j += 1
        runs.append((j-i, i, j))
        i = j
    count, begin, insert = max(runs)
    assert count > 0
    added = suffix[begin:insert].repeat((delta+count-1)//count)[:delta]
    expanded = torch.cat((suffix[:insert], added, suffix[insert:]))
    expanded_policy = torch.cat((policy[:insert], torch.zeros(delta, dtype=torch.bool), policy[insert:]))
    expanded_target = torch.cat((target[:insert], torch.zeros(delta, dtype=torch.bool), target[insert:]))
    result = dict(row, input_ids=torch.cat((original['selected'][:prompt], expanded)),
                  attention_mask=torch.ones(length, dtype=row['attention_mask'].dtype),
                  responses=expanded, policy_mask=expanded_policy, target_mask=expanded_target)
    checked = prepare(result, 0)
    assert checked['selected'].numel() == checked['full_context_length'] == length
    assert torch.equal(expanded[expanded_policy], suffix[policy])
    assert torch.equal(expanded[expanded_target], suffix[target])
    assert torch.equal(expanded[:insert], suffix[:insert])
    assert torch.equal(expanded[insert+delta:], suffix[insert:])
    return result, dict(original_uid=str(row.get('traj_uid')), original_causal_length=original['selected'].numel(),
        causal_length=length, original_observation_interval=[begin, insert], added_observation_tokens=delta,
        original_policy_tokens=int(policy.sum()), original_target_tokens=int(target.sum()),
        scope='Synthetic storage capacity only: original causal IDs retained in order; repeated native observation IDs masked out; not task, attribution-accuracy or PPO-update validation')


def make_worker(with_vllm=False):
    assert with_vllm
    import ray
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

    @ray.remote
    class CapacityWorker(AsyncActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def profile_existing(self, source_path, native_path, output, failed_batch):
            import psutil
            import reward_readout
            from deltatrace_rollout import DeltaTraceRolloutProducer
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

            source = json.loads(Path(source_path).read_bytes())
            check_imports(source)
            assert sha(source_path) == CASES['appworld']['source_sha256']
            assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
            assert (self.config.model.lora_rank, self.config.model.lora_alpha) == (8, 16)
            saved = torch.load(Path(failed_batch)/f'rank{self.rank}-failed-batch.pt', map_location='cpu', weights_only=False)
            assert len(saved['rows']) == 4
            prepared = [capacity_row(row, reward_readout.DirectActionTargetReadout._prepare_row) for row in saved['rows']]
            capacity = [item[0] for item in prepared]
            root = Path(output)
            log = (root/f'rank{self.rank}-phases.jsonl').open('a', buffering=1)
            mode = ['initializing']
            def emit(phase, **extra):
                torch.cuda.synchronize()
                log.write(json.dumps(dict(phase=phase, mode=mode[0], unix=time.time(), rank=self.rank,
                    allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated(), device_free=torch.cuda.mem_get_info()[0],
                    pss_bytes=psutil.Process().memory_full_info().pss, **extra))+'\n')
            producer = None
            try:
                if self._is_offload_param:
                    load_fsdp_model_to_gpu(self.actor_module_fsdp)
                producer = DeltaTraceRolloutProducer(self.actor_module_fsdp,
                    eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)
                runner = producer.runner
                assert runner.offload_replay_mixer
                emit('producer_ready', runner=inspect.getsourcefile(type(runner)),
                     runner_sha256=sha(inspect.getsourcefile(type(runner))), capacity=[item[1] for item in prepared],
                     official_rollout_present=self._is_rollout, offload=True)
                replay_owner = runner.model.replay_finite_layer
                release_owner = runner.model.release_finite_layer
                cache_ref = [None]
                def replay(_self, layer, callback):
                    index = next(i for i, value in enumerate(runner.model.model.language_model.layers) if value is layer)
                    cache = inspect.getclosurevars(callback).nonlocals.get('kw', {}).get('past_key_values')
                    cache_ref[0] = None if cache is None else weakref.ref(cache)
                    del cache
                    def before_mlp(_module, _args):
                        emit('before_native_mlp', layer=index, owners=profile_owner.closure_owners(callback))
                    handle = layer.mlp.register_forward_pre_hook(before_mlp)
                    emit('before_native_layer', layer=index, owners=profile_owner.closure_owners(callback))
                    try:
                        return replay_owner(layer, callback)
                    finally:
                        handle.remove()
                def release(_self, layer):
                    value = release_owner(layer)
                    index = next(i for i, item in enumerate(runner.model.model.language_model.layers) if item is layer)
                    cache = None if cache_ref[0] is None else cache_ref[0]()
                    emit('after_finite_layer', layer=index, replay_cache=profile_owner.cache_metadata(cache))
                    return value
                runner.model.replay_finite_layer = MethodType(replay, runner.model)
                runner.model.release_finite_layer = MethodType(release, runner.model)
                previous = None
                for name, rows in [('exact32768_first', capacity), ('exact32768_repeat', capacity), ('original_failed_B4', saved['rows'])]:
                    mode[0] = name
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    emit('DT_begin')
                    values = producer.attribute_episodes([rows], [0.0])[0]
                    emit('DT_complete', seconds=time.perf_counter()-started, report=producer.direct_readout.last_report)
                    assert all(torch.isfinite(t).all() for row in values for t in row.values())
                    if previous is not None and name == 'exact32768_repeat':
                        for old, new in zip(previous, values):
                            for key in old:
                                torch.testing.assert_close(old[key], new[key])
                        emit('repeat_comparison', QVA_exact_equal={key:all(torch.equal(old[key],new[key]) for old,new in zip(previous,values)) for key in values[0]},
                            assertion='Original torch.testing.assert_close dtype defaults for unchanged repeated outputs; not an FA/FLA finite-attribution accuracy threshold')
                    if name == 'exact32768_first':
                        previous = values
                    if name == 'original_failed_B4':
                        earlier = torch.load(Path(native_path).parent/f'rank{self.rank}-failed-actual-complete.pt', map_location='cpu', weights_only=False)['values']
                        for old, new in zip(earlier, values):
                            for key in old:
                                torch.testing.assert_close(old[key], new[key])
                        emit('real_regression_comparison', QVA_exact_equal={key:all(torch.equal(old[key],new[key]) for old,new in zip(earlier,values)) for key in values[0]},
                             comparison='Earlier completed actual failed B4 with same candidate and original async-vLLM lifecycle; no invented threshold')
                    torch.save(dict(values=values, report=producer.direct_readout.last_report), root/f'rank{self.rank}-{name}.pt')
                    emit('after_saved_outputs')
                emit('complete', optimizer_steps=0)
                return dict(rank=self.rank, completed=True, optimizer_steps=0)
            except BaseException:
                import traceback
                emit('failed', traceback=traceback.format_exc())
                raise
            finally:
                if producer is not None:
                    producer.runner.model.release_owner_params()
                if self._is_offload_param:
                    offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log.close()
    return CapacityWorker


if __name__ == '__main__':
    # The preceding diagnostic already owns original VERL pool/async-server
    # initialization and physical mx-smi sampling. Reuse it unchanged.
    profile_owner.make_worker = make_worker
    profile_owner.main()
