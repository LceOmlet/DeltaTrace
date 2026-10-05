"""Observe saved original paired roots, stopping before finite propagation.

Seven literal original B4 groups per rank are kept intact.  Each group runs
its full-response EOS reference and the two already sampled single-token EOS
references for first-response slots.  Other partners retain their original
reference.  The original c9 runner owns packing, prefix forward, synchronization,
HF cache transitions and the FP32 categorical score.  This isolated diagnostic
does not compute credits, collect rollouts, run finite propagation or update.
"""
from contextlib import nullcontext
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import ray
import torch
from omegaconf import OmegaConf
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker


OUT = Path(os.environ['DT_TEXTCRAFT_READOUT_ROOT'])
INPUT = Path(os.environ['DT_TEXTCRAFT_READOUT_CASES'])
BASE = Path(os.environ['DT_TEXTCRAFT_READOUT_BASE'])
REFERENCE = BASE.parent / 'textcraft-native-readout-20261006-v2'
# Reuse the already recorded diagnostic metadata helpers.  Make the module
# importable in Ray's child processes, without changing any owner import root.
sys.path.insert(0, str(REFERENCE))
os.environ['PYTHONPATH'] = str(REFERENCE) + os.pathsep + os.environ.get('PYTHONPATH', '')
from verify_textcraft_native_readout import identity, resources


RUNNER_SHA = 'c7fc969f9f521993f2449ea5f364adcb3e0fdac5b01c38c103963639551516c1'
PRODUCER_SHA = '0ad37a17aede30089fd2ac9a609a42689e6a3a68d00b601520db0a5cff8b8e6e'
HELPER_SHA = '1fd8d8981bf82f0992db78cce43251cd2040afb4a3f039d702eb528b8d8f69cb'
PACK_SHA = '2e8390747c02801bdb763e4eac76112c0ac4caf3f513893f8218e7124e560f27'


class _ObservedOriginalRoot(Exception):
    """Control-flow stop after the original score has actually returned."""


def tensor_metadata(value, *, hash_values=False):
    if not isinstance(value, torch.Tensor):
        return value
    result = dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                  stride=list(value.stride()))
    if hash_values:
        result['int64_values_sha256'] = hashlib.sha256(
            value.detach().to(device='cpu', dtype=torch.int64).contiguous().numpy().tobytes()).hexdigest()
    return result


def cache_metadata(cache):
    if cache is None:
        return None
    result = dict(type=type(cache).__name__, module=type(cache).__module__)
    sequence_length = getattr(cache, 'get_seq_length', None)
    if callable(sequence_length):
        result['native_sequence_length'] = int(sequence_length())
    # Metadata only: never retain, copy, rebuild or mutate the cache tensors.
    for name in ('conv_states', 'recurrent_states', 'key_cache', 'value_cache'):
        values = getattr(cache, name, None)
        if isinstance(values, (list, tuple)):
            result[name] = [tensor_metadata(value) for value in values]
    return result


def observe_original_root(runner, reference, selected, cases, labels, *, fixed_cut=None):
    """Invoke the original call boundary and read its one original score call."""
    from deltatrace_credit import trace_token_attribution
    from qwen35_answer_finite import PackedAnswerTargets

    original_globals = runner.attribute.__func__.__globals__
    original_score = original_globals['selected_target_log_probs']
    original_sync = runner.model.synchronize_prefix_start
    missing = object()
    prior_sync_override = runner.model.__dict__.get('synchronize_prefix_start', missing)
    observation = dict(native_forward_calls=[], sync_calls=[], fixed_cut=fixed_cut,
                       original_sequence_length=selected.shape[1], finite_seed_called=False)

    def native_call(_module, args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        assert isinstance(ids, torch.Tensor)
        record = dict(input_ids=tensor_metadata(ids, hash_values=True),
                      role=('native_shared_prefix' if ids.shape[0] == selected.shape[0]
                            else 'native_paired_root'),
                      use_cache=kwargs.get('use_cache'),
                      logits_to_keep=tensor_metadata(kwargs.get('logits_to_keep')),
                      attention_mask=tensor_metadata(kwargs.get('attention_mask')),
                      past_key_values=cache_metadata(kwargs.get('past_key_values')))
        observation['native_forward_calls'].append(record)

    def fixed_native_sync(candidate_cut):
        # This changes only the diagnostic cut selection.  The original owner
        # still executes its numeric MIN collective and all cache operations.
        assert fixed_cut is not None and int(candidate_cut) >= fixed_cut
        actual = int(original_sync(fixed_cut))
        assert actual == fixed_cut
        observation['sync_calls'].append(dict(candidate_cut=int(candidate_cut),
            delegated_original_cut=fixed_cut, original_returned_cut=actual))
        return actual

    def score_then_stop(original_logits, selection, *, outcome_logits=None):
        # Exactly one call to the original function; no alternate log-softmax
        # or score recomputation is used by this observer.
        value = original_score(original_logits, selection, outcome_logits=outcome_logits)
        cut = selected.shape[1] - selection.length
        if fixed_cut is not None:
            assert cut == fixed_cut
        assert value.shape == (2 * len(selection.labels),)
        observation.update(observed_prefix_length=cut,
            original_packed_logits=tensor_metadata(original_logits),
            categorical_logits=tensor_metadata(outcome_logits),
            categorical_logits_values=outcome_logits.detach().cpu().tolist(),
            categorical_log_probability_dtype=str(value.dtype),
            target_log_probs=value.detach().cpu().tolist(),
            selection=dict(batch=selection.batch, length=selection.length,
                positions=selection.positions.detach().cpu().tolist(),
                paired_positions=selection.paired_positions.detach().cpu().tolist(),
                samples=selection.samples.detach().cpu().tolist(),
                labels=selection.labels.detach().cpu().tolist(),
                outcome_token_ids=selection.outcome_token_ids.detach().cpu().tolist()))
        raise _ObservedOriginalRoot()

    handle = runner.model._conditional.register_forward_pre_hook(native_call, with_kwargs=True)
    stopped = False
    original_globals['selected_target_log_probs'] = score_then_stop
    if fixed_cut is not None:
        runner.model.synchronize_prefix_start = fixed_native_sync
    try:
        try:
            trace_token_attribution(runner, reference, selected, cases,
                [[0] for _ in cases], packed_answer_targets=PackedAnswerTargets,
                outcome_token_ids=labels)
        except _ObservedOriginalRoot:
            stopped = True
        assert stopped, 'The diagnostic must stop at the original score before finite_seed.'
    finally:
        original_globals['selected_target_log_probs'] = original_score
        if fixed_cut is not None:
            if prior_sync_override is missing:
                del runner.model.synchronize_prefix_start
            else:
                runner.model.synchronize_prefix_start = prior_sync_override
        handle.remove()
        runner.model.release_owner_params()
    observation.update(stopped_before_finite_seed=stopped,
                       score_function=identity(original_score),
                       synchronization_function=identity(original_sync))
    return observation


@ray.remote
class MatchedLayoutWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def inspect_matched_layout(self):
        from deltatrace_rollout import DeltaTraceRolloutProducer
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

        assert self.config.model.lora_rank == 8 and self.config.model.lora_alpha == 16
        assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
        raw = INPUT.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == PACK_SHA
        inputs = json.loads(raw)
        groups = inputs['rank_groups'][self.rank]
        assert len(groups) == 7 and [group['original_owner_batch_index'] for group in groups] == list(range(7))
        assert inputs['outcome_token_ids'] == [15, 16]
        assert inputs['eos_token_id'] == self.tokenizer.eos_token_id
        result = dict(scope=__doc__, rank=self.rank, pid=os.getpid(), input_path=str(INPUT),
            input_sha256=PACK_SHA, checkpoint=os.environ['DT_TEXTCRAFT_CHECKPOINT'],
            optimizer_steps=0, scheduler_steps=0, backward_calls=0, finite_trace_calls=0,
            finite_seed_calls=0, paired_root_entries=0, native_forward_calls=0,
            native_root_calls=0, native_prefix_calls=0, groups=[], cases=[],
            sources=dict(original_worker=identity(ActorRolloutRefWorker),
                         producer=identity(DeltaTraceRolloutProducer),
                         observer=identity(self.inspect_matched_layout),
                         metadata_helpers=identity(identity)),
            selection=inputs['coverage'], input_sources=inputs['sources'],
            timing_scope='Root observation timers only; intentionally stopped calls are not complete DT timings.')
        assert result['sources']['producer']['sha256'] == PRODUCER_SHA
        assert result['sources']['metadata_helpers']['sha256'] == HELPER_SHA
        path = OUT / f'rank{self.rank}-readout.json'

        def save(phase, **extra):
            result.update(phase=phase, observed_unix=time.time())
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            print('TEXTCRAFT_MATCHED_LAYOUT ' + json.dumps(dict(phase=phase,
                rank=self.rank, pid=os.getpid(), completed_groups=len(result['groups']),
                completed_cases=len(result['cases']),
                paired_root_entries=result['paired_root_entries'], unix=time.time(), **extra)), flush=True)

        producer = runner = text = None
        previous_attention = None
        training = self.actor_module_fsdp.training
        save('original_worker_ready')
        try:
            if self._is_offload_param:
                load_fsdp_model_to_gpu(self.actor_module_fsdp)
            self.actor_module_fsdp.eval()
            producer = DeltaTraceRolloutProducer(self.actor_module_fsdp,
                eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
                invalid_action_penalty_coef=(self.config.actor.invalid_action_penalty_coef
                    if self.config.actor.get('use_invalid_action_penalty', True) else 0.0))
            runner = producer.runner
            result['sources'].update(runner=identity(type(runner)),
                original_attribute=identity(runner.attribute),
                native_forward=identity(runner.model.forward_root),
                packing=identity(producer.packed_answer_targets))
            assert result['sources']['runner']['sha256'] == RUNNER_SHA
            assert result['sources']['native_forward']['sha256'] == PRODUCER_SHA
            assert runner.reuse_native_prefix
            text = runner.model.model.language_model
            previous_attention = text.config._attn_implementation
            text.set_attn_implementation('flash_attention_2')
            if producer.native_fla_fp16:
                from accelerated.qwen35.native_fla_precision import native_fla_fp16
                precision = native_fla_fp16(self.actor_module_fsdp)
                result['sources']['native_precision'] = identity(native_fla_fp16)
            else:
                precision = nullcontext()
            labels = producer.readout.alphabet.label_ids(self.tokenizer)
            assert labels == inputs['outcome_token_ids']
            result.update(outcome_token_ids=labels, model_dtype=str(next(self.actor_module_fsdp.parameters()).dtype),
                native_fla_fp16=producer.native_fla_fp16, event_attention_backend='flash_attention_2',
                numerical_runner_options={key: getattr(runner, key, None) for key in
                    ('copy_replay_captures', 'offload_replay_mixer', 'gdn_head_batch_size',
                     'defer_diagnostics', 'pin_replay_host', 'pin_root_host', 'reuse_native_prefix')})
            with torch.no_grad(), precision:
                torch.cuda.reset_peak_memory_stats()
                result['allocation_before'] = resources()
                for index, group in enumerate(groups):
                    assert group['rank'] == self.rank and len(group['samples']) == 4
                    selected = torch.tensor(group['selected_input_ids'], device='cuda', dtype=torch.long)
                    original_reference = torch.tensor(group['reference_input_ids'], device='cuda', dtype=torch.long)
                    assert selected.shape == original_reference.shape and selected.shape[0] == 4
                    assert selected.shape[1] <= 32768
                    assert tensor_metadata(selected, hash_values=True)['int64_values_sha256'] == group['selected_input_ids_sha256']
                    assert tensor_metadata(original_reference, hash_values=True)['int64_values_sha256'] == group['reference_input_ids_sha256']
                    cases = [dict(target_ids=torch.tensor(sample['native_target_case']['target_ids'], dtype=torch.long),
                                  prompt_length=sample['native_target_case']['prompt_length'])
                             for sample in group['samples']]
                    record = dict(original_owner_batch_index=index, samples=group['samples'],
                        original_paired_owner_abi=group['paired_owner_abi'], variant_case_indices=[])
                    original_cut = None
                    for variant, probe in [('full_response_eos', None), ('single_eos_0', 0), ('single_eos_1', 1)]:
                        reference = original_reference.clone()
                        changed_slots = []
                        if probe is not None:
                            for slot, sample in enumerate(group['samples']):
                                if sample['source_step'] != 0:
                                    continue
                                position = sample['probe_input_positions'][probe]
                                assert sample['source_start'] <= position < sample['source_end']
                                reference[slot].copy_(selected[slot])
                                reference[slot, position] = self.tokenizer.eos_token_id
                                changed_slots.append(slot)
                        # No rank-local skip: even a group without first-response
                        # slots keeps the original owner's collective schedule.
                        save('original_paired_root_begin', original_owner_batch_index=index,
                             variant=variant, changed_first_response_slots=changed_slots)
                        torch.cuda.synchronize()
                        began = time.perf_counter()
                        observation = observe_original_root(runner, reference, selected, cases, labels,
                            fixed_cut=original_cut if probe is not None else None)
                        torch.cuda.synchronize()
                        observation.update(variant=variant, original_owner_batch_index=index,
                            samples=group['samples'], root_observation_seconds=time.perf_counter()-began,
                            changed_first_response_slots=changed_slots, allocation=resources())
                        if probe is None:
                            original_cut = observation['observed_prefix_length']
                        result['paired_root_entries'] += 1
                        result['native_forward_calls'] += len(observation['native_forward_calls'])
                        result['native_root_calls'] += 1
                        result['native_prefix_calls'] += sum(call['role'] == 'native_shared_prefix'
                            for call in observation['native_forward_calls'])
                        record['variant_case_indices'].append(len(result['cases']))
                        result['cases'].append(observation)
                        del reference
                        save('original_paired_root_complete', original_owner_batch_index=index, variant=variant)
                    result['groups'].append(record)
                    del selected, original_reference, cases
                    save('matched_group_complete', original_owner_batch_index=index)
                result['allocation_after'] = resources()
            assert result['paired_root_entries'] == 21
            save('complete_native_readout')
            return dict(rank=self.rank, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        groups=7, cases=21, paired_root_entries=21, finite_seed_calls=0, finite_trace_calls=0)
        finally:
            if text is not None and previous_attention is not None:
                text.set_attn_implementation(previous_attention)
            self.actor_module_fsdp.train(training)
            if runner is not None:
                runner.model.release_owner_params()
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-only', action='store_true')
    args = parser.parse_args()
    checkpoint = Path(os.environ['DT_TEXTCRAFT_CHECKPOINT'])
    assert checkpoint.name == 'global_step_25'
    assert hashlib.sha256(INPUT.read_bytes()).hexdigest() == PACK_SHA
    assert identity(identity)['sha256'] == HELPER_SHA
    cfg = OmegaConf.load(Path(os.environ['VERL_ROOT']) / 'verl/trainer/config/ppo_trainer.yaml')
    launch = json.loads((BASE / 'launch.json').read_bytes())
    for key, value in launch['options'].items():
        OmegaConf.update(cfg, key.lstrip('+'), value, force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps = 330
    (OUT / 'effective-config.yaml').write_text(OmegaConf.to_yaml(cfg))
    if args.inspect_only:
        import cloudpickle
        method = MatchedLayoutWorker.__ray_metadata__.modified_class.inspect_matched_layout
        serialized = cloudpickle.loads(cloudpickle.dumps(method))
        observer_identity = identity(serialized)
        assert observer_identity == dict(path=str(Path(__file__).resolve()),
            sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (OUT / 'native-owner-inspection.json').write_text(json.dumps(dict(
            scope='CPU-only original-owner/config/cloudpickle metadata inspection; no model constructed.',
            worker=identity(ActorRolloutRefWorker), worker_constructor=identity(ActorRolloutRefWorker.__init__),
            model_loader=identity(ActorRolloutRefWorker.init_model),
            checkpoint_loader=identity(ActorRolloutRefWorker.load_checkpoint),
            native_reader_inputs_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
            serialized_observer=observer_identity, metadata_helpers=identity(identity),
            cuda_initialized=torch.cuda.is_initialized(),
            actor_config=OmegaConf.to_container(cfg.actor_rollout_ref, resolve=True)), indent=2) + '\n')
    else:
        ray.init(num_cpus=8, include_dashboard=False)
        try:
            group = RayWorkerGroup(RayResourcePool([2], use_gpu=True, max_colocate_count=1),
                RayClassWithInitArgs(MatchedLayoutWorker, cfg.actor_rollout_ref, 'actor'))
            print('TEXTCRAFT_MATCHED_LAYOUT_DRIVER ' + json.dumps(dict(
                phase='original_model_initialization_begin', unix=time.time())), flush=True)
            group.init_model()
            print('TEXTCRAFT_MATCHED_LAYOUT_DRIVER ' + json.dumps(dict(
                phase='original_checkpoint_load_begin', unix=time.time())), flush=True)
            group.load_checkpoint(local_path=str(checkpoint / 'actor'), del_local_after_load=False)
            print('TEXTCRAFT_MATCHED_LAYOUT_DRIVER ' + json.dumps(dict(
                phase='original_checkpoint_load_complete', unix=time.time())), flush=True)
            ranks = group.inspect_matched_layout()
            (OUT / 'completed.json').write_text(json.dumps(dict(completed_unix=time.time(),
                groups=14, cases=42, paired_root_entries=42, optimizer_steps=0, scheduler_steps=0,
                backward_calls=0, finite_trace_calls=0, finite_seed_calls=0, ranks=ranks), indent=2) + '\n')
        finally:
            ray.shutdown()
