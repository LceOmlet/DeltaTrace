"""Observe label sensitivity with the original native outcome reader.

For each saved literal first-response prefix, B4 contains factual/full-EOS
under the original legend and factual/full-EOS under its label bijection.
Only RewardAlphabet.labels changes; values, meanings, budget, sampling and
trajectory IDs stay original. No collection, DT, backward or update occurs.
"""
from contextlib import nullcontext
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import psutil
import ray
import torch
from omegaconf import OmegaConf
from reward_readout import RewardAlphabet
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker

OUT = Path(os.environ['DT_TEXTCRAFT_LABEL_ROOT'])
INPUT = Path(os.environ['DT_TEXTCRAFT_READOUT_CASES'])
BASE = Path(os.environ['DT_TEXTCRAFT_READOUT_BASE'])
sys.path.insert(0, str(BASE.parent / 'textcraft-native-readout-20261006-v2'))
from verify_textcraft_native_readout import identity, resources, RUNNER_SHA


class ReversedLabels(RewardAlphabet):
    def labels(self):
        return '10'


def queries(tokenizer, alphabet, inputs, case):
    """Call the owner's query encoder, retaining the saved trajectory IDs."""
    switched = ReversedLabels(alphabet.task, alphabet.values, alphabet.meanings,
                              alphabet.invalid_action_penalty_coef)
    arguments = dict(current_step=case['source_step'], max_steps=inputs['max_steps'],
                     sampling=inputs['sampling'])
    original = alphabet.query_ids(tokenizer, **arguments)
    swapped = switched.query_ids(tokenizer, **arguments)
    ids, end = case['selected_input_ids'], case['source_end']
    labels = alphabet.label_ids(tokenizer)
    assert labels == case['outcome_token_ids'] == [15, 16]
    assert switched.label_ids(tokenizer) == labels[::-1]
    assert original == ids[end:-1], 'Original query must match the saved literal IDs'
    assert len(original) == len(swapped)
    changed = [i for i, pair in enumerate(zip(original, swapped)) if pair[0] != pair[1]]
    assert len(changed) == 2
    assert sorted((original[i], swapped[i]) for i in changed) == [(15, 16), (16, 15)]
    index = alphabet.observed_index(case['observed_return'])
    assert index == case['observed_class_index'] and ids[-1] == labels[index]
    return original, swapped, changed, index


@ray.remote
class LabelEncodingWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def inspect_label_encoding(self):
        from deltatrace_rollout import DeltaTraceRolloutProducer
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

        assert self.config.model.lora_rank == 8 and self.config.model.lora_alpha == 16
        assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
        inputs = json.loads(INPUT.read_bytes())
        cases = inputs['rank_cases'][self.rank]
        assert len(cases) == 32
        result = dict(scope=__doc__, rank=self.rank, pid=os.getpid(),
            pid_birth=psutil.Process().create_time(), input_path=str(INPUT),
            input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
            checkpoint=os.environ['DT_TEXTCRAFT_CHECKPOINT'], selection=inputs['selection'],
            sampling=inputs['sampling'], max_steps=inputs['max_steps'],
            optimizer_steps=0, scheduler_steps=0, backward_calls=0, finite_trace_calls=0,
            native_forward_calls=0, cases=[], sources=dict(
                original_worker=identity(ActorRolloutRefWorker),
                worker_constructor=identity(ActorRolloutRefWorker.__init__),
                producer=identity(DeltaTraceRolloutProducer), observer=identity(self.inspect_label_encoding),
                query_encoder=identity(RewardAlphabet.query_ids)))
        path = OUT / f'rank{self.rank}-readout.json'

        def save(phase, **extra):
            result.update(phase=phase, observed_unix=time.time())
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            print('TEXTCRAFT_LABEL_ENCODING ' + json.dumps(dict(phase=phase, rank=self.rank,
                pid=os.getpid(), completed_cases=len(result['cases']), unix=time.time(), **extra)), flush=True)

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
                native_reader=identity(runner.read_outcomes), native_forward=identity(runner.model.forward_root))
            assert result['sources']['runner']['sha256'] == RUNNER_SHA
            baseline = json.loads((BASE / 'prepared-diagnostic.json').read_bytes())
            assert result['sources']['producer']['sha256'] == baseline['sources'][result['sources']['producer']['path']]
            text = runner.model.model.language_model
            previous_attention = text.config._attn_implementation
            text.set_attn_implementation('flash_attention_2')
            precision = nullcontext()
            if producer.native_fla_fp16:
                from accelerated.qwen35.native_fla_precision import native_fla_fp16
                precision = native_fla_fp16(self.actor_module_fsdp)
                result['sources']['native_precision'] = identity(native_fla_fp16)
            alphabet = producer.readout.alphabet
            labels = alphabet.label_ids(self.tokenizer)
            result.update(outcome_token_ids=labels, model_dtype=str(next(self.actor_module_fsdp.parameters()).dtype),
                native_fla_fp16=producer.native_fla_fp16, event_attention_backend='flash_attention_2',
                numerical_runner_options={key: getattr(runner, key, None) for key in
                    ('copy_replay_captures', 'offload_replay_mixer', 'gdn_head_batch_size',
                     'defer_diagnostics', 'pin_replay_host', 'pin_root_host', 'reuse_native_prefix')})
            with torch.no_grad(), precision:
                torch.cuda.reset_peak_memory_stats()
                result['allocation_before'] = resources()
                for index, case in enumerate(cases):
                    original, swapped, changed, target_index = queries(self.tokenizer, alphabet, inputs, case)
                    ids, start, end = case['selected_input_ids'], case['source_start'], case['source_end']
                    assert 0 <= start < end < len(ids) - 1 and len(ids) <= 32768
                    variants = torch.tensor([ids[:-1], ids[:-1], ids[:end] + swapped, ids[:end] + swapped],
                                            device='cuda', dtype=torch.long)
                    variants[1, start:end] = self.tokenizer.eos_token_id
                    variants[3, start:end] = self.tokenizer.eos_token_id
                    save('native_readout_case_begin', case_index=index, input_shape=list(variants.shape))
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    try:
                        values = runner.read_outcomes(variants, labels)
                        torch.cuda.synchronize()
                        values = values.detach().cpu()
                        result['native_forward_calls'] += 1
                        seconds = time.perf_counter() - began
                    finally:
                        runner.model.release_owner_params()
                    assert values.shape == (4, 2)
                    target_indices = [target_index, target_index, 1 - target_index, 1 - target_index]
                    targets = [float(values[row, column]) for row, column in enumerate(target_indices)]
                    result['cases'].append(dict(case_index=index, traj_uid=case['traj_uid'], source_step=case['source_step'],
                        observed_return=case['observed_return'], source_start=start, source_end=end,
                        context_tokens=len(ids), original_query_ids=original, swapped_query_ids=swapped,
                        changed_query_positions=changed, native_input_shape=list(variants.shape),
                        native_logits_dtype=str(values.dtype),
                        variant_names=['original_factual', 'original_full_response_eos', 'swapped_factual', 'swapped_full_response_eos'],
                        success_class_indices=[1, 1, 0, 0], observed_class_indices=target_indices,
                        native_outcome_log_probs=values.tolist(), native_target_log_probs=targets,
                        seconds=seconds, allocation=resources()))
                    del variants, values
                    save('native_readout_case_complete', last_seconds=seconds)
                result['allocation_after'] = resources()
            save('complete_native_readout')
            return dict(rank=self.rank, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cases=len(cases))
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
    cfg = OmegaConf.load(Path(os.environ['VERL_ROOT']) / 'verl/trainer/config/ppo_trainer.yaml')
    launch = json.loads((BASE / 'launch.json').read_bytes())
    for key, value in launch['options'].items():
        OmegaConf.update(cfg, key.lstrip('+'), value, force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps = 330
    (OUT / 'effective-config.yaml').write_text(OmegaConf.to_yaml(cfg))
    if args.inspect_only:
        import cloudpickle
        from verl.utils import hf_tokenizer
        inputs = json.loads(INPUT.read_bytes())
        assert [len(rows) for rows in inputs['rank_cases']] == [32, 32]
        assert len({case['traj_uid'] for rows in inputs['rank_cases'] for case in rows}) == 64
        assert inputs['sampling'] == dict(temperature=launch['options']['actor_rollout_ref.rollout.temperature'],
                                         max_tokens=launch['options']['data.max_response_length'])
        tokenizer = hf_tokenizer(cfg.actor_rollout_ref.model.path,
            trust_remote_code=cfg.actor_rollout_ref.model.get('trust_remote_code', False), local_files_only=True)
        actor = cfg.actor_rollout_ref.actor
        assert cfg.actor_rollout_ref.model.lora_rank == 8 and cfg.actor_rollout_ref.model.lora_alpha == 16
        assert actor.ppo_micro_batch_size_per_gpu == 4
        alphabet = RewardAlphabet.for_task('TextCraft', inputs['max_steps'],
            actor.invalid_action_penalty_coef if actor.get('use_invalid_action_penalty', True) else 0.0)
        encoded = [queries(tokenizer, alphabet, inputs, case)
                   for rows in inputs['rank_cases'] for case in rows]
        worker = cloudpickle.loads(cloudpickle.dumps(LabelEncodingWorker.__ray_metadata__.modified_class))
        observer_identity = identity(worker.inspect_label_encoding)
        assert observer_identity == dict(path=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        assert identity(worker.__init__) == identity(ActorRolloutRefWorker.__init__)
        process = psutil.Process()
        (OUT / 'native-owner-inspection.json').write_text(json.dumps(dict(
            scope='CPU-only original worker/query inspection; tokenizer only, no model constructed.',
            worker=identity(ActorRolloutRefWorker), worker_constructor=identity(worker.__init__),
            model_loader=identity(ActorRolloutRefWorker.init_model), checkpoint_loader=identity(ActorRolloutRefWorker.load_checkpoint),
            native_reader_inputs_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(), serialized_observer=observer_identity,
            query_encoder=identity(RewardAlphabet.query_ids), tokenizer_owner=identity(hf_tokenizer),
            checked_cases=len(encoded), query_lengths=sorted({len(row[0]) for row in encoded}),
            changed_query_positions=[row[2] for row in encoded], original_queries_exact=True,
            swapped_queries_equal_length=True, only_two_label_ids_changed=True,
            sampling=inputs['sampling'], max_steps=inputs['max_steps'], pid=process.pid, pid_birth=process.create_time(),
            rss_bytes=process.memory_info().rss, host_available_bytes=psutil.virtual_memory().available,
            cuda_initialized=torch.cuda.is_initialized(), actor_config=OmegaConf.to_container(cfg.actor_rollout_ref, resolve=True)), indent=2) + '\n')
    else:
        ray.init(num_cpus=8, include_dashboard=False)
        try:
            group = RayWorkerGroup(RayResourcePool([2], use_gpu=True, max_colocate_count=1),
                RayClassWithInitArgs(LabelEncodingWorker, cfg.actor_rollout_ref, 'actor'))
            for phase, call in [('original_model_initialization', group.init_model),
                                ('original_checkpoint_load', lambda: group.load_checkpoint(
                                    local_path=str(checkpoint / 'actor'), del_local_after_load=False))]:
                print('TEXTCRAFT_LABEL_ENCODING_DRIVER ' + json.dumps(dict(phase=phase + '_begin', unix=time.time())), flush=True)
                call()
                print('TEXTCRAFT_LABEL_ENCODING_DRIVER ' + json.dumps(dict(phase=phase + '_complete', unix=time.time())), flush=True)
            result = group.inspect_label_encoding()
            (OUT / 'completed.json').write_text(json.dumps(dict(completed_unix=time.time(), cases=64,
                optimizer_steps=0, scheduler_steps=0, backward_calls=0, finite_trace_calls=0,
                native_forward_calls=64, ranks=result), indent=2) + '\n')
        finally:
            ray.shutdown()
