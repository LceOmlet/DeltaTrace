"""Isolate current-response clock wording with the original native reader.

The original environment executes the response as emitted, including a response
ended by the generation limit. The historical forecast instead asks to complete
that same response. This diagnostic changes those two clauses only. Original
IDs, labels, target, EOS intervention, sampling, return and owners are retained.
Equal-length real cases are grouped without padding or duplicated rows. This is
not a production query patch, attribution quality gate or task-gradient test.
"""
from contextlib import nullcontext
import argparse
from collections import defaultdict
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

OUT = Path(os.environ['DT_TEXTCRAFT_READOUT_ROOT'])
INPUT = Path(os.environ['DT_TEXTCRAFT_READOUT_CASES'])
BASE = Path(os.environ['DT_TEXTCRAFT_READOUT_BASE'])
sys.path.insert(0, str(BASE.parent / 'textcraft-native-readout-20261006-v2'))
from verify_textcraft_native_readout import identity, resources, RUNNER_SHA

REPLACEMENTS = (
    ('possibly ending inside an unfinished response. Do not treat this forecast request ',
     'after generation of the current response has ended, possibly at its token limit. Do not treat this forecast request '),
    ('as an environment action. Imagine completing that response and continuing with ',
     'as an environment action. Imagine the environment immediately processing that response as emitted and then continuing with '),
)


class ClockQueryTokenizer:
    """Transform only the owner's new forecast text before original encoding."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.original_text = self.corrected_text = None

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def encode(self, text, **kwargs):
        if text.startswith('<|im_end|>\n<|im_start|>user\nFuture cumulative return forecast'):
            assert self.original_text is None and kwargs == dict(add_special_tokens=False)
            self.original_text = text
            for old, new in REPLACEMENTS:
                assert text.count(old) == 1
                text = text.replace(old, new)
            self.corrected_text = text
        return self.tokenizer.encode(text, **kwargs)


def queries(tokenizer, alphabet, inputs, case):
    arguments = dict(current_step=case['source_step'], max_steps=inputs['max_steps'], sampling=inputs['sampling'])
    original = alphabet.query_ids(tokenizer, **arguments)
    proxy = ClockQueryTokenizer(tokenizer)
    corrected = alphabet.query_ids(proxy, **arguments)
    assert proxy.original_text is not None
    restored = proxy.corrected_text
    for old, new in REPLACEMENTS:
        assert restored.count(new) == 1
        restored = restored.replace(new, old)
    assert restored == proxy.original_text
    ids, end = case['selected_input_ids'], case['source_end']
    labels = alphabet.label_ids(tokenizer)
    assert original == ids[end:-1] and labels == case['outcome_token_ids'] == [15, 16]
    assert ids[end - 1] == tokenizer.eos_token_id
    index = alphabet.observed_index(case['observed_return'])
    assert index == case['observed_class_index'] and ids[-1] == labels[index]
    return dict(original_ids=original, corrected_ids=corrected,
        original_text=proxy.original_text, corrected_text=proxy.corrected_text, observed_class_index=index)


@ray.remote
class ResponseClockWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def inspect_response_clock(self):
        from deltatrace_rollout import DeltaTraceRolloutProducer
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

        assert self.config.model.lora_rank == 8 and self.config.model.lora_alpha == 16
        assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
        inputs = json.loads(INPUT.read_bytes())
        cases = inputs['rank_cases'][self.rank]
        assert len(cases) == 32
        result = dict(scope=__doc__, rank=self.rank, pid=os.getpid(), pid_birth=psutil.Process().create_time(),
            input_path=str(INPUT), input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
            checkpoint=os.environ['DT_TEXTCRAFT_CHECKPOINT'], selection=inputs['selection'],
            sampling=inputs['sampling'], max_steps=inputs['max_steps'], replacements=REPLACEMENTS,
            optimizer_steps=0, scheduler_steps=0, backward_calls=0, finite_trace_calls=0,
            native_forward_calls=0, cases=[], calls=[], sources=dict(
                original_worker=identity(ActorRolloutRefWorker), producer=identity(DeltaTraceRolloutProducer),
                observer=identity(self.inspect_response_clock), query_encoder=identity(RewardAlphabet.query_ids),
                query_text_adapter=identity(ClockQueryTokenizer)))
        path = OUT / f'rank{self.rank}-readout.json'

        def save(phase, **extra):
            result.update(phase=phase, observed_unix=time.time())
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            print('TEXTCRAFT_RESPONSE_CLOCK ' + json.dumps(dict(phase=phase, rank=self.rank,
                pid=os.getpid(), completed_cases=len(result['cases']), unix=time.time(), **extra)), flush=True)

        runner = text = None
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
            result['sources'].update(runner=identity(type(runner)), native_reader=identity(runner.read_outcomes),
                                    native_forward=identity(runner.model.forward_root))
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
                native_fla_fp16=producer.native_fla_fp16, event_attention_backend='flash_attention_2')
            encoded = [queries(self.tokenizer, alphabet, inputs, case) for case in cases]
            # The owner scores the last predictor. Never right-pad unequal
            # sequences: bucket real cases by their two exact native lengths.
            buckets = defaultdict(list)
            for index, (case, query) in enumerate(zip(cases, encoded)):
                buckets[(len(case['selected_input_ids']) - 1,
                         case['source_end'] + len(query['corrected_ids']))].append(index)
            with torch.no_grad(), precision:
                torch.cuda.reset_peak_memory_stats()
                result['allocation_before'] = resources()
                for key, indices in buckets.items():
                    for offset in range(0, len(indices), 2):
                        group = indices[offset:offset + 2]
                        outputs = {}
                        for mode in ('original', 'corrected'):
                            rows = []
                            for index in group:
                                case, query = cases[index], encoded[index]
                                prefix = case['selected_input_ids'][:case['source_end']]
                                row = prefix + query[mode + '_ids']
                                reference = row.copy()
                                reference[case['source_start']:case['source_end']] = [self.tokenizer.eos_token_id] * (case['source_end']-case['source_start'])
                                assert len(row) == (key[0] if mode == 'original' else key[1]) <= 32768
                                rows.extend((row, reference))
                            variants = torch.tensor(rows, device='cuda', dtype=torch.long)
                            save('native_readout_group_begin', case_indices=group, query_mode=mode, input_shape=list(variants.shape))
                            torch.cuda.synchronize()
                            began = time.perf_counter()
                            try:
                                values = runner.read_outcomes(variants, labels)
                                torch.cuda.synchronize()
                                values = values.detach().cpu()
                                seconds = time.perf_counter() - began
                            finally:
                                runner.model.release_owner_params()
                            assert values.shape == (len(rows), 2)
                            outputs[mode] = values.tolist()
                            result['native_forward_calls'] += 1
                            result['calls'].append(dict(mode=mode, case_indices=group, shape=list(variants.shape),
                                seconds=seconds, categorical_log_prob_dtype=str(values.dtype), allocation=resources()))
                            del variants, values
                        for position, index in enumerate(group):
                            case, query = cases[index], encoded[index]
                            result['cases'].append(dict(case_index=index, traj_uid=case['traj_uid'], source_step=case['source_step'],
                                observed_return=case['observed_return'], source_start=case['source_start'], source_end=case['source_end'],
                                original_context_tokens=key[0]+1, corrected_context_tokens=key[1]+1,
                                query=query, native_outcome_log_probs={mode: outputs[mode][2*position:2*position+2] for mode in outputs}))
                        save('native_readout_group_complete', case_indices=group)
                result['allocation_after'] = resources()
            save('complete_native_readout')
            return dict(rank=self.rank, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cases=len(cases), calls=result['native_forward_calls'])
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
        assert inputs['sampling'] == dict(temperature=launch['options']['actor_rollout_ref.rollout.temperature'], max_tokens=launch['options']['data.max_response_length'])
        tokenizer = hf_tokenizer(cfg.actor_rollout_ref.model.path, trust_remote_code=cfg.actor_rollout_ref.model.get('trust_remote_code', False), local_files_only=True)
        actor = cfg.actor_rollout_ref.actor
        assert cfg.actor_rollout_ref.model.lora_rank == 8 and cfg.actor_rollout_ref.model.lora_alpha == 16
        assert actor.ppo_micro_batch_size_per_gpu == 4
        alphabet = RewardAlphabet.for_task('TextCraft', inputs['max_steps'], actor.invalid_action_penalty_coef if actor.get('use_invalid_action_penalty', True) else 0.0)
        encoded = [queries(tokenizer, alphabet, inputs, case) for rows in inputs['rank_cases'] for case in rows]
        worker = cloudpickle.loads(cloudpickle.dumps(ResponseClockWorker.__ray_metadata__.modified_class))
        observer_identity = identity(worker.inspect_response_clock)
        assert observer_identity == dict(path=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        assert identity(worker.__init__) == identity(ActorRolloutRefWorker.__init__)
        (OUT / 'native-owner-inspection.json').write_text(json.dumps(dict(
            scope='CPU original owner and exact query-only changes; no model constructed.',
            worker=identity(ActorRolloutRefWorker), worker_constructor=identity(worker.__init__),
            native_reader_inputs_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(), serialized_observer=observer_identity,
            query_encoder=identity(RewardAlphabet.query_ids), query_text_adapter=identity(ClockQueryTokenizer),
            checked_cases=len(encoded), original_queries_exact=True, replacements=REPLACEMENTS,
            query_length_pairs=sorted({(len(row['original_ids']), len(row['corrected_ids'])) for row in encoded}),
            queries=encoded, cuda_initialized=torch.cuda.is_initialized(), actor_config=OmegaConf.to_container(cfg.actor_rollout_ref, resolve=True)), indent=2)+'\n')
    else:
        ray.init(num_cpus=8, include_dashboard=False)
        try:
            group = RayWorkerGroup(RayResourcePool([2], use_gpu=True, max_colocate_count=1), RayClassWithInitArgs(ResponseClockWorker, cfg.actor_rollout_ref, 'actor'))
            for phase, call in [('original_model_initialization', group.init_model), ('original_checkpoint_load', lambda: group.load_checkpoint(local_path=str(checkpoint / 'actor'), del_local_after_load=False))]:
                print('TEXTCRAFT_RESPONSE_CLOCK_DRIVER ' + json.dumps(dict(phase=phase+'_begin', unix=time.time())), flush=True)
                call()
                print('TEXTCRAFT_RESPONSE_CLOCK_DRIVER ' + json.dumps(dict(phase=phase+'_complete', unix=time.time())), flush=True)
            ranks = group.inspect_response_clock()
            (OUT / 'completed.json').write_text(json.dumps(dict(completed_unix=time.time(), cases=64,
                optimizer_steps=0, scheduler_steps=0, backward_calls=0, finite_trace_calls=0,
                native_forward_calls=sum(rank['calls'] for rank in ranks), ranks=ranks), indent=2)+'\n')
        finally:
            ray.shutdown()
