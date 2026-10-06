"""Original author cumulative deletion over saved successful first responses.

Uses saved d_from_A, not fresh DT. The author adapter owns deletion ordering,
grouping and metrics. Original VERL owns the sole model/checkpoint; existing
runner.read_outcomes owns categorical scoring. No rollout/DT/backward/update.
"""
from contextlib import nullcontext
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import cloudpickle
import psutil
import ray
import torch
from omegaconf import OmegaConf
from reward_readout import RewardAlphabet
from verl.protocol import DataProto, pad_dataproto_to_divisor
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from author_reward_curve_adapter import inspect_case_contract, run_saved_reward_curves
from verify_textcraft_native_readout import identity, resources, RUNNER_SHA

OUT = Path(os.environ['DT_TEXTCRAFT_AUTHOR_CURVE_ROOT'])
INPUT = Path(os.environ['DT_TEXTCRAFT_READOUT_CASES'])
BASE = Path(os.environ['DT_TEXTCRAFT_READOUT_BASE'])


def saved_partition(inputs):
    """Filter the saved population, then delegate padding/partition to VERL."""
    population = [case for rows in inputs['rank_cases'] for case in rows]
    cases = [case for case in population if case['actual_G'] == 1.0]
    assert len(cases) == len({case['traj_uid'] for case in cases}) == 21
    proto = DataProto.from_dict(non_tensors={'cases': cases})
    padded, padding = pad_dataproto_to_divisor(proto, 2)
    chunks = padded.chunk(2)
    rows = [chunk.non_tensor_batch['cases'].tolist() for chunk in chunks]
    assert padding == 1 and [len(chunk) for chunk in chunks] == [11, 11]
    return rows, padding


@ray.remote
class AuthorRewardCurveWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def inspect_author_reward_curves(self):
        from deltatrace_rollout import DeltaTraceRolloutProducer
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

        assert self.config.model.lora_rank == 8 and self.config.model.lora_alpha == 16
        assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
        inputs = json.loads(INPUT.read_bytes())
        rank_rows, padding = saved_partition(inputs)
        cases = rank_rows[self.rank]
        result = dict(scope=__doc__, rank=self.rank, pid=os.getpid(),
            pid_birth=psutil.Process().create_time(), input_path=str(INPUT),
            input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
            checkpoint=os.environ['DT_TEXTCRAFT_CHECKPOINT'], sampling=inputs['sampling'],
            max_steps=inputs['max_steps'], population_unique=21, padded_slots=22,
            padding=padding, padding_owner=identity(pad_dataproto_to_divisor),
            partition_owner=identity(DataProto.chunk), ordering_source='saved_source_d_from_A',
            optimizer_steps=0, scheduler_steps=0, backward_calls=0, finite_trace_calls=0,
            native_forward_calls=0, cases=[], sources=dict(
                original_worker=identity(ActorRolloutRefWorker),
                worker_constructor=identity(ActorRolloutRefWorker.__init__),
                producer=identity(DeltaTraceRolloutProducer),
                observer=identity(self.inspect_author_reward_curves),
                author_adapter=identity(run_saved_reward_curves)))
        path = OUT / f'rank{self.rank}-curves.json'

        def save(phase, **extra):
            result.update(phase=phase, observed_unix=time.time())
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            print('TEXTCRAFT_AUTHOR_REWARD_CURVE ' + json.dumps(dict(
                phase=phase, rank=self.rank, pid=os.getpid(), completed_cases=len(result['cases']),
                native_forward_calls=result['native_forward_calls'], unix=time.time(), **extra)), flush=True)

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
            labels = producer.readout.alphabet.label_ids(self.tokenizer)
            result.update(outcome_token_ids=labels, model_dtype=str(next(self.actor_module_fsdp.parameters()).dtype),
                native_fla_fp16=producer.native_fla_fp16, event_attention_backend='flash_attention_2',
                numerical_runner_options={key: getattr(runner, key, None) for key in
                    ('copy_replay_captures', 'offload_replay_mixer', 'gdn_head_batch_size',
                     'defer_diagnostics', 'pin_replay_host', 'pin_root_host', 'reuse_native_prefix')})
            with torch.no_grad(), precision:
                torch.cuda.reset_peak_memory_stats()
                result['allocation_before'] = resources()
                for index, case in enumerate(cases):
                    contract = inspect_case_contract(case, self.tokenizer, self.tokenizer.eos_token_id)
                    assert labels == case['outcome_token_ids']
                    target_index = case['observed_class_index']
                    score_seconds = []

                    def score(prompt_ids, response_ids):
                        # Author callback's target is one saved categorical label;
                        # the predictor remains the last original prefix column.
                        assert response_ids.shape == (1, 1)
                        assert int(response_ids[0, 0]) == case['target_id'] == labels[target_index]
                        torch.cuda.synchronize()
                        began = time.perf_counter()
                        try:
                            values = runner.read_outcomes(prompt_ids.to('cuda'), labels)
                            torch.cuda.synchronize()
                            result['native_forward_calls'] += 1
                            score_seconds.append(time.perf_counter() - began)
                            return values[:, target_index:target_index + 1].detach().cpu()
                        finally:
                            runner.model.release_owner_params()

                    def phase(view, phase, **extra):
                        save('author_' + phase, case_index=index, traj_uid=case['traj_uid'],
                             view=view, score_seconds=sum(score_seconds), **extra)

                    began = time.perf_counter()
                    save('author_case_begin', case_index=index, traj_uid=case['traj_uid'])
                    curves = run_saved_reward_curves(case, self.tokenizer, score,
                        eos_token_id=self.tokenizer.eos_token_id, phase_callback=phase)
                    result['cases'].append(dict(case_index=index, traj_uid=case['traj_uid'],
                        source_step=case['source_step'], observed_return=case['observed_return'],
                        source_start=case['source_start'], source_end=case['source_end'],
                        contract=contract, curves=curves, native_scoring_calls=len(score_seconds),
                        native_scoring_seconds=sum(score_seconds), seconds=time.perf_counter() - began,
                        allocation=resources()))
                    save('author_case_complete', last_seconds=result['cases'][-1]['seconds'])
                result['allocation_after'] = resources()
            save('complete_author_reward_curves')
            return dict(rank=self.rank, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        cases=len(cases), native_forward_calls=result['native_forward_calls'])
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
        from verl.utils import hf_tokenizer
        inputs = json.loads(INPUT.read_bytes())
        rows, padding = saved_partition(inputs)
        tokenizer = hf_tokenizer(cfg.actor_rollout_ref.model.path,
            trust_remote_code=cfg.actor_rollout_ref.model.get('trust_remote_code', False), local_files_only=True)
        actor = cfg.actor_rollout_ref.actor
        alphabet = RewardAlphabet.for_task('TextCraft', inputs['max_steps'],
            actor.invalid_action_penalty_coef if actor.get('use_invalid_action_penalty', True) else 0.0)
        assert cfg.actor_rollout_ref.model.lora_rank == 8 and cfg.actor_rollout_ref.model.lora_alpha == 16
        assert actor.ppo_micro_batch_size_per_gpu == 4
        contracts = []
        for rank_cases in rows:
            for case in rank_cases:
                assert alphabet.query_ids(tokenizer, current_step=case['source_step'],
                    max_steps=inputs['max_steps'], sampling=inputs['sampling']) == case['selected_input_ids'][case['source_end']:-1]
                assert alphabet.label_ids(tokenizer) == case['outcome_token_ids'] == [15, 16]
                contracts.append(inspect_case_contract(case, tokenizer, tokenizer.eos_token_id))
        worker = cloudpickle.loads(cloudpickle.dumps(AuthorRewardCurveWorker.__ray_metadata__.modified_class))
        observer = identity(worker.inspect_author_reward_curves)
        assert observer == dict(path=str(Path(__file__).resolve()), sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        assert identity(worker.__init__) == identity(ActorRolloutRefWorker.__init__)
        import author_reward_curve_adapter as adapter
        dependencies = adapter.owner_sources()
        process = psutil.Process()
        (OUT / 'native-owner-inspection.json').write_text(json.dumps(dict(
            scope='CPU-only literal IDs, original author imports, VERL partition and serialized original-worker inspection; tokenizer only, no model/scoring.',
            worker=identity(ActorRolloutRefWorker), worker_constructor=identity(worker.__init__),
            model_loader=identity(ActorRolloutRefWorker.init_model), checkpoint_loader=identity(ActorRolloutRefWorker.load_checkpoint),
            native_reader_inputs_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(), serialized_observer=observer,
            author_adapter=identity(run_saved_reward_curves), author_dependencies=dependencies,
            padding_owner=identity(pad_dataproto_to_divisor), partition_owner=identity(DataProto.chunk),
            checked_cases=len(contracts), unique_cases=21, padding=padding,
            rank_uids=[[case['traj_uid'] for case in rank_cases] for rank_cases in rows], contracts=contracts,
            planned_scoring_calls_per_rank=462, native_batch_per_call=1, config_actor_microbatch=4,
            optimizer_steps=0, scheduler_steps=0, backward_calls=0, finite_trace_calls=0, native_forward_calls=0,
            pid=process.pid, pid_birth=process.create_time(), rss_bytes=process.memory_info().rss,
            host_available_bytes=psutil.virtual_memory().available, cuda_initialized=torch.cuda.is_initialized(),
            actor_config=OmegaConf.to_container(cfg.actor_rollout_ref, resolve=True)), indent=2) + '\n')
    else:
        ray.init(num_cpus=8, include_dashboard=False)
        try:
            group = RayWorkerGroup(RayResourcePool([2], use_gpu=True, max_colocate_count=1),
                RayClassWithInitArgs(AuthorRewardCurveWorker, cfg.actor_rollout_ref, 'actor'))
            for phase, call in [('original_model_initialization', group.init_model),
                                ('original_checkpoint_load', lambda: group.load_checkpoint(
                                    local_path=str(checkpoint / 'actor'), del_local_after_load=False))]:
                print('TEXTCRAFT_AUTHOR_REWARD_CURVE_DRIVER ' + json.dumps(dict(phase=phase + '_begin', unix=time.time())), flush=True)
                call()
                print('TEXTCRAFT_AUTHOR_REWARD_CURVE_DRIVER ' + json.dumps(dict(phase=phase + '_complete', unix=time.time())), flush=True)
            result = group.inspect_author_reward_curves()
            (OUT / 'completed.json').write_text(json.dumps(dict(completed_unix=time.time(), unique_cases=21,
                padded_slots=22, optimizer_steps=0, scheduler_steps=0, backward_calls=0, finite_trace_calls=0,
                ranks=result), indent=2) + '\n')
        finally:
            ray.shutdown()
