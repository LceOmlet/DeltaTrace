"""Read saved real prefixes with the existing DT native outcome reader.

No collection, finite attribution, credit replacement, backward, or optimizer
step. Each B4 is one literal prefix and its full/single EOS interventions, so
all four rows retain the same original predictor position. The original VERL
worker, model/checkpoint loader, DT producer and categorical reader own all
model computation. This isolated research entry is not a production patch.
"""
from contextlib import nullcontext
import argparse
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


OUT = Path(os.environ['DT_TEXTCRAFT_READOUT_ROOT'])
INPUT = Path(os.environ['DT_TEXTCRAFT_READOUT_CASES'])
BASE = Path(os.environ['DT_TEXTCRAFT_READOUT_BASE'])
RUNNER_SHA = 'c7fc969f9f521993f2449ea5f364adcb3e0fdac5b01c38c103963639551516c1'


def identity(value):
    path = Path(inspect.getsourcefile(inspect.unwrap(value))).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def resources():
    import psutil
    memory = psutil.Process().memory_full_info()
    return dict(rss_bytes=memory.rss, pss_bytes=getattr(memory, 'pss', None),
                host_available_bytes=psutil.virtual_memory().available,
                cuda_free_bytes=torch.cuda.mem_get_info()[0],
                cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_reserved_bytes=torch.cuda.memory_reserved(),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())


@ray.remote
class NativeReadoutWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def inspect_saved_readout(self):
        from deltatrace_rollout import DeltaTraceRolloutProducer
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

        assert self.config.model.lora_rank == 8 and self.config.model.lora_alpha == 16
        assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
        inputs = json.loads(INPUT.read_bytes())
        cases = inputs['rank_cases'][self.rank]
        assert len(cases) == 32
        result = dict(scope=__doc__, rank=self.rank, pid=os.getpid(), input_path=str(INPUT),
                      input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
                      checkpoint=os.environ['DT_TEXTCRAFT_CHECKPOINT'],
                      optimizer_steps=0, scheduler_steps=0, backward_calls=0,
                      finite_trace_calls=0, native_forward_calls=0, cases=[],
                      sources={'original_worker': identity(ActorRolloutRefWorker),
                               'producer': identity(DeltaTraceRolloutProducer),
                               'observer': identity(self.inspect_saved_readout)},
                      selection=inputs['selection'])
        path = OUT / f'rank{self.rank}-readout.json'

        def save(phase, **extra):
            result.update(phase=phase, observed_unix=time.time())
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            print('TEXTCRAFT_NATIVE_READOUT ' + json.dumps(dict(
                phase=phase, rank=self.rank, pid=os.getpid(), completed_cases=len(result['cases']),
                unix=time.time(), **extra)), flush=True)

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
                                     native_reader=identity(runner.read_outcomes),
                                     native_forward=identity(runner.model.forward_root))
            assert result['sources']['runner']['sha256'] == RUNNER_SHA
            baseline = json.loads((BASE / 'prepared-diagnostic.json').read_bytes())
            producer_path = result['sources']['producer']['path']
            assert result['sources']['producer']['sha256'] == baseline['sources'][producer_path]
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
            result.update(outcome_token_ids=labels, model_dtype=str(next(self.actor_module_fsdp.parameters()).dtype),
                          native_fla_fp16=producer.native_fla_fp16,
                          event_attention_backend='flash_attention_2',
                          numerical_runner_options={key: getattr(runner, key, None) for key in
                              ('copy_replay_captures', 'offload_replay_mixer', 'gdn_head_batch_size',
                               'defer_diagnostics', 'pin_replay_host', 'pin_root_host', 'reuse_native_prefix')})
            with torch.no_grad(), precision:
                torch.cuda.reset_peak_memory_stats()
                result['allocation_before'] = resources()
                for index, case in enumerate(cases):
                    ids = case['selected_input_ids']
                    start, end = case['source_start'], case['source_end']
                    assert labels == case['outcome_token_ids']
                    assert ids[-1] == labels[case['observed_class_index']]
                    assert 0 <= start < end < len(ids) - 1 and len(ids) <= 32768
                    positions = case['probe_source_positions']
                    assert len(set(positions)) == 2 and all(0 <= p < end - start for p in positions)
                    # Native read_outcomes scores the final predictor. Do not
                    # batch different prefix lengths or pad beyond that point.
                    prefix = torch.tensor(ids[:-1], device='cuda', dtype=torch.long)
                    variants = prefix.unsqueeze(0).expand(4, -1).clone()
                    variants[1, start:end] = self.tokenizer.eos_token_id
                    for row, position in enumerate(positions, start=2):
                        variants[row, start + position] = self.tokenizer.eos_token_id
                    save('native_readout_case_begin', case_index=index, context_tokens=len(ids),
                         input_shape=list(variants.shape))
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    try:
                        values = runner.read_outcomes(variants, labels)
                        torch.cuda.synchronize()
                        values = values.detach().cpu()
                        result['native_forward_calls'] += 1
                        repeated = None
                        if index == 0:
                            second = runner.read_outcomes(variants, labels)
                            torch.cuda.synchronize()
                            second = second.detach().cpu()
                            result['native_forward_calls'] += 1
                            repeated = dict(log_probs=second.tolist(),
                                max_abs_difference=float((values - second).abs().max()),
                                torch_equal=torch.equal(values, second),
                                scope='Same-input native repetition; descriptive, not a new tolerance gate.')
                        seconds = time.perf_counter() - began
                    finally:
                        runner.model.release_owner_params()
                    assert values.shape == (4, len(labels))
                    target = values[:, case['observed_class_index']]
                    result['cases'].append(dict(case_index=index, traj_uid=case['traj_uid'],
                        source_step=case['source_step'], observed_return=case['observed_return'],
                        observed_class_index=case['observed_class_index'],
                        source_start=start, source_end=end, context_tokens=len(ids),
                        probe_source_positions=positions, probe_token_ids=[ids[start + p] for p in positions],
                        saved_probe_credit=case['saved_probe_credit'],
                        native_input_shape=list(variants.shape), native_logits_dtype=str(values.dtype),
                        variant_names=['factual', 'full_response_eos', 'single_eos_0', 'single_eos_1'],
                        native_outcome_log_probs=values.tolist(), native_target_log_probs=target.tolist(),
                        native_full_span_log_ratio=float(target[0] - target[1]),
                        native_single_log_ratios=(target[0] - target[2:]).tolist(),
                        same_input_repeat=repeated, seconds=seconds, allocation=resources()))
                    del values, target, variants, prefix
                    save('native_readout_case_complete', last_seconds=seconds)
                result['allocation_after'] = resources()
            save('complete_native_readout')
            return {'rank': self.rank, 'path': str(path),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'cases': len(cases)}
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
        # Ray reconstructs the subclass; its class object may not have a file,
        # while the actual Python method retains its source-code filename.
        # Check that exact metadata operation before any model initialization.
        import cloudpickle
        observer = NativeReadoutWorker.__ray_metadata__.modified_class.inspect_saved_readout
        serialized_observer = cloudpickle.loads(cloudpickle.dumps(observer))
        observer_identity = identity(serialized_observer)
        assert observer_identity == {'path': str(Path(__file__).resolve()),
                                     'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (OUT / 'native-owner-inspection.json').write_text(json.dumps(dict(
            scope='CPU-only owner binding/config inspection; no model constructed.',
            worker=identity(ActorRolloutRefWorker),
            worker_constructor=identity(ActorRolloutRefWorker.__init__),
            model_loader=identity(ActorRolloutRefWorker.init_model),
            checkpoint_loader=identity(ActorRolloutRefWorker.load_checkpoint),
            native_reader_inputs_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
            serialized_observer=observer_identity,
            cuda_initialized=torch.cuda.is_initialized(),
            actor_config=OmegaConf.to_container(cfg.actor_rollout_ref, resolve=True)), indent=2) + '\n')
    else:
        ray.init(num_cpus=8, include_dashboard=False)
        try:
            group = RayWorkerGroup(RayResourcePool([2], use_gpu=True, max_colocate_count=1),
                RayClassWithInitArgs(NativeReadoutWorker, cfg.actor_rollout_ref, 'actor'))
            print('TEXTCRAFT_NATIVE_READOUT_DRIVER ' + json.dumps(dict(
                phase='original_model_initialization_begin', unix=time.time())), flush=True)
            group.init_model()
            print('TEXTCRAFT_NATIVE_READOUT_DRIVER ' + json.dumps(dict(
                phase='original_checkpoint_load_begin', unix=time.time())), flush=True)
            group.load_checkpoint(local_path=str(checkpoint / 'actor'), del_local_after_load=False)
            print('TEXTCRAFT_NATIVE_READOUT_DRIVER ' + json.dumps(dict(
                phase='original_checkpoint_load_complete', unix=time.time())), flush=True)
            result = group.inspect_saved_readout()
            (OUT / 'completed.json').write_text(json.dumps(dict(
                completed_unix=time.time(), cases=64, optimizer_steps=0, scheduler_steps=0,
                backward_calls=0, finite_trace_calls=0, ranks=result), indent=2) + '\n')
        finally:
            ray.shutdown()
