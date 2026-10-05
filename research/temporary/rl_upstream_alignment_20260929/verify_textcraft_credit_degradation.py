"""Read-only checkpoint25/step26 TextCraft event-deletion diagnostic.

Uses the original VERL worker/checkpoint manager and original DT event readout.
Each native B4 contains one literal sample's factual/full-span-EOS/two
single-token-EOS variants, so all four predictors keep their original position.
Recorded joint-span credit and direct deletion effects are reported raw. This
does not change credit, run PPO, define acceptance thresholds, or step an optimizer.
"""
from contextlib import nullcontext
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


OUT = Path(os.environ['DT_TEXTCRAFT_PROBE_ROOT'])
RECORDS = OUT / 'checkpoint-matched-records.json'
SOURCE = OUT / 'source.json'
OPTIONS = (OUT / 'native-launch-options.json' if (OUT / 'native-launch-options.json').is_file()
           else OUT / 'launch.json')
RUNNER_SHA = 'c7fc969f9f521993f2449ea5f364adcb3e0fdac5b01c38c103963639551516c1'


def source_of(owner):
    path = Path(inspect.getsourcefile(owner)).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def memory_snapshot():
    try:
        import psutil
        host = psutil.Process().memory_full_info()
        pss = getattr(host, 'pss', None)
        rss = host.rss
    except (ImportError, OSError):
        pss = rss = None
    stats = getattr(torch.cuda.memory, 'host_memory_stats', None)
    return dict(pss_bytes=pss, rss_bytes=rss,
        physical_gpu_free_bytes=torch.cuda.mem_get_info()[0],
        torch_allocated_bytes=torch.cuda.memory_allocated(),
        torch_reserved_bytes=torch.cuda.memory_reserved(),
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        raw_host_allocator_stats=stats() if callable(stats) else None,
        host_allocator_scope='Raw owner counters only; allocated counters are not interpreted as live bytes.')


def sampled_qva(log_ratios, reward):
    """Use the unchanged accepted composition owner, never reconstruct its rule."""
    from counterfactual import reward_event_token_credit
    values = torch.as_tensor(log_ratios, dtype=torch.float64).reshape(1, 1, -1)
    credit = reward_event_token_credit(values,
        torch.tensor([[reward]], dtype=torch.float64),
        torch.ones_like(values, dtype=torch.bool),
        torch.ones((1, values.shape[-1]), dtype=torch.bool))
    return dict(q=credit.q_estimates[0].tolist(), v=credit.v_estimates[0].tolist(),
                a=credit.advantages[0].tolist(), composition_dtype=str(values.dtype))


def selected_positions(sample, tokenizer):
    """Two bounded audit probes, selected using the actual tokenizer strings.

    Non-format means neither a special token nor pure whitespace. It does not
    imply that the token is an executed command; the original text is recorded.
    """
    start, end = sample['source_start'], sample['source_end']
    ids = sample['selected_input_ids'][start:end]
    values = sample['source_signed']
    assert len(ids) == len(values)
    tokens = [tokenizer.decode([value], skip_special_tokens=False) for value in ids]
    special = set(tokenizer.all_special_ids)
    is_format = [value in special or not text.strip() for value, text in zip(ids, tokens)]
    negative_format = [i for i, (flag, value) in enumerate(zip(is_format, values))
                       if flag and value < 0]
    positive_nonformat = [i for i, (flag, value) in enumerate(zip(is_format, values))
                          if not flag and value > 0]
    if not negative_format or not positive_nonformat:
        raise ValueError('The literal sample has no requested negative-format/positive-nonformat probe.')
    positions = [min(negative_format, key=values.__getitem__),
                 max(positive_nonformat, key=values.__getitem__)]
    return [dict(kind=kind, source_position=i, input_position=start+i,
                 token_id=ids[i], token=tokens[i], recorded_joint_signed=values[i])
            for kind, i in zip(('negative_format', 'maximum_positive_nonformat'), positions)]


@ray.remote
class TextCraftDeletionWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def inspect_gradients(self):
        from observe_native_actor_loss_gradients import observe_gradients
        entries=json.loads(RECORDS.read_bytes())
        entry,=[item for item in entries if item['step']==26]
        lines=json.loads(os.environ.get('DT_TEXTCRAFT_RECORD_LINES','[727,730]'))
        record,=[item for item in entry['minimum_records'] if item['line']==lines[self.rank]]
        def save(phase, **values):
            print(json.dumps(dict(rank=self.rank,phase=phase,**values)),flush=True)
        return observe_gradients(self,record['data']['samples'],OUT,save)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def inspect_deletions(self):
        from deltatrace_rollout import DeltaTraceRolloutProducer
        from counterfactual import reward_event_token_credit
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

        assert self.config.model.lora_rank == 8 and self.config.model.lora_alpha == 16
        assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
        lines = json.loads(os.environ.get('DT_TEXTCRAFT_RECORD_LINES', '[727,730]'))
        assert len(lines) == 2
        # Parent mapped these exact original stdout lines to worker ranks:
        # 727 -> PID3236008/rank0; 730 -> PID3240133/rank1. Do not deduplicate.
        entries = json.loads(RECORDS.read_bytes())
        entry, = [item for item in entries if item['step'] == 26]
        record, = [item for item in entry['minimum_records'] if item['line'] == lines[self.rank]]
        samples = record['data']['samples']
        assert len(samples) == 4
        eos = record['data']['eos_token_id']
        outcomes = record['data']['outcome_token_ids']
        assert eos == self.tokenizer.eos_token_id
        result = dict(scope=__doc__, rank=self.rank, pid=os.getpid(), step=26,
            checkpoint=os.environ['DT_TEXTCRAFT_CHECKPOINT'], original_record_line=record['line'],
            records_path=str(RECORDS), records_sha256=hashlib.sha256(RECORDS.read_bytes()).hexdigest(),
            source_manifest_path=str(SOURCE), source_manifest_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            options_path=str(OPTIONS), options_sha256=hashlib.sha256(OPTIONS.read_bytes()).hexdigest(),
            effective_worker_config=OmegaConf.to_container(self.config, resolve=True),
            event_outcome_token_ids=outcomes,
            event_outcome_tokens=[self.tokenizer.decode([value]) for value in outcomes],
            original_task='TextCraft', checkpoint_completed_step=25,
            local_batch_size=4, optimizer_steps=0, samples=[],
            selection_scope='First original logged B4 per rank; selected worst-minimum batches, not representative batch statistics.',
            comparison_scope='Raw recorded joint-span attribution versus native single-input deletion; no numerical pass/fail criterion.',
            runtime_cache_environment={key: os.environ.get(key) for key in
                ('TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'HF_HOME', 'TRANSFORMERS_CACHE',
                 'CUDA_VISIBLE_DEVICES', 'MACA_VISIBLE_DEVICES')},
            imported_sources={'verl_worker': source_of(ActorRolloutRefWorker),
                              'original_actor': source_of(type(self.actor)),
                              'dt_producer': source_of(DeltaTraceRolloutProducer),
                              'qva_composition': source_of(reward_event_token_credit)})
        path = OUT / f'rank{self.rank}.json'
        def save(phase):
            result.update(phase=phase, observed_unix=time.time())
            path.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(dict(rank=self.rank, phase=phase, samples=len(result['samples']))), flush=True)
        save('original_worker_ready')
        training = self.actor_module_fsdp.training
        producer = runner = text = None
        previous_attention = None
        try:
            if self._is_offload_param:
                load_fsdp_model_to_gpu(self.actor_module_fsdp)
            self.actor_module_fsdp.eval()
            producer = DeltaTraceRolloutProducer(self.actor_module_fsdp,
                eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
                invalid_action_penalty_coef=(self.config.actor.invalid_action_penalty_coef
                    if self.config.actor.get('use_invalid_action_penalty', True) else 0.0))
            runner = producer.runner
            from qwen35_answer_finite import categorical_head_logits
            result['imported_sources'].update(dt_runner=source_of(type(runner)),
                native_read_outcomes=source_of(runner.read_outcomes),
                categorical_head=source_of(categorical_head_logits),
                original_registered_forward=source_of(runner.model.forward_root))
            assert result['imported_sources']['dt_runner']['sha256'] == RUNNER_SHA
            text = runner.model.model.language_model
            previous_attention = text.config._attn_implementation
            text.set_attn_implementation('flash_attention_2')
            if producer.native_fla_fp16:
                from accelerated.qwen35.native_fla_precision import native_fla_fp16
                precision = native_fla_fp16(self.actor_module_fsdp)
                result['imported_sources']['native_precision_context'] = source_of(native_fla_fp16)
            else:
                precision = nullcontext()
            result.update(torch_version=torch.__version__, native_fla_fp16=producer.native_fla_fp16,
                previous_attention_backend=previous_attention, event_attention_backend='flash_attention_2',
                owner_readout_uses_cache=True, persistent_prefix_bank=False,
                numerical_runner_options={key: getattr(runner, key, None) for key in
                    ('copy_replay_captures', 'offload_replay_mixer', 'gdn_head_batch_size',
                     'defer_diagnostics', 'pin_replay_host', 'pin_root_host', 'reuse_native_prefix')})
            with torch.no_grad(), precision:
                torch.cuda.reset_peak_memory_stats()
                result['allocation_before'] = memory_snapshot()
                for sample_index, sample in enumerate(samples):
                    probes = selected_positions(sample, self.tokenizer)
                    ids = sample['selected_input_ids']
                    start, end = sample['source_start'], sample['source_end']
                    assert len(ids) == sample['trace']['context_tokens']
                    assert ids[-1] in outcomes and 0 <= start < end < len(ids)-1
                    factual = torch.tensor(ids[:-1], device='cuda', dtype=torch.long)
                    variants = factual.unsqueeze(0).expand(4, -1).clone()
                    variants[1, start:end] = eos
                    for row, probe in enumerate(probes, start=2):
                        variants[row, probe['input_position']] = eos
                    torch.cuda.synchronize()
                    began = time.perf_counter()
                    try:
                        logp = runner.read_outcomes(variants, outcomes)
                        torch.cuda.synchronize()
                    finally:
                        # The original owner view remains the parameter lifetime owner.
                        runner.model.release_owner_params()
                    seconds = time.perf_counter()-began
                    target_index = outcomes.index(ids[-1])
                    target_logp = logp[:, target_index].detach().cpu()
                    direct = target_logp[0] - target_logp[1:]
                    probe_values = []
                    for probe, deletion in zip(probes, direct[1:].tolist()):
                        probe_values.append(dict(**probe, direct_single_eos_log_ratio=deletion,
                            recorded_minus_direct=probe['recorded_joint_signed']-deletion,
                            recorded_qva=sampled_qva([probe['recorded_joint_signed']], sample['observed_return']),
                            direct_qva=sampled_qva([deletion], sample['observed_return'])))
                    result['samples'].append(dict(sample_index=sample_index,
                        traj_uid=sample['traj_uid'], source_step=sample['trace']['source_step'],
                        observed_return=sample['observed_return'], source_start=start, source_end=end,
                        original_selected_input_ids=ids, original_source_signed=sample['source_signed'],
                        original_trace=sample['trace'], original_source_qva=sampled_qva(
                            sample['source_signed'], sample['observed_return']),
                        original_response_text=self.tokenizer.decode(ids[start:end], skip_special_tokens=False),
                        predictor_input_position=len(ids)-2, predictor_token_id=ids[-2],
                        target_token_id=ids[-1], target_index=target_index,
                        native_variant_names=['factual', 'full_response_eos',
                            'negative_format_eos', 'maximum_positive_nonformat_eos'],
                        native_input_shape=list(variants.shape), native_logits_dtype=str(logp.dtype),
                        native_outcome_log_probs=logp.detach().cpu().tolist(),
                        native_target_log_probs=target_logp.tolist(), native_forward_seconds=seconds,
                        direct_full_span_log_ratio=float(direct[0]),
                        recorded_full_span_root=sample['trace']['root_effect'],
                        recorded_root_minus_direct=sample['trace']['root_effect']-float(direct[0]),
                        probes=probe_values, allocation=memory_snapshot()))
                    del variants, factual, logp, target_logp, direct
                    save('native_event_deletion_observed')
                result['allocation_after'] = memory_snapshot()
            save('completed_raw_diagnostic')
            return result
        finally:
            if text is not None and previous_attention is not None:
                text.set_attn_implementation(previous_attention)
            self.actor_module_fsdp.train(training)
            if runner is not None:
                runner.model.release_owner_params()
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)


if __name__ == '__main__':
    checkpoint = Path(os.environ['DT_TEXTCRAFT_CHECKPOINT'])
    assert checkpoint.name == 'global_step_25'
    cfg = OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    launch = json.loads(OPTIONS.read_bytes())
    options = launch['options'] if 'options' in launch else launch
    for key, value in options.items():
        OmegaConf.update(cfg, key.lstrip('+'), value, force_add=True)
    # Original train.log line523 and both owner workers report 330. Loading
    # the checkpoint retains the original optimizer state; it is never stepped.
    cfg.actor_rollout_ref.actor.optim.total_training_steps = 330
    (OUT/'effective-config.yaml').write_text(OmegaConf.to_yaml(cfg))
    ray.init(num_cpus=8, include_dashboard=False)
    try:
        group = RayWorkerGroup(RayResourcePool([2], use_gpu=True, max_colocate_count=1),
            RayClassWithInitArgs(TextCraftDeletionWorker, cfg.actor_rollout_ref, 'actor'))
        group.init_model()
        group.load_checkpoint(local_path=str(checkpoint/'actor'), del_local_after_load=False)
        (OUT/'checkpoint-load.json').write_text(json.dumps(dict(path=str(checkpoint),
            actor_path=str(checkpoint/'actor'), original_loader='ActorRolloutRefWorker.load_checkpoint',
            returned_unix=time.time()), indent=2)+'\n')
        results = group.inspect_deletions()
        (OUT/'result.json').write_text(json.dumps(results, indent=2)+'\n')
        if os.environ.get('DT_TEXTCRAFT_GRADIENT_DIAGNOSTIC')=='1':
            gradients=group.inspect_gradients()
            (OUT/'gradient-results.json').write_text(json.dumps(gradients,indent=2)+'\n')
    finally:
        ray.shutdown()
