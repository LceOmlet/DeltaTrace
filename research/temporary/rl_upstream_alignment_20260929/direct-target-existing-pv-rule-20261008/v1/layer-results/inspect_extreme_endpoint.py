"""One saved real B4: native factual versus one-source EOS joint-target score.

Prepared diagnostic only. VERL owns actor/PEFT/FSDP initialization. The current
DT owners select and score targets; no DT attribute, rollout, optimizer step,
checkpoint restore, alternate scorer, tolerance, or reward rule is implemented.
Both diagnostic ranks consume the same original four cases for equal FSDP call
counts. Only the evidenced source row changes; the other three original pairs
are identity controls. Each task keeps its own original source and head.
This single-candidate diagnostic does not replace author cumulative deletion.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import time

import torch


ROOT = '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
CASES = {
    'appworld': dict(source_sha256='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0',
        native_sha256='aaa03be7fb10624918e5289aa1fc3409ae2d81ce32243a84df5539c48727be85',
        uid='b98de8ee-995e-49e7-acb8-1510b4433a6a', row=0, trajectory_index=19,
        response_slot=4498, packed_slot=7260, token_id=198, rank=1, batch=6,
        native=ROOT+'/receipts/direct-target-prefix-runtime-20261007-v1/appworld-first-dt/rank1-readout-native-batch-6.pt'),
    'textcraft': dict(source_sha256='2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52',
        native_sha256='19b18c5f4204ec4d88e40d72d350a06954f64452824a8662e95497319f89a37a',
        uid='6e76f70e-bdeb-4726-8b35-d9e21eff2f68', row=3, trajectory_index=77,
        response_slot=351, packed_slot=666, token_id=14606, rank=1, batch=21,
        native=ROOT+'/receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt/rank1-readout-native-batch-21.pt'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(owner):
    path = Path(inspect.getsourcefile(owner)).resolve()
    return dict(path=str(path), sha256=sha(path))


def load_request(source_path, native_path, spec):
    """Keep original case/offset/row objects from the bound native-return file."""
    if sha(source_path) != spec['source_sha256'] or sha(native_path) != spec['native_sha256']:
        raise ValueError('Use the selected evidenced source and original native B4')
    source = json.loads(Path(source_path).read_bytes())
    native = torch.load(native_path, map_location='cpu', weights_only=False)
    rows = sorted(native['rows'], key=lambda row: row['batch_row'])
    if len(rows) != 4 or [row['batch_row'] for row in rows] != list(range(4)):
        raise ValueError('Retain the complete original B4 and its native batch order')
    candidate = rows[spec['row']]
    positions = candidate['suffix_positions'].eq(spec['response_slot']).nonzero().flatten()
    if (candidate['traj_uid'] != spec['uid'] or
            candidate['trajectory_index'] != spec['trajectory_index'] or
            positions.numel() != 1 or
            candidate['prompt_length'] + int(positions[0]) != spec['packed_slot'] or
            int(candidate['selected'][spec['packed_slot']]) != spec['token_id'] or
            not bool(candidate['prior'][spec['response_slot']]) or
            bool(candidate['target'][spec['response_slot']])):
        raise ValueError('The saved candidate identity/token/source mapping changed')
    for row in rows:
        if not torch.equal(row['case']['target_ids'], row['selected'][row['prompt_length']:]):
            raise ValueError('Use the original target IDs, not reconstructed text')
    if (not source['fresh_base_model'] or source['checkpoint_restore_requested'] or
            source['resume_mode'] != 'disable' or
            (source['lora_rank'], source['lora_alpha']) != (8, 16)):
        raise ValueError('Use the recorded fresh-base LoRA8/16 owner configuration')
    return source, native, rows


def make_pair(rows, width, eos_token_id, pad_rows, spec):
    """Compose original VERL right padding; perturb exactly one existing ID."""
    selected = pad_rows([row['selected'].tolist() for row in rows],
                        eos_token_id, max_length=width)
    reference = selected.clone()
    reference[spec['row'], spec['packed_slot']] = eos_token_id
    return torch.stack((reference, selected), dim=1).flatten(0, 1)


def geometry(source, native, rows, spec):
    candidate = rows[spec['row']]
    return dict(source_sha256=spec['source_sha256'], native_sha256=spec['native_sha256'],
        source_rank=spec['rank'], source_batch=spec['batch'], original_batch_rows=4, paired_rows=8,
        candidate=dict(row=spec['row'], traj_uid=spec['uid'], trajectory_index=spec['trajectory_index'],
                       response_slot=spec['response_slot'], packed_slot=spec['packed_slot'],
                       original_input_slot=candidate['row']['input_ids'].numel()
                           - candidate['width'] + spec['response_slot'],
                       token_id=spec['token_id'], saved_native_signed_dtype=str(native['native_signed'].dtype),
                       saved_native_signed=float(native['native_signed'][spec['row'], spec['packed_slot']]),
                       reward=float(candidate['row']['dt_direct_reward'])),
        selected_lengths=[row['selected'].numel() for row in rows],
        target_counts=[len(row['target_offsets']) for row in rows],
        target_offsets=[row['target_offsets'] for row in rows],
        provenance=native.get('provenance'),
        formal_initialization=dict(base=source['fresh_base_model'],
            resume_mode=source['resume_mode'], checkpoint_restore_requested=False),
        weight_scope='Independent original VERL fresh actor, not live-weight bitwise replay; '
            'PEFT default B=0 and actual local B shards are checked before scoring; no update is called',
        operations=dict(native_paired_forward_per_rank=1, rollout=0, DT=0,
                        backward=0, optimizer=0, checkpoint_restore=0),
        numeric_scope='Original native logits dtype, original FP32 target log-softmax, '
            'original FP64 joint sums; differences are descriptive, no invented tolerance')


def target_snapshot(selection, logp):
    """Save the original target scorer's output and exact selection axes."""
    reference = logp[0::2].detach().cpu()
    factual = logp[1::2].detach().cpu()
    return dict(samples=selection.samples.detach().cpu(),
        predictor_positions=selection.positions.detach().cpu(),
        labels=selection.labels.detach().cpu(),
        counts=selection.counts, offsets=selection.offsets,
        reference_target_logp=reference, factual_target_logp=factual,
        factual_minus_reference=factual-reference,
        scope='Original target scorer FP32 values; paired reference/factual rows. '
              'Predictor positions are on the original full packed input axis; '
              'descriptive endpoint diagnosis, no numeric acceptance threshold')


def causal_description(targets, saved_signed, spec):
    """Describe the saved scorer outputs; do not change credit or set a tolerance."""
    row = targets['samples'].eq(spec['row'])
    future = row & targets['predictor_positions'].ge(spec['packed_slot'])
    earlier = row & ~future
    factual = targets['factual_target_logp'].double()
    reference = targets['reference_target_logp'].double()
    future_factual = factual[future].sum()
    future_reference = reference[future].sum()
    earlier_delta = factual[earlier]-reference[earlier]
    return dict(source_row=spec['row'], source_packed_slot=spec['packed_slot'],
        earlier_target_count=int(earlier.sum()), future_target_count=int(future.sum()),
        earlier_factual_logp=float(factual[earlier].sum()),
        earlier_reference_logp=float(reference[earlier].sum()),
        earlier_delta_sum=float(earlier_delta.sum()),
        earlier_delta_maxabs=float(earlier_delta.abs().max()) if earlier_delta.numel() else None,
        factual_future_joint_logp=float(future_factual),
        single_eos_future_joint_logp=float(future_reference),
        native_single_eos_future_effect=float(future_factual-future_reference),
        saved_DT_signed=float(saved_signed),
        DT_implied_deleted_future_logp_if_earlier_unchanged=float(future_factual-saved_signed),
        exact_single_delete_d_lower_bound=float(future_factual),
        exact_single_delete_log_ratio_upper_bound=float(-future_factual),
        exact_single_delete_ratio_upper_bound=float((-future_factual).exp()),
        scope='Only descriptive p<=1 and causal ordering checks on original target values. '
              'Future means target predictor>=source, hence target token>source. '
              'No tolerance, clipping, correction, credit replacement, or acceptance gate')


def check_imports(source):
    """Check only the actual owners used by this bounded diagnostic."""
    roles = {
        'verl.workers.fsdp_workers': source['verl_sha256']['verl/workers/fsdp_workers.py'],
        'deltatrace_rollout': source['entry_sha256']['deltatrace_rollout.py'],
        'qwen35_answer_finite': source['dt_source_sha256']['clean/qwen35/qwen35_answer_finite.py'],
        'native_target_logit_rows': source['dt_source_sha256']['clean/qwen35/native_target_logit_rows.py'],
        'transformers.models.qwen3_5.modeling_qwen3_5': source['canonical_HF_owner']['sha256'],
    }
    result = {}
    for name, expected in roles.items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        result[name] = dict(path=str(path), sha256=sha(path))
        if result[name]['sha256'] != expected:
            raise ValueError('Imported owner differs from the selected frozen task source: ' + name)
    return result


def actor_initialization_steps(configured, owner_resolved):
    """Pass the TaskRunner's resolved scalar to the original actor scheduler."""
    if configured is not None:
        if owner_resolved is not None and owner_resolved != configured:
            raise ValueError('Do not replace the task\'s original explicit total steps')
        return configured
    if owner_resolved is None or type(owner_resolved) is not int or owner_resolved < 1:
        raise ValueError('Pass the original TaskRunner-resolved total training steps; '
                         'actor-only initialization does not construct another dataloader')
    return owner_resolved


def make_worker():
    import ray
    from verl.single_controller.base.decorator import Dispatch, register
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    @ray.remote
    class EndpointWorker(ActorRolloutRefWorker):
        @register(dispatch_mode=Dispatch.ONE_TO_ALL)
        def inspect_endpoint(self, source_path, native_path, output, case_name):
            import psutil
            from deltatrace_rollout import DeltaTraceRolloutProducer
            from native_target_logit_rows import NativeTargetLogitRows
            from qwen35_answer_finite import PackedAnswerTargets, selected_target_log_probs
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
            from verl.utils.torch_functional import pad_2d_list_to_length

            spec = CASES[case_name]
            source, native, rows = load_request(source_path, native_path, spec)
            result = dict(status='started', rank=self.rank, pid=os.getpid(),
                          prepared=geometry(source, native, rows, spec), owners=check_imports(source))
            path = Path(output) / f'rank{self.rank}.json'
            def save(phase, **values):
                result.update(phase=phase, observed_unix=time.time(), **values)
                path.write_text(json.dumps(result, indent=2) + '\n')
            assert self._is_actor and not self._is_rollout and not self._is_ref
            assert (self.config.model.lora_rank, self.config.model.lora_alpha) == (8, 16)
            assert self.actor.config.ppo_micro_batch_size_per_gpu == 4
            training = self.actor_module_fsdp.training
            producer = None
            text = None
            previous_attention = None
            try:
                # Read PEFT's actual initialized small shards, not inferred A/B values.
                configs = getattr(self.actor_module_fsdp, 'peft_config')
                adapters = {key: dict(r=value.r, alpha=value.lora_alpha,
                    init_lora_weights=value.init_lora_weights) for key, value in configs.items()}
                if any(v['init_lora_weights'] is not True or
                       (v['r'], v['alpha']) != (8, 16) for v in adapters.values()):
                    raise ValueError('Expected original default PEFT B-zero initialization')
                shards = []
                for name, parameter in self.actor_module_fsdp.named_parameters():
                    if '.lora_B.' not in name:
                        continue
                    value = parameter.detach()
                    value = value.to_local() if hasattr(value, 'to_local') else value
                    shards.append(dict(name=name, elements=value.numel(),
                                       nonzero=int(torch.count_nonzero(value))))
                if not shards or any(item['nonzero'] for item in shards):
                    raise ValueError('Independent fresh actor has nonzero/missing LoRA B shards')
                save('fresh_original_actor_B_zero', adapters=adapters, lora_B_local_shards=shards,
                     no_update_scope='This independent actor has not loaded a checkpoint or called update_actor')
                if self._is_offload_param:
                    load_fsdp_model_to_gpu(self.actor_module_fsdp)
                self.actor_module_fsdp.eval()
                producer = DeltaTraceRolloutProducer(self.actor_module_fsdp,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    invalid_action_penalty_coef=(self.config.actor.invalid_action_penalty_coef
                        if self.config.actor.get('use_invalid_action_penalty', True) else 0.0))
                runner = producer.runner
                text = runner.model.model.language_model
                previous_attention = text.config._attn_implementation
                text.set_attn_implementation('flash_attention_2')
                precision = nullcontext()
                if producer.native_fla_fp16:
                    from accelerated.qwen35.native_fla_precision import native_fla_fp16
                    precision = native_fla_fp16(self.actor_module_fsdp)
                pair = make_pair(rows, native['native_signed'].shape[1],
                                 self.tokenizer.eos_token_id, pad_2d_list_to_length, spec).to('cuda')
                selection = PackedAnswerTargets([row['case'] for row in rows],
                    [row['target_offsets'] for row in rows], pair.shape[1], pair.device)
                selector = NativeTargetLogitRows(selection)
                # Same owner native-convolution option as the current formal runner.
                conv_scope = runner.attribute.__func__.__globals__['_native_conv_initial_states_scope']
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                began = time.perf_counter()
                save('native_pair_begin', shape=list(pair.shape),
                     target_predictor_union=selector.rows.numel(),
                     pss_bytes=psutil.Process().memory_full_info().pss)
                with torch.no_grad(), precision, conv_scope(text.layers, runner.native_conv_initial_states):
                    output_ = runner.model.forward_root(input_ids=pair,
                        attention_mask=torch.ones_like(pair), use_cache=False,
                        logits_to_keep=selector.rows)
                    logits_dtype = str(output_.logits.dtype)
                    packed = selector.pack_logits(output_.logits)
                    del output_
                    logp = selected_target_log_probs(packed, selection)
                    del packed
                    reference = selection.sample_sums(logp[0::2].double()).cpu()
                    factual = selection.sample_sums(logp[1::2].double()).cpu()
                    torch.cuda.synchronize()
                targets = target_snapshot(selection, logp)
                targets.update(source_sha256=spec['source_sha256'], native_sha256=spec['native_sha256'],
                               candidate=geometry(source, native, rows, spec)['candidate'],
                               reference_joint_logp=reference, factual_joint_logp=factual)
                description = causal_description(targets,
                    native['native_signed'][spec['row'], spec['packed_slot']], spec)
                target_path = Path(output) / f'rank{self.rank}-target-logp.pt'
                torch.save(targets, target_path)
                previous = native['detail'].get('per_sample', [])
                save('native_pair_complete', status='completed_diagnostic_not_acceptance',
                    native_logits_dtype=logits_dtype, target_logp_dtype=str(logp.dtype),
                    per_target_artifact=dict(path=str(target_path), sha256=sha(target_path)),
                    causal_description=description,
                    reference_joint_logp=reference.tolist(), factual_joint_logp=factual.tolist(),
                    native_single_delete_effect=(factual-reference).tolist(),
                    saved_original_factual_joint_logp=[item.get('factual_target_logp') for item in previous],
                    factual_minus_saved_original=[float(factual[i])-item['factual_target_logp']
                        for i, item in enumerate(previous)],
                    seconds=time.perf_counter()-began,
                    pss_bytes=psutil.Process().memory_full_info().pss,
                    peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
                    runtime_device_free_bytes=torch.cuda.mem_get_info()[0],
                    control_scope='The other three original rows are identical endpoint pairs; differences '
                        'and factual drift are recorded without a new tolerance or correction')
                return dict(rank=self.rank, path=str(path), sha256=sha(path))
            except BaseException as error:
                import traceback
                save('failed', status='failed', error=repr(error), traceback=traceback.format_exc())
                raise
            finally:
                if text is not None and previous_attention is not None:
                    text.set_attn_implementation(previous_attention)
                self.actor_module_fsdp.train(training)
                if producer is not None:
                    producer.runner.model.release_owner_params()
                if self._is_offload_param:
                    offload_fsdp_model_to_cpu(self.actor_module_fsdp)
    return EndpointWorker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--native', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--owner-total-training-steps', type=int,
        help='Original TaskRunner-resolved scalar, only needed when source trainer total is null')
    parser.add_argument('--owner-total-steps-evidence', type=Path,
        help='Original log/receipt that supplied the resolved scalar; record only')
    args = parser.parse_args()
    spec = CASES[args.case]
    args.native = args.native or Path(spec['native'])
    source, native, rows = load_request(args.source, args.native, spec)
    args.output.mkdir(parents=True, exist_ok=True)
    prepared = geometry(source, native, rows, spec)
    prepared.update(status='prepared_only', script_sha256=sha(__file__),
                    cuda_initialized=torch.cuda.is_initialized())
    (args.output/'prepared.json').write_text(json.dumps(prepared, indent=2)+'\n')
    if args.inspect_only:
        print(json.dumps(prepared))
        return
    import ray
    from omegaconf import OmegaConf
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
    check_imports(source)
    cfg_path = Path(source['verl_root'])/'verl/trainer/config/ppo_trainer.yaml'
    if sha(cfg_path) != source['source_bindings'][str(cfg_path)]:
        raise ValueError('Use the original bound VERL config')
    cfg = OmegaConf.load(cfg_path)
    for key, value in source['startup_options'].items():
        OmegaConf.update(cfg, key.lstrip('+'), value, force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps = actor_initialization_steps(
        cfg.trainer.total_training_steps, args.owner_total_training_steps)
    (args.output/'actor-initialization.json').write_text(json.dumps(dict(
        source_trainer_total_training_steps=cfg.trainer.total_training_steps,
        original_owner_resolved=args.owner_total_training_steps,
        actor_optim_total_training_steps=cfg.actor_rollout_ref.actor.optim.total_training_steps,
        evidence=(dict(path=str(args.owner_total_steps_evidence),
                       sha256=sha(args.owner_total_steps_evidence))
                  if args.owner_total_steps_evidence else None),
        scope='Original actor initialization needs the original TaskRunner-resolved '
              'scheduler scalar; no new training budget, optimizer step, or dataloader'), indent=2)+'\n')
    (args.output/'effective-config.yaml').write_text(OmegaConf.to_yaml(cfg))
    ray.init(num_cpus=8, include_dashboard=False)
    try:
        group = RayWorkerGroup(RayResourcePool([2], use_gpu=True, max_colocate_count=1),
            RayClassWithInitArgs(make_worker(), cfg.actor_rollout_ref, 'actor'))
        group.init_model()
        result = group.inspect_endpoint(str(args.source), str(args.native), str(args.output), args.case)
        (args.output/'completed.json').write_text(json.dumps(dict(
            scope=__doc__, ranks=result, optimizer_steps=0, completed_unix=time.time()), indent=2)+'\n')
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
