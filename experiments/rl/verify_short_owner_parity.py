"""Short-chain numerical artifacts from the actual pinned PPO actor.

The upstream case imports an archived pristine owner source file for comparison
only. Both cases use the same real Qwen, DT advantages, worker and optimizer.
This is a numerical interface test, not task evaluation or a timing benchmark.
"""
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
import random
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor-source', type=Path, help='Pristine pinned actor file; omitted uses installed actor')
    p.add_argument('--paired-owner-source', type=Path,
                   help='Also run the pristine actor in this worker after restoring weights and optimizer state')
    p.add_argument('--attention', choices=['flash_attention_2', 'sdpa'], default='flash_attention_2')
    p.add_argument('--reshard-after-forward', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--attention-check', action='store_true',
                   help='Compare original FA/math forwards in this worker with identical weights and DT inputs')
    p.add_argument('--trim-shared-padding', action=argparse.BooleanOptionalAction, default=False,
                   help='Enable the production shared-padding optimization; calibrate against original FA/math updates')
    p.add_argument('--previous-padding-source', type=Path,
                   help='Check production active-token updates before/after the owner mask backport')
    p.add_argument('--left-padding', type=int, default=165,
                   help='Explicit shared padding; 165 exercises removal of two complete FLA chunks')
    p.add_argument('--effective-input-tokens', type=int, default=0,
                   help='Explicit long numerical fixture length; 0 keeps the recorded short row')
    p.add_argument('--response-tokens', type=int, default=0,
                   help='Explicit synthetic action width for long numerical comparison; not task evaluation')
    p.add_argument('--owner-math', action=argparse.BooleanOptionalAction, default=True,
                   help='Also run the BF16 math baseline; long direct owner comparisons can omit it')
    p.add_argument('--head-only-comparison', action='store_true',
                   help='Also use the installed actor with trimming disabled to separate head slicing from padding effects')
    p.add_argument('--repeat-installed', action='store_true',
                   help='Repeat the identical installed path after restoring state to measure execution variability')
    p.add_argument('--fp32-reference-from', type=Path,
                   help='Original FP32 math reference from the same checkpoint, saved LoRA initial values, IDs and DT advantages')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--artifacts', type=Path, required=True)
    args = p.parse_args()
    if args.actor_source and args.paired_owner_source:
        p.error('Use either a standalone reference actor or a paired owner comparison')
    result = dict(scope=__doc__, attention=args.attention, reshard_after_forward=args.reshard_after_forward,
                  actor_source=str(args.actor_source) if args.actor_source else 'installed', context_cap=32768)
    result['verifier_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['comparison_performed'] = False
    result['trim_shared_padding'] = args.trim_shared_padding
    result['runtime_numerics'] = {k:os.environ.get(k) for k in
        ('CUBLAS_WORKSPACE_CONFIG', 'FLASH_ATTENTION_DETERMINISTIC')}
    artifacts = {}
    started = time.perf_counter()
    try:
        random.seed(2026)
        np.random.seed(2026)
        torch.manual_seed(2026)
        torch.use_deterministic_algorithms(True)
        os.environ['VERL_ATTN_IMPLEMENTATION'] = args.attention
        os.environ['VERL_TRIM_SHARED_PADDING'] = '1' if args.trim_shared_padding else '0'
        cfg = OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
        c = cfg.actor_rollout_ref
        c.model.path = os.environ['MODEL_PATH']
        c.model.lora_rank, c.model.lora_alpha = 1, 2
        c.model.trust_remote_code = True
        c.actor.strategy = 'fsdp2'
        c.actor.ppo_mini_batch_size = 4
        c.actor.ppo_micro_batch_size_per_gpu = 1
        c.actor.ppo_max_token_len_per_gpu = 32768
        c.actor.use_torch_compile = False
        c.actor.entropy_coeff = 0.0
        c.actor.clip_ratio_c = float('inf')
        c.actor.optim.total_training_steps = 3
        c.actor.fsdp_config.model_dtype = 'bfloat16'
        c.actor.fsdp_config.optimizer_offload = True
        c.actor.fsdp_config.reshard_after_forward = args.reshard_after_forward
        fp32_input = None
        if args.fp32_reference_from:
            assert args.attention == 'sdpa' and args.actor_source and not args.trim_shared_padding
            assert not args.paired_owner_source and not args.attention_check
            fp32_input = torch.load(args.fp32_reference_from, map_location='cpu', weights_only=True)
            c.actor.fsdp_config.model_dtype = 'float32'
            c.actor.fsdp_config.mixed_precision = dict(param_dtype='fp32', reduce_dtype='fp32', buffer_dtype='fp32')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision('highest')
        c.rollout.name = 'hf'
        c.rollout.n = 1
        c.rollout.tensor_model_parallel_size = 1
        c.rollout.log_prob_micro_batch_size_per_gpu = 1
        result['policy_loss_config'] = {k:c.actor[k] for k in
            ('clip_ratio', 'clip_ratio_low', 'clip_ratio_high', 'loss_agg_mode')}
        result['policy_loss_config']['clip_ratio_c'] = 'inf'
        worker = ActorRolloutRefWorker(c, 'actor_rollout')
        worker.init_model()
        installed_actor_type = type(worker.actor)
        model_module = importlib.import_module('transformers.models.qwen3_5.modeling_qwen3_5')
        result['transformers_model_source_sha256'] = hashlib.sha256(Path(model_module.__file__).read_bytes()).hexdigest()
        if fp32_input is not None:
            with torch.no_grad():
                for name, param in worker.actor_module_fsdp.named_parameters():
                    if param.requires_grad:
                        target = param.to_local() if hasattr(param, 'to_local') else param
                        target.copy_(fp32_input['before'][name].to(target.device, dtype=target.dtype))
            # The FLA chunk kernel accepts low precision only. The same HF
            # module already provides its native differentiable FP32 fallback.
            for layer in worker.actor_module_fsdp.modules():
                if isinstance(layer, model_module.Qwen3_5GatedDeltaNet):
                    layer.chunk_gated_delta_rule = model_module.torch_chunk_gated_delta_rule
            result['fp32_reference'] = dict(source=str(args.fp32_reference_from),
                source_sha256=hashlib.sha256(args.fp32_reference_from.read_bytes()).hexdigest(),
                gdn='original Transformers torch_chunk_gated_delta_rule',
                attention='original math SDPA', autocast=False, tf32=False,
                base_weights='same checkpoint at FP32; ordinary BF16 parameter rounding is included in the baseline error',
                dt_advantages='fixed saved BF16 run; no DT recomputation')
        if args.actor_source:
            spec = importlib.util.spec_from_file_location('pristine_pinned_actor', args.actor_source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            worker.actor = module.DataParallelPPOActor(c.actor, worker.actor_module_fsdp, worker.actor_optimizer)
        source = Path(args.actor_source or __import__('inspect').getfile(type(worker.actor)))
        result['actor_source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        result['ppo_core_sha256'] = hashlib.sha256((Path(os.environ['VERL_ROOT'])/'verl/trainer/ppo/core_algos.py').read_bytes()).hexdigest()

        def cpu_tensor(value):
            if hasattr(value, 'full_tensor'):
                value = value.full_tensor()
            return value.detach().cpu().clone()

        def trainable_state():
            return {name: cpu_tensor(value) for name, value in worker.actor_module_fsdp.named_parameters() if value.requires_grad}

        artifacts['before'] = trainable_state()
        fixture = json.loads((Path(os.environ['DT_RUNTIME_ROOT'])/'receipts/rollout-fixtures.json').read_text())
        row = fixture['tasks']['Sokoban']['rows'][-1]
        row = {k: torch.tensor(v) if k in ('input_ids', 'attention_mask', 'responses') else v for k, v in row.items()}
        pad = worker.tokenizer.pad_token_id
        if args.effective_input_tokens or args.response_tokens:
            width = row['responses'].numel()
            prompt = row['input_ids'][:-width][row['attention_mask'][:-width].bool()]
            actions = row['responses'][row['attention_mask'][-width:].bool()]
            original_count = actions.numel()
            filler = worker.tokenizer.encode(' context', add_special_tokens=False)[0]
            if args.response_tokens:
                assert args.response_tokens >= original_count
                actions = torch.cat((torch.full((args.response_tokens-original_count,), filler), actions))
            length = args.effective_input_tokens or prompt.numel()+actions.numel()
            fill = length-prompt.numel()-actions.numel()
            assert fill >= 0
            row['input_ids'] = torch.cat((torch.full((fill,), filler), prompt, actions))
            row['attention_mask'] = torch.ones(length, dtype=torch.long)
            row['responses'] = actions
            result['numerical_fixture'] = dict(synthetic_context_tokens=fill,
                synthetic_action_tokens=actions.numel()-original_count,
                original_action_tokens=original_count,
                scope='Explicit long numerical input; recorded reward is a test coefficient, not a reward claim for this synthetic trajectory')
        # Exercise the actual shared-padding optimization, including EOS/padding.
        row['input_ids'] = torch.cat((torch.full((args.left_padding,), pad), row['input_ids'], torch.full((19,), pad)))
        row['attention_mask'] = torch.cat((torch.zeros(args.left_padding, dtype=torch.long), row['attention_mask'], torch.zeros(19, dtype=torch.long)))
        row['responses'] = torch.cat((row['responses'], torch.full((19,), pad)))
        if fp32_input is None:
            from deltatrace_rollout import DeltaTraceRolloutProducer
            producer = DeltaTraceRolloutProducer(worker.actor_module_fsdp,
                eos_token_id=worker.tokenizer.eos_token_id, pad_token_id=pad)
            values = producer.attribute_episode([row], float(row['rewards']))[0]
            result['dt_report'] = producer.readout.last_report
        else:
            values = {name: fp32_input[name][0] for name in
                      ('dt_token_advantages', 'dt_q_estimates', 'dt_v_estimates')}
            torch.testing.assert_close(row['input_ids'], fp32_input['input_ids'][0], atol=0, rtol=0)
            torch.testing.assert_close(row['attention_mask'], fp32_input['attention_mask'][0], atol=0, rtol=0)
            torch.testing.assert_close(row['responses'], fp32_input['responses'][0], atol=0, rtol=0)
            torch.testing.assert_close(artifacts['before'],
                                       {k:v.float() for k,v in fp32_input['before'].items()}, atol=0, rtol=0)
        result['input_tokens'] = row['input_ids'].numel()
        result['effective_input_tokens'] = int(row['attention_mask'].sum())
        result['explicit_left_padding'] = args.left_padding
        data = {name: torch.stack([row[name]]*4).cuda() for name in ('input_ids', 'attention_mask', 'responses')}
        data['position_ids'] = (data['attention_mask'].cumsum(-1)-1).clamp_min(0)
        for name in ('dt_token_advantages', 'dt_q_estimates', 'dt_v_estimates'):
            data[name] = torch.stack([values[name]]*4).cuda()
            artifacts[name] = data[name].cpu()
        batch = DataProto(batch=TensorDict(data, batch_size=[4]), meta_info={'temperature':1.0,
            'global_token_num':[int(row['attention_mask'].sum())]*4})
        batch = compute_advantage(batch, adv_estimator='deltatrace', gamma=1.0)
        for name in ('input_ids', 'attention_mask', 'position_ids', 'responses', 'response_mask'):
            artifacts[name] = cpu_tensor(batch.batch[name])
        if args.attention_check:
            text_model = producer.runner.model.model.language_model
            saved_attention = text_model.config._attn_implementation
            artifacts['attention_check'] = {}
            try:
                for label, backend in [('fa_before', 'flash_attention_2'), ('math', 'sdpa'),
                                       ('fa_after', 'flash_attention_2')]:
                    text_model.set_attn_implementation(backend)
                    ctx = (torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
                           if backend == 'sdpa' else contextlib.nullcontext())
                    with ctx:
                        check = worker.compute_log_prob(batch)
                    artifacts['attention_check'][label] = cpu_tensor(check.batch['old_log_probs'])
            finally:
                text_model.set_attn_implementation(saved_attention)
        initial_optimizer = copy.deepcopy(worker.actor_optimizer.state_dict())
        initial_scheduler = copy.deepcopy(worker.actor_lr_scheduler.state_dict())
        cpu_rng, device_rng = torch.get_rng_state(), torch.cuda.get_rng_state()

        def run_updates(backend=args.attention):
            captured, metrics = {}, []
            forward = worker.actor._forward_micro_batch
            forward_records = []
            def record_forward(*a, **kw):
                entropy, log_prob = forward(*a, **kw)
                forward_records.append(cpu_tensor(log_prob))
                return entropy, log_prob
            optimizer_step = worker.actor._optimizer_step
            gradient_records = []
            def record_optimizer_step():
                gradient_records.append({name: cpu_tensor(value.grad)
                    for name, value in worker.actor_module_fsdp.named_parameters()
                    if value.requires_grad and value.grad is not None})
                return optimizer_step()
            worker.actor._forward_micro_batch = record_forward
            worker.actor._optimizer_step = record_optimizer_step
            ctx = (torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
                   if backend == 'sdpa' else contextlib.nullcontext())
            autocast = torch.autocast
            precision_ctx = (patch('torch.autocast', lambda device_type, **kw:
                             autocast(device_type=device_type, **{**kw, 'enabled':False}))
                             if fp32_input is not None else contextlib.nullcontext())
            try:
                with ctx, precision_ctx:
                    out = worker.compute_log_prob(batch)
                    batch.batch['old_log_probs'] = out.batch['old_log_probs'].cuda()
                    captured['old_log_probs'] = cpu_tensor(batch.batch['old_log_probs'])
                    forward_records.clear()
                    for index in range(2):
                        update = worker.update_actor(batch)
                        metrics.append(update.meta_info['metrics'])
                        captured['after_'+str(index)] = trainable_state()
            finally:
                worker.actor._forward_micro_batch = forward
                worker.actor._optimizer_step = optimizer_step
            captured['policy_forward_log_probs'] = forward_records
            captured['raw_gradients'] = gradient_records
            return captured, metrics

        current, result['updates'] = run_updates()
        artifacts.update(current)
        result['status'] = 'artifacts_ready'
        def restore_training_state():
            with torch.no_grad():
                for name, param in worker.actor_module_fsdp.named_parameters():
                    if param.requires_grad:
                        target = param.to_local() if hasattr(param, 'to_local') else param
                        target.copy_(artifacts['before'][name].to(target.device))
                    param.grad = None
            worker.actor_optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
            worker.actor_lr_scheduler.load_state_dict(copy.deepcopy(initial_scheduler))
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(device_rng)

        if args.repeat_installed:
            restore_training_state()
            artifacts['repeat_installed'], result['repeat_installed_updates'] = run_updates()
        if args.paired_owner_source:
            restore_training_state()
            spec = importlib.util.spec_from_file_location('paired_pristine_actor', args.paired_owner_source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            worker.actor = module.DataParallelPPOActor(c.actor, worker.actor_module_fsdp, worker.actor_optimizer)
            artifacts['paired_initial'] = trainable_state()
            torch.testing.assert_close(artifacts['before'], artifacts['paired_initial'], atol=0, rtol=0)
            reference, result['paired_updates'] = run_updates()
            artifacts['paired_owner'] = reference
            result['paired_owner_sha256'] = hashlib.sha256(args.paired_owner_source.read_bytes()).hexdigest()
            if args.head_only_comparison:
                assert args.trim_shared_padding
                restore_training_state()
                worker.actor = installed_actor_type(c.actor, worker.actor_module_fsdp, worker.actor_optimizer)
                trim = os.environ['VERL_TRIM_SHARED_PADDING']
                try:
                    os.environ['VERL_TRIM_SHARED_PADDING'] = '0'
                    artifacts['paired_untrimmed_head'], result['head_only_updates'] = run_updates()
                finally:
                    os.environ['VERL_TRIM_SHARED_PADDING'] = trim
                    worker.actor = module.DataParallelPPOActor(c.actor, worker.actor_module_fsdp, worker.actor_optimizer)
            if args.trim_shared_padding and args.owner_math:
                # Measure the owner's own FA/math error on identical weights,
                # DT advantages and optimizer state; do not inflate it to fit
                # any difference introduced by the shared-padding patch.
                restore_training_state()
                text_model = producer.runner.model.model.language_model
                saved_attention = text_model.config._attn_implementation
                try:
                    text_model.set_attn_implementation('sdpa')
                    artifacts['paired_owner_math'], result['paired_math_updates'] = run_updates('sdpa')
                finally:
                    text_model.set_attn_implementation(saved_attention)
                result['comparison_tolerance'] = 'owner FA/math max-absolute and L2 error, no multiplier'
            elif not args.trim_shared_padding:
                result['comparison_performed'] = True
                result['comparison_tolerance'] = {'atol': 0, 'rtol': 0}
                torch.testing.assert_close(current, reference, atol=0, rtol=0)
                for actual, expected in zip(result['updates'], result['paired_updates']):
                    for key in actual:
                        if key.startswith('actor/'):
                            assert actual[key] == expected[key], (key, actual[key], expected[key])
                result['status'] = 'passed'
            else:
                result['comparison_performed'] = True
                result['comparison_tolerance'] = 'Direct paired owner measurements only; no borrowed FP32 error bound'
        if args.previous_padding_source:
            assert args.trim_shared_padding, 'Continuity check targets the production trimmed path'
            restore_training_state()
            worker.actor = installed_actor_type(c.actor, worker.actor_module_fsdp, worker.actor_optimizer)
            node = next(n for n in ast.parse(args.previous_padding_source.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'apply_mask_to_padding_states')
            namespace = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<archived original mask>', 'exec'), namespace)
            fixed_mask = model_module.apply_mask_to_padding_states
            try:
                model_module.apply_mask_to_padding_states = namespace[node.name]
                previous, result['previous_mask_updates'] = run_updates()
            finally:
                model_module.apply_mask_to_padding_states = fixed_mask
            artifacts['previous_mask_policy'] = previous
            mask = artifacts['response_mask'].bool()
            for key in ('after_0', 'after_1', 'raw_gradients'):
                torch.testing.assert_close(current[key], previous[key], atol=0, rtol=0)
            torch.testing.assert_close(current['old_log_probs'][mask], previous['old_log_probs'][mask], atol=0, rtol=0)
            torch.testing.assert_close(torch.cat(current['policy_forward_log_probs'])[mask.repeat(2, 1)],
                                       torch.cat(previous['policy_forward_log_probs'])[mask.repeat(2, 1)], atol=0, rtol=0)
            result['production_mask_backport_continuity'] = dict(status='passed', atol=0, rtol=0,
                previous_sha256=hashlib.sha256(args.previous_padding_source.read_bytes()).hexdigest())
    except Exception as exc:
        result.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        result['seconds'] = time.perf_counter()-started
        if torch.cuda.is_initialized():
            result['peak_allocated'] = torch.cuda.max_memory_allocated()
            result['peak_reserved'] = torch.cuda.max_memory_reserved()
        args.output.write_text(json.dumps(result, indent=2, default=str)+'\n')
        torch.save(artifacts, args.artifacts)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    if result['status'] not in ('artifacts_ready', 'passed'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
