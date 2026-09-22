"""Measure original HF rollout with FSDP2 resharding on/off on identical input.

Uses a synthetic long prefix and deterministic decoding for a capacity/cost
comparison, not task training or success-rate evaluation. Only the public FSDP
setter changes; the pinned worker and HF generation implementation are reused.
"""
import argparse
import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.distributed.fsdp import FSDPModule
from verl import DataProto
from verl.workers.fsdp_workers import ActorRolloutRefWorker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare-active-mask', action='store_true',
                        help='Compare original all-row generation with skipping finished rows.')
    args = parser.parse_args()
    result = dict(scope=__doc__, prompt_tokens=32256, response_cap=512,
                  context_cap=32768, rollout_batch=4, cases=[],
                  allocator_config=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))
    if args.compare_active_mask:
        result['scope'] = ('Same pinned HF/FSDP2 owner, real Qwen weights, fixed 32k capacity fixture; '
                           'compare all-row generation with collector active-mask transport. '
                           'No claim about task success or complete iteration throughput.')
    try:
        torch.manual_seed(2026)
        torch.use_deterministic_algorithms(True)
        cfg = OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
        c = cfg.actor_rollout_ref
        c.model.path = os.environ['MODEL_PATH']
        c.model.lora_rank, c.model.lora_alpha = 1, 2
        c.model.trust_remote_code = True
        c.actor.strategy = 'fsdp2'
        c.actor.ppo_mini_batch_size = 4
        c.actor.ppo_micro_batch_size_per_gpu = 1
        c.actor.use_torch_compile = False
        c.actor.optim.total_training_steps = 3
        c.actor.fsdp_config.model_dtype = 'bfloat16'
        c.actor.fsdp_config.reshard_after_forward = True
        c.rollout.name = 'hf'
        c.rollout.n = 1
        c.rollout.tensor_model_parallel_size = 1
        c.rollout.micro_batch_size = 4
        c.rollout.prompt_length = result['prompt_tokens']
        c.rollout.response_length = result['response_cap']
        worker = ActorRolloutRefWorker(c, 'actor_rollout')
        worker.init_model()
        result['generation_eos_token_id'] = worker.generation_config.eos_token_id
        fixture = json.loads((Path(os.environ['DT_RUNTIME_ROOT'])/'receipts/rollout-fixtures.json').read_text())
        row = fixture['tasks']['Sokoban']['rows'][-1]
        width = len(row['responses'])
        original = torch.tensor(row['input_ids'][:-width])[torch.tensor(row['attention_mask'][:-width]).bool()]
        filler_id = worker.tokenizer.encode(' context', add_special_tokens=False)[0]
        fill = result['prompt_tokens']-original.numel()
        assert fill > 0
        ids = torch.cat((torch.full((fill,), filler_id), original)).repeat(4, 1)
        mask = torch.ones_like(ids)
        prompts = DataProto(batch=TensorDict(dict(input_ids=ids, attention_mask=mask,
            position_ids=mask.cumsum(-1)-1), batch_size=[4]), meta_info={'do_sample': False})
        modules = [m for m in worker.actor_module_fsdp.modules() if isinstance(m, FSDPModule)]
        assert modules
        reference = None
        # Default FSDP2 root does not reshard; change only descendant settings.
        cases = [(True, None), (False, None)] if not args.compare_active_mask else [
            (False, None), (False, [True, False, True, False]), (False, [False]*4)]
        for enabled, active in cases:
            for m in modules[1:]:
                m.set_reshard_after_forward(enabled, recurse=False)
            case = dict(reshard_after_forward=enabled, runs=[])
            owner_active_reference = None
            if active is not None and any(active):
                # Same-shape owner comparison: ordinary HF on these exact rows,
                # with no mask extension. Cross-batch long greedy continuations
                # can diverge due to the owner's own BF16 rounding; record that
                # difference, rather than inventing a bitwise cross-batch gate.
                prompts.non_tensor_batch.pop('rollout_active_mask', None)
                selected_prompts = prompts.select_idxs(np.flatnonzero(active))
                owner_active_reference = worker.generate_sequences(selected_prompts).batch['responses'].cpu()
                case['original_owner_selected_rows'] = int(sum(active))
                case['original_owner_changed_tokens_vs_batch4'] = int(
                    (owner_active_reference != reference[torch.tensor(active)]).sum())
            if active is None:
                prompts.non_tensor_batch.pop('rollout_active_mask', None)
            else:
                prompts.non_tensor_batch['rollout_active_mask'] = np.array(active, dtype=bool)
            if args.compare_active_mask:
                case['active_rows'] = active or [True]*4
            result['cases'].append(case)
            for repeat in range(2):
                for m in reversed(modules):
                    m.reshard()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                output = worker.generate_sequences(prompts)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                responses = output.batch['responses'].cpu()
                effective = output.batch['attention_mask'][:, -result['response_cap']:].sum(-1).tolist()
                if reference is None:
                    reference = responses.clone()
                elif owner_active_reference is not None:
                    torch.testing.assert_close(responses[torch.tensor(active)], owner_active_reference, rtol=0, atol=0)
                else:
                    selected = torch.tensor(active if active is not None else [True]*4)
                    torch.testing.assert_close(responses[selected], reference[selected], rtol=0, atol=0)
                if active is not None:
                    assert responses[~torch.tensor(active)].eq(worker.tokenizer.pad_token_id).all()
                assert output.batch['input_ids'].shape == (4, 32768)
                rec = dict(repeat=repeat, warmup=repeat == 0, seconds=elapsed,
                    actual_response_lengths=effective,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    response_sha256=hashlib.sha256(responses.numpy().tobytes()).hexdigest())
                case['runs'].append(rec)
                args.output.write_text(json.dumps(result, indent=2)+'\n')
                print('GENERATION_COST', enabled, rec, flush=True)
            case['warm_seconds'] = statistics.median(r['seconds'] for r in case['runs'][1:])
        key = 'warm_ratio_active2_over_all4' if args.compare_active_mask else 'warm_ratio_false_over_true'
        result[key] = result['cases'][1]['warm_seconds']/result['cases'][0]['warm_seconds']
        result['active_generated_ids_exactly_equal_to_same_shape_owner' if args.compare_active_mask else 'all_generated_ids_exactly_equal'] = True
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
