"""Real Qwen/DT pre/post event readout and native cache/full-forward comparison.

No model-generated task-performance or training claim is made by this probe.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(os.environ['DT_ROOT'])
    env = json.loads(Path(os.environ['DT_ENVIRONMENT_JSON']).read_text())['qwen35']
    sys.path[:0] = [str(root), env['official_root'], str(root / 'clean/qwen35'), env['ft_extension_root']]
    os.environ.update(env.get('runtime_environment', {}))
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from profiles.official import make_qwen35_runner
    from finite_fla_gpu import make_compiled_finite_pullback, verify_native_sources
    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256
    from reward_readout import RewardAlphabet

    verify_native_sources(env['native_stage_source_sha256'])
    tokenizer = AutoTokenizer.from_pretrained(env['checkpoint'], local_files_only=True)
    print('loading existing Qwen checkpoint', flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        env['checkpoint'], dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map={'': 'cuda:0'}, local_files_only=True,
    ).eval()
    runner = make_qwen35_runner(
        model, VendorFAFiniteP1BF16D256(env['finite_library'], env['finite_library_sha256']),
        make_compiled_finite_pullback(reuse_scalar_products=False),
    )
    result = dict(scope='pre/post event readout and cache comparison; not task training',
                  model=env['checkpoint'], context_cap=32768, prefix_task='Sokoban',
                  task_alphabets_only=True, attention='sdpa', tasks={})
    # Prefix from an actual official Sokoban state. The intervention is the
    # first actual token of the disclosed scripted response, not a new reference.
    from verify_task_reward_events import sokoban_events
    transitions = sokoban_events()
    observation = str(transitions[0][0])
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': observation}], tokenize=True,
        add_generation_prompt=True, return_dict=True,
    )['input_ids']
    prefix = torch.tensor([prompt], device='cuda')
    actual = tokenizer.encode(transitions[0][1], add_special_tokens=False)[0]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.no_grad():
        prefill = runner.forward_prefix(prefix)
        base = prefill.past_key_values
        del prefill
        for task in ['Sokoban', 'Webshop', 'AppWorld']:
            alphabet = RewardAlphabet.for_task(task)
            labels = torch.tensor(alphabet.label_ids(tokenizer), device='cuda')
            query = torch.tensor([alphabet.query_ids(tokenizer, current_step=0, event_step=0, max_steps=15)], device='cuda')
            cached, full = [], []
            for token_ids in [[], [actual]]:
                branch_cache = deepcopy(base)
                if token_ids:
                    branch = runner.forward_prefix(torch.tensor([token_ids], device='cuda'), past_key_values=branch_cache)
                    branch_cache = branch.past_key_values
                    del branch
                cached.append(runner.read_outcomes(query, labels, past_key_values=branch_cache)[0])
                complete = torch.cat((prefix, torch.tensor([token_ids], dtype=torch.long, device='cuda'), query), dim=1)
                full.append(runner.read_outcomes(complete, labels)[0])
                del branch_cache
            cached, full = torch.stack(cached), torch.stack(full)
            torch.testing.assert_close(cached.exp().sum(-1), torch.ones(2, device='cuda'))
            assert torch.isfinite(cached).all()
            result['tasks'][task] = dict(
                values=alphabet.values, cached_probabilities=cached.exp().tolist(),
                full_probabilities=full.exp().tolist(),
                cached_full_max_logp_difference=float((cached-full).abs().max()),
                pre_post_log_ratios=(cached[1]-cached[0]).tolist(),
                boundary_order=['before_actual_token', 'after_actual_token'],
                query_tokens=query.numel(), actual_context_tokens=complete.numel(),
            )
            print(task, json.dumps(result['tasks'][task]), flush=True)
        # Fresh base branch after every query must still reproduce the same
        # prefix state; mutation in place would change its sequence length.
        assert base.get_seq_length() == prefix.shape[1]
        del base
    result.update(seconds=time.perf_counter()-started,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(), status='completed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
