"""Approved event-target encoding and same-prefix native DT readout.

This module does not score environments, reconstruct generated tokens, train a
critic, or implement model/cache transitions. Qwen's original head estimates
declared reward categories; the official DT runner owns endpoint evaluation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import time
from collections import defaultdict
from typing import Any

import torch

from counterfactual import reward_event_credit_for_episode


@dataclass(frozen=True)
class RewardAlphabet:
    task: str
    values: tuple[float, ...]
    meanings: tuple[str, ...]

    @classmethod
    def for_task(cls, task: str) -> 'RewardAlphabet':
        if task in ('Webshop', 'AppWorld'):
            return cls(task, (0.0, 10.0), (
                'zero reward, including an interaction that never happens after termination',
                'successful task completion with official training reward 10',
            ))
        if task == 'Sokoban':
            return cls(task, (-0.1, 0.0, 10.9), (
                'one penalized interaction without solving the one-box puzzle (reward -0.1)',
                'no interaction because the episode already terminated (reward 0)',
                'push the single box onto its target and finish (reward 10.9)',
            ))
        raise ValueError(f'No approved official reward alphabet for {task}')

    def observed_index(self, reward: float) -> int:
        matches = [i for i, value in enumerate(self.values)
                   if math.isclose(reward, value, rel_tol=0.0, abs_tol=1e-6)]
        if len(matches) != 1:
            raise ValueError(f'Official reward {reward} outside {self.task} alphabet {self.values}')
        return matches[0]

    def label_ids(self, tokenizer: Any) -> list[int]:
        ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in range(len(self.values))]
        if any(len(value) != 1 for value in ids) or len({value[0] for value in ids}) != len(ids):
            raise ValueError('Reward readout requires distinct single-token numeric labels')
        return [value[0] for value in ids]

    def query_ids(self, tokenizer: Any, *, current_step: int, event_step: int,
                  max_steps: int) -> list[int]:
        if not 0 <= current_step <= event_step < max_steps:
            raise ValueError('Reward query must refer to a future event inside the rollout horizon')
        legend = '; '.join(f'{i}: {meaning}' for i, meaning in enumerate(self.meanings))
        # Deliberately contains no sampled future observation, action, reward or
        # stopping time. The same query is appended at both token endpoints.
        query = (
            '<|im_end|>\n<|im_start|>user\n'
            f'Reward forecast for {self.task}. The preceding text is the exact agent prefix, '
            'possibly ending inside an unfinished response. Do not treat this forecast request '
            'as an environment action. Imagine completing that response and continuing with '
            'this same policy, sampling tokens at temperature 1 with no top-k or top-p truncation. '
            f'The current interaction is {current_step + 1}; the episode allows {max_steps} interactions. '
            f'Predict ONLY the reward at interaction {event_step + 1}, including the possibility '
            f'that it never occurs. Categories: {legend}. Return only the category number.'
            '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        )
        return tokenizer.encode(query, add_special_tokens=False)


class EventRatioReadout:
    """Read event probabilities before/after each actual rollout token."""

    def __init__(self, runner: Any, tokenizer: Any, *, task: str, max_steps: int,
                 max_length: int = 32768, event_batch_size: int = 4):
        if max_steps < 1:
            raise ValueError('Horizon must be positive')
        self.runner, self.tokenizer = runner, tokenizer
        self.alphabet = RewardAlphabet.for_task(task)
        self.max_steps, self.max_length = max_steps, max_length
        if event_batch_size < 1:
            raise ValueError('Event readout microbatch must be positive')
        self.event_batch_size = event_batch_size
        self.last_report: dict[str, Any] = {}

    @torch.no_grad()
    def episode(self, rows: list[dict[str, Any]]) -> list[dict[str, torch.Tensor]]:
        started = time.perf_counter()
        device = self.runner.model.lm_head.weight.device
        events = [row for row in rows if bool(row['active_masks'])]
        labels = torch.tensor(self.alphabet.label_ids(self.tokenizer), device=device)
        observed = [self.alphabet.observed_index(float(row['rewards'])) for row in events]
        # Zero observed reward contributes exactly zero to both sampled Q/V;
        # its conditional probability remains in the categorical normalizer.
        log_ratios = []
        report = dict(task=self.alphabet.task,
                      policy_tokens=0, nonzero_reward_events=sum(float(r['rewards']) != 0 for r in events),
                      native_forward_calls=0, prefix_prefill_tokens=0, factual_decode_tokens=0,
                      query_tokens=0, max_readout_length=0, actual_row_lengths=[],
                      boundary_readouts=0,
                      max_length=self.max_length, ratio_source='native_dt_pre_post_event_log_ratio')

        def forward(ids, cache=None):
            report['native_forward_calls'] += 1
            return self.runner.forward_prefix(ids, past_key_values=cache)

        for row in rows:
            row_started = time.perf_counter()
            response = row['responses']
            width = response.numel()
            matrix = torch.zeros((len(events), width), dtype=torch.float32)
            log_ratios.append(matrix)
            if not bool(row['active_masks']):
                continue
            attention = row['attention_mask'].bool()
            mask = attention[-width:]
            positions = mask.nonzero().flatten().tolist()
            if positions != list(range(len(positions))):
                raise ValueError('Original generated response must have right padding only')
            prompt = row['input_ids'][:-width][attention[:-width]].to(device)[None, :]
            action_ids = response[:len(positions)].to(device)
            report['policy_tokens'] += len(positions)
            report['actual_row_lengths'].append(prompt.numel() + len(positions))
            future = [k for k, event in enumerate(events)
                      if int(event['env_step']) >= int(row['env_step']) and float(event['rewards']) != 0]
            if not future or not positions:
                continue
            print(f"[DT event readout] step={int(row['env_step'])} policy_tokens={len(positions)} "
                  f"future_nonzero_events={len(future)}", flush=True)
            queries = {k: torch.tensor(self.alphabet.query_ids(
                self.tokenizer, current_step=int(row['env_step']),
                event_step=int(events[k]['env_step']), max_steps=self.max_steps,
            ), device=device)[None, :] for k in future}
            # Reuse the identical query prefix across future event indices.
            # Fork before the differing event index; no outcome from one
            # forecast conditions another forecast.
            shared_query_length = 0
            if len(future) > 1:
                token_lists = [queries[k][0].tolist() for k in future]
                for tokens in zip(*token_lists):
                    if len(set(tokens)) != 1:
                        break
                    shared_query_length += 1
                shared_query_length = min(shared_query_length, min(map(len, token_lists))-1)
            # Exact-length buckets avoid padding, altered GDN cache semantics,
            # and per-length compilation. HF owns cache batch expansion.
            by_length = defaultdict(list)
            for k in future:
                by_length[queries[k].numel()].append(k)
            groups = [keys[start:start+self.event_batch_size]
                      for keys in by_length.values()
                      for start in range(0, len(keys), self.event_batch_size)]
            length = prompt.numel() + len(positions) + max(q.numel() for q in queries.values())
            if length > self.max_length:
                raise ValueError(f'Reward readout context {length} exceeds cap {self.max_length}; no silent truncation')
            report['max_readout_length'] = max(report['max_readout_length'], length)
            report['prefix_prefill_tokens'] += prompt.numel()
            out = forward(prompt)
            cache = out.past_key_values
            del out

            def read_boundary(factual_cache):
                # The query reads a prefix; it must never become part of it.
                # Before a_i this estimates the policy-marginal event law;
                # after a_i it estimates the action-conditional event law.
                report['boundary_readouts'] += 1
                query_base = factual_cache
                if shared_query_length:
                    common = queries[future[0]][:, :shared_query_length]
                    common_output = forward(common, deepcopy(factual_cache))
                    query_base = common_output.past_key_values
                    del common_output
                    report['query_tokens'] += common.numel()
                values = {}
                for group in groups:
                    report['native_forward_calls'] += 1
                    query = torch.cat([queries[k][:, shared_query_length:] for k in group])
                    report['query_tokens'] += query.numel()
                    query_cache = deepcopy(query_base)
                    if len(group) > 1:
                        # HF owns expansion of both KV and recurrent states.
                        query_cache.reorder_cache(torch.zeros(len(group), dtype=torch.long, device=device))
                    lp = self.runner.read_outcomes(query, labels, past_key_values=query_cache)
                    for j, k in enumerate(group):
                        values[k] = lp[j, observed[k]]
                    del query_cache
                return torch.stack([values[k] for k in future])

            before = read_boundary(cache)
            for i, actual in enumerate(action_ids.tolist()):
                # Advance only the actual rollout token. No alternate-token
                # sampling, enumeration, replacement or reference forward.
                out = forward(torch.tensor([[actual]], device=device), cache)
                cache = out.past_key_values
                del out
                after = read_boundary(cache)
                matrix[future, i] = (after - before).cpu()
                # Only within this response: the next row includes a real
                # environment transition and must have a fresh initial read.
                before = after
                report['factual_decode_tokens'] += 1
            del cache, before, after
            print(f"[DT event readout] step={int(row['env_step'])} "
                  f"seconds={time.perf_counter()-row_started:.3f}", flush=True)
        report['seconds'] = time.perf_counter() - started
        report['nonzero_advantages'] = 0
        result = reward_event_credit_for_episode(rows, log_ratios)
        report['nonzero_advantages'] = sum(int(r['dt_token_advantages'].count_nonzero()) for r in result)
        self.last_report = report
        return result
