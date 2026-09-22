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

from counterfactual import policy_marginal_log_ratio, reward_event_credit_for_episode


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
    """Native, cached, single-token counterfactuals on original rollout IDs."""

    def __init__(self, runner: Any, tokenizer: Any, *, task: str, max_steps: int,
                 reference_samples: int = 1, max_length: int = 32768, event_batch_size: int = 4):
        if reference_samples < 1 or max_steps < 1:
            raise ValueError('Reference sample count and horizon must be positive')
        self.runner, self.tokenizer = runner, tokenizer
        self.alphabet = RewardAlphabet.for_task(task)
        self.max_steps, self.reference_samples, self.max_length = max_steps, reference_samples, max_length
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
        report = dict(task=self.alphabet.task, reference_samples=self.reference_samples,
                      policy_tokens=0, nonzero_reward_events=sum(float(r['rewards']) != 0 for r in events),
                      native_forward_calls=0, prefix_prefill_tokens=0, factual_decode_tokens=0,
                      query_tokens=0, max_readout_length=0, actual_row_lengths=[],
                      identical_reference_prefixes=0,
                      max_length=self.max_length, ratio_source='native_dt_categorical_endpoint_difference')

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
            if not future:
                continue
            print(f"[DT event readout] step={int(row['env_step'])} policy_tokens={len(positions)} "
                  f"future_nonzero_events={len(future)}", flush=True)
            queries = {k: torch.tensor(self.alphabet.query_ids(
                self.tokenizer, current_step=int(row['env_step']),
                event_step=int(events[k]['env_step']), max_steps=self.max_steps,
            ), device=device)[None, :] for k in future}
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
            cache, next_logits = out.past_key_values, out.logits[:, -1].float()
            del out
            for i, actual in enumerate(action_ids.tolist()):
                # This is the very same old-policy distribution as DT rollout:
                # temperature=1, no truncation, actor update has not begun.
                references = torch.multinomial(next_logits.softmax(-1), self.reference_samples,
                                               replacement=True)[0].tolist()
                if all(b == actual for b in references):
                    # All endpoint differences are exactly zero. No target
                    # scoring, cache copy or finite replay can add information.
                    branch = forward(torch.tensor([[actual]], device=device), cache)
                    cache, next_logits = branch.past_key_values, branch.logits[:, -1].float()
                    del branch
                    report['factual_decode_tokens'] += 1
                    report['identical_reference_prefixes'] += 1
                    continue
                candidates = list(dict.fromkeys([actual, *references]))
                scores = {}
                actual_cache = actual_logits = None
                for candidate in candidates:
                    ids = torch.tensor([[candidate]], device=device)
                    # HF cache mutates in place. Fork before every branch; a
                    # query or reference must never contaminate the factual h_i.
                    branch = forward(ids, deepcopy(cache))
                    candidate_cache = branch.past_key_values
                    if candidate == actual:
                        actual_cache = candidate_cache
                        actual_logits = branch.logits[:, -1].float()
                    del branch
                    values = {}
                    for group in groups:
                        report['native_forward_calls'] += 1
                        query = torch.cat([queries[k] for k in group])
                        report['query_tokens'] += query.numel()
                        query_cache = deepcopy(candidate_cache)
                        if len(group) > 1:
                            # Qwen's hybrid cache supports beam reordering for
                            # both KV and recurrent layers. Its installed
                            # linear layers do not expose batch_repeat_interleave.
                            query_cache.reorder_cache(torch.zeros(len(group), dtype=torch.long, device=device))
                        lp = self.runner.read_outcomes(
                            query, labels, past_key_values=query_cache)
                        for j, k in enumerate(group):
                            values[k] = lp[j, observed[k]]
                        del query_cache
                    scores[candidate] = torch.stack([values[k] for k in future])
                    del candidate_cache
                contrasts = torch.stack([scores[actual] - scores[b] for b in references], dim=-1)
                weights = torch.full_like(contrasts, -math.log(self.reference_samples))
                d = policy_marginal_log_ratio([(contrasts, weights)])
                matrix[future, i] = d.cpu()
                cache, next_logits = actual_cache, actual_logits
                report['factual_decode_tokens'] += 1
            del cache, next_logits
            print(f"[DT event readout] step={int(row['env_step'])} "
                  f"seconds={time.perf_counter()-row_started:.3f}", flush=True)
        report['seconds'] = time.perf_counter() - started
        report['nonzero_advantages'] = 0
        result = reward_event_credit_for_episode(rows, log_ratios)
        report['nonzero_advantages'] = sum(int(r['dt_token_advantages'].count_nonzero()) for r in result)
        self.last_report = report
        return result
