"""Approved reward-event targets traced by the official EOS DeltaTrace runner.

This module does not score environments, reconstruct generated tokens, train a
critic, or implement model/cache transitions. Qwen's original head estimates
declared reward categories; the official DT runner owns endpoint evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import torch

from counterfactual import reward_event_credit_for_episode
from deltatrace_credit import trace_token_attribution


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

    def readout_token_budget(self, tokenizer: Any, max_steps: int) -> int:
        """Exact worst-case query plus target length, without a model call."""
        return 1 + max(len(self.query_ids(tokenizer, current_step=i, event_step=k,
                                          max_steps=max_steps))
                       for i in range(max_steps) for k in range(i, max_steps))


class EventRatioReadout:
    """One official EOS trace per response/future reward event, not per token.

    DT's signed vector estimates the individual deletion log-prob effects in
    PLAN.md. It is not a measured collection of leave-one-out forward passes.
    All tokens keep their own signed entries; no span broadcast or O routing.
    """

    def __init__(self, runner: Any, tokenizer: Any, *, task: str, max_steps: int,
                 packed_answer_targets: Any, max_length: int = 32768):
        if max_steps < 1:
            raise ValueError('Horizon must be positive')
        if tokenizer.eos_token_id is None:
            raise ValueError('EOS attribution requires the checkpoint EOS token')
        self.runner, self.tokenizer = runner, tokenizer
        self.alphabet = RewardAlphabet.for_task(task)
        self.max_steps, self.max_length = max_steps, max_length
        self.packed_answer_targets = packed_answer_targets
        self.last_report: dict[str, Any] = {}

    @torch.no_grad()
    def episode(self, rows: list[dict[str, Any]]) -> list[dict[str, torch.Tensor]]:
        started = time.perf_counter()
        device = self.runner.model.lm_head.weight.device
        events = [row for row in rows if bool(row['active_masks'])]
        labels = self.alphabet.label_ids(self.tokenizer)
        observed = [self.alphabet.observed_index(float(row['rewards'])) for row in events]
        steps = [int(row['env_step']) for row in events]
        if not rows or steps != sorted(set(steps)):
            raise ValueError('Episode must retain unique increasing reward event identities')
        if any(row['traj_uid'] != rows[0]['traj_uid'] for row in rows):
            raise ValueError('Cannot attribute mixed trajectories')
        log_ratios = []
        report = dict(task=self.alphabet.task, policy_tokens=0,
                      nonzero_reward_events=sum(float(r['rewards']) != 0 for r in events),
                      finite_trace_calls=0, max_readout_length=0, actual_row_lengths=[],
                      traces=[], max_length=self.max_length,
                      ratio_source='official_eos_dt_signed_attribution_estimate',
                      reference_token_samples=0, per_token_probability_queries=0)

        for row in rows:
            response = row['responses']
            width = response.numel()
            matrix = torch.zeros((len(events), width), dtype=torch.float32)
            log_ratios.append(matrix)
            if not bool(row['active_masks']):
                continue
            attention = row['attention_mask'].bool()
            positions = attention[-width:].nonzero().flatten().tolist()
            if positions != list(range(len(positions))):
                raise ValueError('Original generated response must have right padding only')
            prompt = row['input_ids'][:-width][attention[:-width]].to(device)
            actions = response[:len(positions)].to(device)
            if not torch.equal(row['input_ids'][-width:].cpu(), response.cpu()):
                raise ValueError('Original rollout input and response token identities differ')
            report['policy_tokens'] += len(positions)
            report['actual_row_lengths'].append(prompt.numel() + actions.numel())
            if not positions:
                continue
            for k, event in enumerate(events):
                if int(event['env_step']) < int(row['env_step']) or float(event['rewards']) == 0:
                    continue
                query = torch.tensor(self.alphabet.query_ids(
                    self.tokenizer, current_step=int(row['env_step']),
                    event_step=int(event['env_step']), max_steps=self.max_steps,
                ), device=device, dtype=torch.long)
                target = torch.tensor([labels[observed[k]]], device=device, dtype=torch.long)
                selected = torch.cat((prompt, actions, query, target))[None, :]
                length = selected.shape[1]
                if length > self.max_length:
                    raise ValueError(f'Reward readout context {length} exceeds cap {self.max_length}; no silent truncation')
                reference = selected.clone()
                start, end = prompt.numel(), prompt.numel() + actions.numel()
                # The original row prompt includes *past* observations. It and
                # the event query/label are identical at both endpoints. Future
                # observations and future generated actions never enter this
                # input. Only the current generated source tokens become EOS.
                reference[:, start:end] = self.tokenizer.eos_token_id
                signed, _, detail = trace_token_attribution(
                    self.runner, reference, selected,
                    {'target_ids': target.cpu(), 'prompt_length': length - 1}, [0],
                    packed_answer_targets=self.packed_answer_targets,
                    outcome_token_ids=labels,
                )
                matrix[k, :len(positions)] = signed[0, start:end].cpu()
                # Keep per-event attribution separate until after expm1.
                report['finite_trace_calls'] += 1
                report['max_readout_length'] = max(report['max_readout_length'], length)
                report['traces'].append(dict(
                    source_step=int(row['env_step']), event_step=int(event['env_step']),
                    context_tokens=length, query_tokens=query.numel(),
                    root_effect=detail['root_effect'], signed_sum=detail['policy_credit_signed_sum'],
                    conservation_residual=detail['conservation_residual'],
                    conservation_tolerance=detail['conservation_tolerance'],
                    conservation_verified=detail['conservation_verified'],
                    seconds=detail.get('complete_attribution_seconds_with_diagnostics'),
                    peak_allocated=detail.get('peak_allocated'), peak_reserved=detail.get('peak_reserved'),
                    source_log_ratio_min=float(matrix[k, :len(positions)].min()),
                    source_log_ratio_max=float(matrix[k, :len(positions)].max()),
                ))
                print(f"[DT EOS event] step={int(row['env_step'])} event={int(event['env_step'])} "
                      f"tokens={len(positions)} length={length} "
                      f"seconds={report['traces'][-1]['seconds']} "
                      f"conservation_verified={detail['conservation_verified']} "
                      f"residual={detail['conservation_residual']}", flush=True)
        result = reward_event_credit_for_episode(rows, log_ratios)
        report['seconds'] = time.perf_counter() - started
        report['nonzero_advantages'] = sum(int(r['dt_token_advantages'].count_nonzero()) for r in result)
        report['conservation_failures'] = sum(not trace['conservation_verified'] for trace in report['traces'])
        self.last_report = report
        return result
