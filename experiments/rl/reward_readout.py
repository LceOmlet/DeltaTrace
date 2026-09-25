"""Complete-return targets traced by the official EOS DeltaTrace runner.

This module does not score environments, reconstruct generated tokens, train a
critic, or implement model/cache transitions. Qwen's original head estimates
declared return categories; the official DT runner owns endpoint evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import string
import time
from typing import Any

import torch

from counterfactual import episode_returns, return_credit_for_episode
from deltatrace_credit import trace_token_attribution


@dataclass(frozen=True)
class RewardAlphabet:
    task: str
    values: tuple[float, ...]
    meanings: tuple[str, ...]

    @classmethod
    def for_task(cls, task: str, max_steps: int = 15) -> 'RewardAlphabet':
        if task in ('Webshop', 'AppWorld'):
            return cls(task, (0.0, 10.0), (
                '0 (no subsequent successful completion)',
                '10 (subsequent successful task completion)',
            ))
        if task == 'Sokoban':
            values = (0.0,) + tuple(-n / 10 for n in range(1, max_steps + 1)) + tuple(
                11 - n / 10 for n in range(1, max_steps + 1))
            return cls(task, values, tuple(f'{value:g}' for value in values))
        raise ValueError(f'No approved official reward alphabet for {task}')

    def observed_index(self, reward: float) -> int:
        matches = [i for i, value in enumerate(self.values)
                   if math.isclose(reward, value, rel_tol=0.0, abs_tol=1e-6)]
        if len(matches) != 1:
            raise ValueError(f'Official cumulative return {reward} outside {self.task} alphabet {self.values}')
        return matches[0]

    def labels(self) -> str:
        # Qwen splits numbers >=10; each category must still be ONE target token.
        labels = string.digits + string.ascii_uppercase + string.ascii_lowercase
        if len(self.values) > len(labels):
            raise ValueError('Return alphabet exceeds the available distinct single-token labels')
        return labels[:len(self.values)]

    def label_ids(self, tokenizer: Any) -> list[int]:
        ids = [tokenizer.encode(label, add_special_tokens=False) for label in self.labels()]
        if any(len(value) != 1 for value in ids) or len({value[0] for value in ids}) != len(ids):
            raise ValueError('Return readout requires distinct single-token labels')
        return [value[0] for value in ids]

    def query_ids(self, tokenizer: Any, *, current_step: int,
                  max_steps: int) -> list[int]:
        if not 0 <= current_step < max_steps:
            raise ValueError('Return query must refer to a current step inside the rollout horizon')
        legend = '; '.join(f'{label}: {meaning}' for label, meaning in zip(self.labels(), self.meanings))
        reward_rule = ('Each executed interaction costs -0.1; solving the single-box puzzle adds 11 '
                       'on that interaction and ends the episode. ' if self.task == 'Sokoban' else
                       'Successful completion gives 10 once; all other interactions give 0. ')
        # Deliberately contains no sampled future observation, action, reward or
        # stopping time. The same query is appended at both token endpoints.
        query = (
            '<|im_end|>\n<|im_start|>user\n'
            f'Future cumulative return forecast for {self.task}. The preceding text is the exact agent prefix, '
            'possibly ending inside an unfinished response. Do not treat this forecast request '
            'as an environment action. Imagine completing that response and continuing with '
            'this same policy, sampling tokens at temperature 1 with no top-k or top-p truncation. '
            f'The current interaction is {current_step + 1}; the episode allows {max_steps} interactions. '
            'Predict the SUM of all official rewards from the current interaction through the end '
            'of this episode, with discount 1. Exclude earlier rewards. '
            + reward_rule + 'Early stopping or exhaustion of the interaction/context budget adds no '
            'extra reward; interactions that never occur contribute zero. '
            f'Categories (label: cumulative return): {legend}. Return only the category label.'
            '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        )
        return tokenizer.encode(query, add_special_tokens=False)

    def readout_token_budget(self, tokenizer: Any, max_steps: int) -> int:
        """Exact worst-case query plus target length, without a model call."""
        return 1 + max(len(self.query_ids(tokenizer, current_step=i,
                                          max_steps=max_steps))
                       for i in range(max_steps))


class EventRatioReadout:
    """One complete-return contrast per response through the official runner.

    DT's signed vector estimates the individual deletion log-prob effects in
    PLAN.md. It is not a measured collection of leave-one-out forward passes.
    All tokens keep their own signed entries; no span broadcast or O routing.
    """

    def __init__(self, runner: Any, tokenizer: Any, *, task: str, max_steps: int,
                 packed_answer_targets: Any, max_length: int = 32768, minibatch_size: int = 4):
        if max_steps < 1:
            raise ValueError('Horizon must be positive')
        if tokenizer.eos_token_id is None:
            raise ValueError('EOS attribution requires the checkpoint EOS token')
        if minibatch_size < 1:
            raise ValueError('DT minibatch_size must be positive')
        self.minibatch_size = minibatch_size
        self.runner, self.tokenizer = runner, tokenizer
        self.alphabet = RewardAlphabet.for_task(task, max_steps)
        self.max_steps, self.max_length = max_steps, max_length
        self.packed_answer_targets = packed_answer_targets
        self.last_report: dict[str, Any] = {}

    def _prepare_episode(self, rows, report):
        events = [row for row in rows if bool(row['active_masks'])]
        labels = self.alphabet.label_ids(self.tokenizer)
        returns = episode_returns(rows)
        log_ratios = []
        requests = []
        report['nonzero_reward_events'] += sum(float(r['rewards']) != 0 for r in events)

        for row_index, (row, value) in enumerate(zip(rows, returns)):
            response = row['responses']
            width = response.numel()
            vector = torch.zeros(width, dtype=torch.float32)
            log_ratios.append(vector)
            if not bool(row['active_masks']):
                continue
            attention = row['attention_mask'].bool()
            positions = attention[-width:].nonzero().flatten().tolist()
            if positions != list(range(len(positions))):
                raise ValueError('Original generated response must have right padding only')
            prompt = row['input_ids'][:-width][attention[:-width]].cpu()
            actions = response[:len(positions)].cpu()
            if not torch.equal(row['input_ids'][-width:].cpu(), response.cpu()):
                raise ValueError('Original rollout input and response token identities differ')
            report['policy_tokens'] += len(positions)
            report['actual_row_lengths'].append(prompt.numel() + actions.numel())
            observed = self.alphabet.observed_index(value)
            if not positions or value == 0:
                continue
            query = torch.tensor(self.alphabet.query_ids(
                self.tokenizer, current_step=int(row['env_step']), max_steps=self.max_steps,
            ), device='cpu', dtype=torch.long)
            target = torch.tensor([labels[observed]], device='cpu', dtype=torch.long)
            length = prompt.numel() + actions.numel() + query.numel() + target.numel()
            if length > self.max_length:
                raise ValueError(f'Reward readout context {length} exceeds cap {self.max_length}; no silent truncation')
            start, end = prompt.numel(), prompt.numel() + actions.numel()
            # Only original current actions become EOS. Future observations,
            # actions and actual stopping time never enter either prefix.
            requests.append(dict(prompt=prompt, actions=actions, query=query, target=target,
                case={'target_ids': target, 'prompt_length': length - 1},
                start=start, end=end, vector=vector,
                source_step=int(row['env_step']),
                traj_uid=str(row['traj_uid']), observed_return=value,
                context_tokens=length, query_tokens=query.numel(), row_index=row_index))
        return log_ratios, requests

    @torch.no_grad()
    def episodes(self, episodes: list[list[dict[str, Any]]]) -> list[list[dict[str, torch.Tensor]]]:
        started = time.perf_counter()
        device = getattr(self.runner.model, 'execution_device', self.runner.model.lm_head.weight.device)
        report = dict(task=self.alphabet.task, policy_tokens=0, nonzero_reward_events=0,
                      finite_trace_calls=0, event_contrasts=0, minibatch_size=self.minibatch_size,
                      max_readout_length=0, actual_row_lengths=[], traces=[], max_length=self.max_length,
                      ratio_source='official_eos_dt_complete_return_signed_attribution_estimate',
                      target_semantics='complete_future_return',
                      reference_token_samples=0, per_token_probability_queries=0)
        vectors, requests = [], []
        for episode_index, rows in enumerate(episodes):
            values, pending = self._prepare_episode(rows, report)
            vectors.append(values)
            for request in pending:
                request['episode_index'] = episode_index
            requests.extend(pending)
        # Right extension comes strictly after the scored target. Causality
        # leaves every real prefix and predictor unchanged; it is compute
        # padding, not task history or a masked model/finite implementation.
        # The official runner receives its existing dense interleaved ABI.
        requests.sort(key=lambda request: request['context_tokens'])
        labels = self.alphabet.label_ids(self.tokenizer)
        planned_batches = (len(requests) + self.minibatch_size - 1) // self.minibatch_size
        print(f"[DT EOS plan] events={report['nonzero_reward_events']} "
              f"contrasts={len(requests)} batches={planned_batches}", flush=True)
        minimum_log_ratio = float('inf')
        minimum_batch = None
        for offset in range(0, len(requests), self.minibatch_size):
            batch = requests[offset:offset + self.minibatch_size]
            length = max(request['context_tokens'] for request in batch)
            selected = torch.full((len(batch), length), self.tokenizer.eos_token_id,
                                  device=device, dtype=torch.long)
            reference = selected.clone()
            for index, request in enumerate(batch):
                end = request['context_tokens']
                selected[index, :end] = torch.cat(tuple(request[name] for name in
                    ('prompt', 'actions', 'query', 'target'))).to(device)
                reference[index, :end] = selected[index, :end]
                reference[index, request['start']:request['end']] = self.tokenizer.eos_token_id
            signed, _, detail = trace_token_attribution(
                self.runner, reference, selected, [request['case'] for request in batch],
                [[0] for _ in batch], packed_answer_targets=self.packed_answer_targets,
                outcome_token_ids=labels,
            )
            report['finite_trace_calls'] += 1
            report['event_contrasts'] += len(batch)
            report['max_readout_length'] = max(report['max_readout_length'], length)
            batch_traces, batch_values = [], []
            for index, request in enumerate(batch):
                values = signed[index, request['start']:request['end']].cpu()
                request['vector'][:len(values)] = values
                item = detail['per_sample'][index] if len(batch) > 1 else detail
                trace = dict(
                    episode_index=request['episode_index'], source_step=request['source_step'],
                    observed_return=request['observed_return'], context_tokens=request['context_tokens'],
                    compute_tokens=length, query_tokens=request['query_tokens'],
                    root_effect=item['root_effect'], signed_sum=item['policy_credit_signed_sum'],
                    conservation_residual=item['conservation_residual'],
                    conservation_tolerance=item['conservation_tolerance'],
                    conservation_verified=item['conservation_verified'],
                    owner_batch_index=report['finite_trace_calls'] - 1,
                    source_log_ratio_min=float(values.min()), source_log_ratio_max=float(values.max()),
                    source_log_ratio_min_index=int(values.argmin()),
                    source_log_ratio_max_index=int(values.argmax()),
                    factual_target_logp=item.get('factual_target_logp'),
                    reference_target_logp=item.get('reference_target_logp'),
                )
                report['traces'].append(trace)
                batch_traces.append(trace)
                batch_values.append(values)
            batch_minimum = min(trace['source_log_ratio_min'] for trace in batch_traces)
            if batch_minimum < minimum_log_ratio:
                minimum_log_ratio = batch_minimum
                # Persist a new minimum through the existing owner stdout now:
                # a later failure must not lose the only exact replay input.
                # The final report still retains just the most-negative batch.
                # No thresholds, extra model calls, clipping or credit changes.
                minimum_batch = dict(
                    eos_token_id=self.tokenizer.eos_token_id,
                    outcome_token_ids=labels,
                    samples=[dict(
                        trace=trace, traj_uid=request['traj_uid'], observed_return=request['observed_return'],
                        source_start=request['start'], source_end=request['end'],
                        selected_input_ids=torch.cat(tuple(request[name] for name in
                            ('prompt', 'actions', 'query', 'target'))).tolist(),
                        source_signed=value.tolist(),
                    ) for request, trace, value in zip(batch, batch_traces, batch_values)],
                )
                print('[DT EOS minimum] ' + json.dumps(minimum_batch), flush=True)
            print(f"[DT EOS minibatch] contrasts={len(batch)} length={length} "
                  f"seconds={detail.get('complete_attribution_seconds_with_diagnostics')} "
                  f"batch={report['finite_trace_calls']}/{planned_batches} "
                  f"d_min={batch_minimum} "
                  f"d_max={max(trace['source_log_ratio_max'] for trace in batch_traces)}", flush=True)
        result = [return_credit_for_episode(rows, values)
                  for rows, values in zip(episodes, vectors)]
        report['seconds'] = time.perf_counter() - started
        report['nonzero_advantages'] = sum(int(row['dt_token_advantages'].count_nonzero())
                                          for episode in result for row in episode)
        report['conservation_failures'] = sum(not trace['conservation_verified'] for trace in report['traces'])
        if minimum_batch is not None:
            report['minimum_log_ratio_batch'] = minimum_batch
        self.last_report = report
        return result

    def episode(self, rows: list[dict[str, Any]]) -> list[dict[str, torch.Tensor]]:
        return self.episodes([rows])[0]
