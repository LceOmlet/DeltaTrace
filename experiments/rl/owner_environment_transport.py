"""Exact artifacts between environment owners and the existing VERL collector.

No environment parser, reward, history policy, sampler, or optimizer lives here.
Padding/positions are the installed VERL functions; request sampling is vLLM's
SamplingParams. These seams are used only by the explicitly selected env factory.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyReply:
    token_ids: list[int]
    logprobs: list[float]
    text: str
    finish_reason: str


def preprocess_owner_tokens(collector, item, gen_batch, obs):
    import torch
    from verl.utils.model import compute_position_id_with_mask
    from verl.utils.torch_functional import postprocess_data

    raw_ids = list(obs['raw_prompt_ids'][item])
    ids = torch.tensor([raw_ids], dtype=torch.long)
    ids, mask = postprocess_data(
        ids, torch.ones_like(ids), collector.config.data.max_prompt_length,
        collector.tokenizer.pad_token_id, left_pad=True, truncation='error')
    row = dict(input_ids=ids[0], attention_mask=mask[0],
               position_ids=compute_position_id_with_mask(mask)[0],
               raw_prompt_ids=raw_ids, anchor_obs=None, index=item,
               data_source=gen_batch.non_tensor_batch['data_source'][item])
    row['owner_sampling_kwargs'] = obs['sampling_kwargs'][item]
    return row


def owner_sampling_params(base, overrides, active_rows):
    """Construct the vLLM per-request API, without implementing sampling."""
    from inspect import signature
    from vllm import SamplingParams

    rows = range(len(overrides)) if active_rows is None else active_rows
    fields = signature(SamplingParams.from_optional).parameters
    base_values = {name: getattr(base, name) for name in fields}
    result = []
    for row in rows:
        for key in overrides[row]:
            if key not in fields:
                raise ValueError(f'Unknown vLLM sampling parameter: {key}')
        # Let the owner normalize None/defaults, validate, and compute stop caches.
        params = SamplingParams.from_optional(**(base_values | dict(overrides[row])))
        # VERL owns trajectory grouping; each row is one actual env action.
        if params.n != 1:
            raise ValueError('Environment callback requires one response per row')
        result.append(params)
    return result


def policy_reply(batch, row):
    """Use exact engine lengths/text/reason, not EOS or text-marker inference."""
    n = int(batch.non_tensor_batch['owner_response_length'][row])
    return PolicyReply(
        batch.batch['responses'][row, :n].tolist(),
        batch.batch['rollout_log_probs'][row, :n].tolist(),
        batch.non_tensor_batch['owner_response_text'][row],
        batch.non_tensor_batch['owner_finish_reason'][row])
