"""Map owner-reported payload character spans to unchanged owner token IDs.

Parsing and execution belong to AgentGym/LOOP. This module only transports
their character spans. The decoded-prefix path uses the tokenizer's decoder,
including for noncanonical sampled tokenizations; it never re-encodes IDs.
"""
from __future__ import annotations


def encoded_payload_mask(tokenizer, content, native_content_ids, spans):
    """Offsets of the same encoding that AgentGym uses for training content."""
    encoded = tokenizer(content, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded['input_ids']) != list(native_content_ids):
        raise ValueError('Payload offsets differ from original AgentGym content IDs')
    mask = [False] * len(native_content_ids)
    ranges = []
    for start, end in spans:
        if not 0 <= start <= end <= len(content):
            raise ValueError('Owner payload span is outside content')
        selected = [i for i, (left, right) in enumerate(encoded['offset_mapping'])
                    if start < right and left < end and left < right]
        for i in selected:
            mask[i] = True
        ranges.append(dict(char_span=[start, end], token_span=(
            [selected[0], selected[-1] + 1] if selected else [0, 0])))
    return mask, dict(mapping='original_native_encoding_offsets', ranges=ranges)


def decoded_payload_mask(tokenizer, content, generated_ids, spans):
    """Cover owner character spans with original sampled tokens.

    Qwen's owner obtains content by decoding these IDs. A partial UTF-8 token
    prefix may end in a replacement character; only its prefix agreeing with
    the complete owner content counts as completed characters. Whole tokens
    overlapping a character boundary are included, without altering IDs.
    """
    ids = list(generated_ids)
    decoded = tokenizer.decode(ids)
    if decoded != content:
        raise ValueError('Original LOOP content differs from its original decoded IDs')
    cache = {0: 0, len(ids): len(content)}

    def stable_chars(count):
        if count not in cache:
            prefix = tokenizer.decode(ids[:count])
            length = 0
            for left, right in zip(prefix, content):
                if left != right:
                    break
                length += 1
            cache[count] = length
        return cache[count]

    def first_at_least(position):
        low, high = 0, len(ids)
        while low < high:
            middle = (low + high) // 2
            if stable_chars(middle) < position:
                low = middle + 1
            else:
                high = middle
        return low

    mask = [False] * len(ids)
    ranges = []
    for start, end in spans:
        if not 0 <= start <= end <= len(content):
            raise ValueError('Owner payload span is outside content')
        if start == end:
            ranges.append(dict(char_span=[start, end], token_span=[0, 0]))
            continue
        first = first_at_least(start)
        if stable_chars(first) > start:
            first -= 1
        last = first_at_least(end)
        for i in range(first, last):
            mask[i] = True
        ranges.append(dict(char_span=[start, end], token_span=[first, last],
            covered_char_span=[stable_chars(first), stable_chars(last)]))
    return mask, dict(mapping='original_decoder_prefix_character_cover',
                     boundary_tokens='whole_overlapping_tokens', ranges=ranges,
                     decoded_prefix_calls=len(cache) - 2)


def full_trajectory_artifact(prompt_ids, full_ids, policy_mask, target_mask,
                             retained_response_ids, *, retained_start=None):
    """Keep full executed history separate from the original retained row."""
    prompt_ids, full_ids = list(prompt_ids), list(full_ids)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError('Full execution history does not start with original prompt IDs')
    suffix = full_ids[len(prompt_ids):]
    policy = list(policy_mask[len(prompt_ids):])
    target = list(target_mask[len(prompt_ids):])
    if len(policy) != len(suffix) or len(target) != len(suffix):
        raise ValueError('Full trajectory masks have a different length')
    if any(t and not p for t, p in zip(target, policy)):
        raise ValueError('Executed payload target includes a non-policy token')
    retained = list(retained_response_ids)
    start = len(prompt_ids) if retained_start is None else retained_start
    positions = []
    for index, value in enumerate(retained):
        full_index = start + index
        if len(prompt_ids) <= full_index < len(full_ids) and full_ids[full_index] == value:
            positions.append(full_index - len(prompt_ids))
        else:
            # Native synthetic termination/padding is not a sampled token.
            positions.append(-1)
    return dict(prompt_ids=prompt_ids, response_ids=suffix,
                policy_mask=policy, target_mask=target,
                retained_response_positions=positions)
