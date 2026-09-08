"""Retokenize author-fixed answers and gold with unchanged author helpers.

No model, generation, evaluator or scoring method is implemented here. Existing
cached token spans are checked against the original tokenizer before replacing
them for a new tokenizer; sentence indices and source texts remain unchanged.
"""
import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def load_author_span_helpers(author_root, expected_hashes):
    wanted = {
        'exp/exp2/dataset_utils.py': {
            'CachedExample', '_BOX_PATTERN', '_find_box_span',
            'extract_boxed_answer', '_find_answer_span',
            'attach_spans_from_answer', 'ruler_gold_prompt_token_indices'},
        'ft_ifr_improve.py': {
            'STOP_TOKENS', 'SKIP_WHITESPACE', 'STRIP_BEFORE_MATCH',
            'is_stop_token', 'keep_token_indices'},
    }
    namespace = dict(dataclass=dataclass, re=re, Any=Any, Dict=Dict, List=List,
                     Optional=Optional, Sequence=Sequence)
    for name, symbols in wanted.items():
        path = Path(author_root) / name
        raw = path.read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(raw).hexdigest() != expected_hashes[name]:
            raise ValueError('Author span/filter source mismatch: ' + name)
        nodes = []
        found = set()
        for node in ast.parse(raw, filename=str(path)).body:
            names = {node.name} if isinstance(node, (ast.ClassDef, ast.FunctionDef)) else set()
            if isinstance(node, ast.Assign):
                names = {x.id for x in node.targets if isinstance(x, ast.Name)}
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = {node.target.id}
            if names & symbols:
                nodes.append(node)
                found.update(names & symbols)
        if found != symbols:
            raise ValueError('Missing author helper symbols: ' + name)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def remap_cached_example(record, old_tokenizer, new_tokenizer, engine, helpers):
    """Return author spans and audited absolute positions after fixed-text preparation."""
    from official_fixed_text_inputs import prepare_fixed_text

    cls = helpers['CachedExample']
    fields = ['prompt', 'target', 'indices_to_explain', 'attr_mask_indices',
              'sink_span', 'thinking_span', 'metadata']
    original = cls(**{k: record.get(k) for k in fields})
    if not original.target or not original.metadata.get('boxed_answer'):
        raise ValueError('An author-cached target and boxed-answer metadata are required.')

    def fresh(tokenizer):
        # The author's function preserves old nonempty spans. Clear them first
        # so this call actually retokenizes rather than inheriting stale indices.
        clear = cls(prompt=original.prompt, target=original.target,
                    indices_to_explain=None, attr_mask_indices=original.attr_mask_indices,
                    sink_span=None, thinking_span=None, metadata=dict(original.metadata))
        out = helpers['attach_spans_from_answer'](clear, tokenizer)
        if not out.sink_span or len(out.sink_span) != 2:
            raise ValueError('Author answer-span mapping failed.')
        # Exact sample_and_filter.py policy for the author processed caches.
        out.indices_to_explain = list(out.sink_span)
        return out

    old, new = fresh(old_tokenizer), fresh(new_tokenizer)
    for name in ['sink_span', 'thinking_span', 'indices_to_explain']:
        if getattr(old, name) != getattr(original, name):
            raise ValueError('Original tokenizer does not reproduce cached ' + name)
    assert new.attr_mask_indices == original.attr_mask_indices
    assert new.prompt == original.prompt and new.target == original.target
    case = prepare_fixed_text(engine, original.prompt, original.target)
    meta = case['metadata']
    if not (meta['user_token_sequence_exact'] and meta['author_positions_equal_offsets']):
        raise ValueError('Author user positions do not match the formatted-input offsets.')
    if meta['boundary_crossing_positions']:
        raise ValueError('User text crosses a chat-template token boundary.')
    positions = meta['user_positions_from_offsets']
    keep = helpers['keep_token_indices'](engine.user_prompt_tokens)
    eligible = [positions[i] for i in keep]
    gold = helpers['ruler_gold_prompt_token_indices'](new, new_tokenizer)
    old_gold = helpers['ruler_gold_prompt_token_indices'](old, old_tokenizer)
    target = new_tokenizer(original.target, add_special_tokens=False)['input_ids']
    assert case['target_ids'][:-1].tolist() == target
    assert 0 <= new.sink_span[0] <= new.sink_span[1] < len(target)
    answer = original.metadata['boxed_answer'].strip()
    start = original.target.rfind(answer)
    assert start >= 0
    per_needle = []
    for needle in original.metadata.get('needle_spans', []):
        if not isinstance(needle, dict) or 'span' not in needle:
            raise ValueError('Malformed official needle metadata.')
        a, b = needle['span']
        assert 0 <= a < b <= len(original.prompt)
        one = cls(**{k: getattr(new, k) for k in fields})
        one.metadata = dict(new.metadata, needle_spans=[needle])
        indices = helpers['ruler_gold_prompt_token_indices'](one, new_tokenizer)
        assert indices
        per_needle.append({'character_span': [a, b], 'user_token_indices': indices,
                           'formatted_input_indices': [positions[i] for i in indices]})
    if per_needle:
        assert sorted({i for x in per_needle for i in x['user_token_indices']}) == gold
    meta.update(gold_and_cached_generation_indices_remapped=True,
                input_ids_sha256=hashlib.sha256(case['input_ids'].numpy().tobytes()).hexdigest())
    mapping = {
        'original_cached_spans_reproduced': True,
        'old_sink_span': old.sink_span, 'new_sink_span': new.sink_span,
        'old_thinking_span': old.thinking_span, 'new_thinking_span': new.thinking_span,
        'indices_to_explain': new.indices_to_explain,
        'answer_character_span': [start, start + len(answer)],
        'attr_mask_indices': new.attr_mask_indices, 'attr_mask_unit': 'sentence',
        'old_gold_user_token_indices': old_gold, 'gold_user_token_indices': gold,
        'gold_formatted_input_indices': [positions[i] for i in gold],
        'per_needle': per_needle, 'keep_local_indices': keep,
        'eligible_input_positions': eligible,
        'baseline': 'Replace only eligible user-prompt tokens by tokenizer EOS; preserve chat template and fixed target.',
    }
    return new, case, mapping
