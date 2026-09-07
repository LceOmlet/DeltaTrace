"""Reuse the author's fixed-target preparation; audit its token/character mapping.

This helper never generates text or reuses cached token indices from another
tokenizer. The supplied engine is the real author's LLMAttribution instance.
"""


def load_author_preparer(author_root, source_hashes):
    """Compile the unchanged author text-preparation class, without plot imports.

    llm_attr's top-level imports include unrelated attribution/plot/word-frequency
    packages. This explicitly extracted class is not a model implementation;
    methods and constants are taken unchanged from hash-verified original ASTs.
    """
    import ast
    import hashlib
    from pathlib import Path
    from typing import Any, Dict, Optional
    import torch
    import torch.nn.functional as F

    root = Path(author_root)
    trees = {}
    for name in ['llm_attr.py', 'shared_utils.py']:
        raw = (root / name).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(raw).hexdigest() != source_hashes[name]:
            raise ValueError('Author source identity mismatch: ' + name)
        trees[name] = ast.parse(raw, filename=str(root / name))
    cls = next(n for n in trees['llm_attr.py'].body if isinstance(n, ast.ClassDef) and n.name == 'LLMAttribution')
    constants = [n for n in trees['shared_utils.py'].body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in ['DEFAULT_PROMPT_TEMPLATE', 'DEFAULT_GENERATE_KWARGS'] for t in n.targets)]
    assert len(constants) == 2
    namespace = {'__name__': 'verified_author_text_preparation', 'Any': Any, 'Dict': Dict,
                 'Optional': Optional, 'torch': torch, 'F': F}
    exec(compile(ast.Module(body=constants, type_ignores=[]), str(root / 'shared_utils.py'), 'exec'), namespace)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(root / 'llm_attr.py'), 'exec'), namespace)
    return namespace['LLMAttribution']


def prepare_fixed_text(engine, prompt, target):
    import hashlib
    import torch

    if not isinstance(prompt, str) or not isinstance(target, str) or not target:
        raise ValueError('A nonempty author-cached target is required.')
    engine.target_response(prompt, target)
    tokenizer = engine.tokenizer
    assert engine.generation == target
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    prompt_ids = engine.prompt_ids[0]
    target_ids = engine.generation_ids[0]
    assert decode(prompt_ids) == engine.prompt
    assert decode(target_ids) == target + tokenizer.eos_token
    assert int(target_ids[-1]) == tokenizer.eos_token_id
    # Tokenizer offsets, rather than a first-token match, locate the actual
    # user text in the unchanged author-formatted input.
    user_text = ' ' + prompt
    start = engine.prompt.find(user_text)
    assert start >= 0 and engine.prompt.find(user_text, start + 1) < 0
    end = start + len(user_text)
    encoded = tokenizer(engine.prompt, add_special_tokens=False, return_offsets_mapping=True)
    assert encoded['input_ids'] == prompt_ids.tolist()
    positions = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if a < end and b > start]
    crossings = [i for i in positions if encoded['offset_mapping'][i][0] < start or encoded['offset_mapping'][i][1] > end]
    user_ids = engine.user_prompt_ids[0].tolist()
    embedded_ids = [encoded['input_ids'][i] for i in positions]
    # Cross-boundary merges remain visible; this helper does not silently
    # certify cached gold or target spans for a new vocabulary.
    metadata = {'prompt_length': len(prompt_ids), 'target_length': len(target_ids),
        'input_length': len(prompt_ids) + len(target_ids), 'user_positions_from_offsets': positions,
        'author_user_positions': list(engine.user_prompt_indices), 'boundary_crossing_positions': crossings,
        'user_token_sequence_exact': embedded_ids == user_ids,
        'author_positions_equal_offsets': list(engine.user_prompt_indices) == positions,
        'target_text_sha256': hashlib.sha256(target.encode()).hexdigest(),
        'prompt_text_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'replacement_token_id': tokenizer.eos_token_id,
        'gold_and_cached_generation_indices_remapped': False}
    return {'input_ids': torch.cat([prompt_ids, target_ids]).cpu(), 'target_ids': target_ids.cpu(),
            'prompt_length': len(prompt_ids), 'metadata': metadata}


def right_pad_batch(cases, pad_token_id, device):
    import torch

    width = max(len(case['input_ids']) for case in cases)
    ids = torch.full((len(cases), width), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, case in enumerate(cases):
        n = len(case['input_ids'])
        ids[i, :n] = case['input_ids'].to(device)
        mask[i, :n] = 1
    return {'input_ids': ids, 'attention_mask': mask}
