"""Derive a seed-only FT control while preserving all original reasoning hops."""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE / 'ft_both_method_source.py').read_text(encoding='utf-8')
old = '        n_hops: int = 1,'
assert source.count(old) == 1
source = source.replace(old, old + '\n        initial_target_mask=None,')
old = '        base_ifr_raw = compute_ifr_sentence_aggregate('
assert source.count(old) == 1
source = source.replace(old, '''        if initial_target_mask is not None:
            seed_mask = torch.as_tensor(initial_target_mask, dtype=params.model_dtype)
            if seed_mask.ndim != 1 or seed_mask.numel() != int(gen_len):
                raise ValueError('Initial target mask must match generation length')
            if not bool(torch.isfinite(seed_mask).all()) or bool((seed_mask < 0).any()):
                raise ValueError('Initial target mask must be finite and nonnegative')
            seed_mask = seed_mask[int(all_gen_start):int(all_gen_end)+1]
            base_weights = seed_mask if base_weights is None else base_weights * seed_mask

''' + old)
code = '''"""Experimental FT target selection: modify only the first hop's seed mask.

The method body is copied from the verified original source; every subsequent
hop and observation rule is unchanged. The original module is never modified.
"""
import ft_ifr_improve as original
for _key, _value in vars(original).items():
    if not _key.startswith('__'):
        globals()[_key] = _value


class InitialTargetFT(original.LLMIFRAttributionBoth):
''' + source
compile(code, 'ft_target_control.py', 'exec')
(HERE / 'ft_target_control.py').write_text(code, encoding='utf-8')
print(hashlib.sha256(code.encode()).hexdigest())
