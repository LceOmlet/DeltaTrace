"""Arithmetic on the existing iteration81 snapshot; no model or new replay."""
import hashlib
import json
from pathlib import Path
import time

import torch
from verl import DataProto

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
sources = [json.loads((OUT/f'dt/rank{rank}-capture.json').read_bytes()) for rank in (0, 1)]
data = DataProto.concat([DataProto.load_from_disk(r['input']) for r in sources])
assert len(data) == 256
width = data.batch['responses'].shape[-1]
mask = data.batch['loss_mask'][:, -width:].bool()
adv = data.batch['advantages'].double()
assert not torch.cuda.is_initialized()
valid = adv[mask]
report = {
    'unix': time.time(), 'scope': __doc__,
    'GPU_model_backward_optimizer_scheduler_calls': 0,
    'input_sources': sources,
    'script_SHA256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'coefficient_scope': 'd(PG loss)/d(logprob) evaluated at the saved old-policy point; this is not a parameter-gradient norm.',
    'whole256_valid_tokens': int(mask.sum()),
    'whole256_whitened_mean': float(valid.mean()),
    'whole256_whitened_second_moment': float(valid.square().mean()),
    'global64_minibatches': [],
    'historical_full_batches_available': [81],
    'missing': 'Historical microbatch gradient Gram matrices, or equivalent paired parameter-gradient vectors at actual update boundaries. Native scalar grad_norm logs and nonzero-only raw-credit cohorts cannot determine these.',
}
for mini in range(4):
    idx = torch.cat((torch.arange(mini*32, (mini+1)*32), torch.arange(128+mini*32, 128+(mini+1)*32)))
    a, m = adv[idx], mask[idx]
    total = int(m.sum())
    old_coeff_sq, correct_coeff_sq = 0., float(a[m].square().sum())/(total*total)
    micro_rows = []
    for start in range(0, 64, 4):
        part = a[start:start+4][m[start:start+4]]
        n = part.numel()
        sq = float(part.square().sum())
        old_coeff_sq += sq/(16*16*n*n)
        micro_rows.append({
            'micro_index': start//4, 'valid_policy_tokens': n,
            'whitened_advantage_sumsq': sq,
            'legacy_micro_weight': 1/16,
            'correct_micro_weight': n/total,
            'legacy_to_correct_token_coefficient_ratio': total/(16*n),
        })
    report['global64_minibatches'].append({
        'index': mini, 'valid_policy_tokens': total, 'microbatches': micro_rows,
        'legacy_logprob_coefficient_L2': old_coeff_sq**.5,
        'correct_logprob_coefficient_L2': correct_coeff_sq**.5,
        'coefficient_L2_ratio': (old_coeff_sq/correct_coeff_sq)**.5,
        'micro_weight_total_variation': .5*sum(abs(1/16-r['correct_micro_weight']) for r in micro_rows),
        'norm_of_micro_gradients_or_cross_inner_products_available': False,
    })
report['decision'] = 'The denominator defect is real, but the shortening/target-density explanation does not yet quantitatively explain temporal parameter-gradient acceleration. Do not treat this arithmetic as a root-cause fix or authorize another training candidate.'
path = OUT/'saved-denominator-norm-accounting-v1.json'
path.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'receipt': str(path), 'report': report}))
