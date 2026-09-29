"""Report saved owner comparisons without inventing a numerical tolerance."""
import argparse
import json
from pathlib import Path
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('artifact', type=Path)
args = parser.parse_args()
data = torch.load(args.artifact, map_location='cpu', weights_only=True)
mask = data['response_mask'].bool()

def flatten(value):
    if isinstance(value, torch.Tensor):
        return value.detach().double().reshape(-1)
    items = (value[k] for k in sorted(value)) if isinstance(value, dict) else value
    return torch.cat([flatten(x) for x in items])

def compare(left, right):
    result = {}
    for key in ['old_log_probs', 'policy_forward_log_probs', 'raw_gradients', 'after_0', 'after_1']:
        a, b = left[key], right[key]
        if key == 'old_log_probs':
            a, b = a[mask], b[mask]
        if key == 'policy_forward_log_probs':
            a, b = torch.cat(a)[mask.repeat(2, 1)], torch.cat(b)[mask.repeat(2, 1)]
        a, b = flatten(a), flatten(b)
        delta = a - b
        result[key] = dict(exact=torch.equal(a, b), max_abs=float(delta.abs().max()),
                           relative_l2=float(delta.norm()/b.norm().clamp_min(1e-30)),
                           mean_difference=float(delta.mean()), elements=a.numel())
    return result

result = {'scope': __doc__, 'production_vs_owner': compare(data, data['paired_owner'])}
if 'repeat_installed' in data:
    result['production_repeat'] = compare(data['repeat_installed'], data)
if 'paired_untrimmed_head' in data:
    result['head_only_vs_owner'] = compare(data['paired_untrimmed_head'], data['paired_owner'])
args.artifact.with_suffix('.summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
