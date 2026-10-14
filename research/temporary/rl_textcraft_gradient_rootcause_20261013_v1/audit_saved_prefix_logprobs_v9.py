"""Read saved identical-prefix inputs and log probabilities; no forward calls."""
import hashlib
import json
from pathlib import Path
import time
import torch
from verl import DataProto

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
forks = json.loads((OUT / 'same-prefix-DT-values-v5.json').read_bytes())['forks']
captures = [json.loads((OUT / f'dt/rank{r}-capture.json').read_bytes()) for r in [0, 1]]
data = DataProto.concat([DataProto.load_from_disk(r['input']) for r in captures])
width = data.batch['responses'].shape[1]
prompt = data.batch['input_ids'].shape[1] - width


def sha(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


out = []
for fork in forks:
    for branch in fork['branches']:
        if branch['n'] < 2:
            continue
        records = []
        for occurrence in branch['occurrences']:
            row, j = occurrence['row'], occurrence['position']
            end = prompt + j
            ids = data.batch['input_ids'][row, :end]
            mask = data.batch['attention_mask'][row, :end].bool()
            positions = data.batch['position_ids'][row, ..., :end]
            entry = {'row': row, 'rank': row // 128, 'position': j,
                     'native_B4_index_within_rank': (row % 128) // 4,
                     'active_prefix_ID_SHA256': sha(ids[mask]),
                     'active_prefix_position_ID_SHA256': sha(positions[..., mask]),
                     'active_prefix_positions_start': positions[..., mask][..., :5].tolist(),
                     'active_prefix_positions_end': positions[..., mask][..., -5:].tolist(),
                     'left_padding_columns': int(data.batch['attention_mask'][row].long().argmax()),
                     'total_active_columns': int(data.batch['attention_mask'][row].sum()),
                     'old_logp': float(data.batch['old_log_probs'][row, j]),
                     'token_id': int(data.batch['responses'][row, j])}
            for key in ['ref_log_prob', 'ref_log_probs', 'rollout_log_probs']:
                if key in data.batch:
                    entry[key] = float(data.batch[key][row, j])
            records.append(entry)
        out.append({'prefix_SHA256': fork['prefix_SHA256'], 'token': branch['token'],
                    'target': branch['target'], 'prefix_tail': fork['prefix_tail'],
                    'same_active_IDs': len({r['active_prefix_ID_SHA256'] for r in records}) == 1,
                    'same_active_position_IDs': len({r['active_prefix_position_ID_SHA256'] for r in records}) == 1,
                    'old_logp_range': max(r['old_logp'] for r in records) - min(r['old_logp'] for r in records),
                    'records': records})
report = {'unix': time.time(), 'scope': __doc__, 'GPU_model_calls': 0,
          'input_sources': captures, 'available_batch_keys': list(data.batch.keys()), 'groups': out,
          'limits': ['This audit compares saved probabilities and tensor inputs only.',
                     'Differences do not yet identify their origin or establish a gradient-growth cause.',
                     'No original training/forward/gradient implementation is copied or replaced.']}
assert not torch.cuda.is_initialized()
dest = OUT / 'saved-prefix-logprobs-v9.json'
dest.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'receipt': str(dest), 'report': report}))
