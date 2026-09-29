"""Read only the saved original-trainer output; no model or GPU allocation."""
import json
from pathlib import Path
import torch

audit = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/upstream-alignment-20260929')
path = audit/'probability-native-trainer/before_old_logprob-rankNone-step2-1790677654687055833.pt'
data = torch.load(path, map_location='cpu', weights_only=False)
print('BATCH_KEYS', list(data.batch.keys()))
print('META_KEYS', list(data.non_tensor_batch))
for i in range(len(data)):
    row = data.batch[i]
    probs = row['rollout_log_probs']
    mask = row['attention_mask'][-probs.numel():].bool()
    print(json.dumps(dict(row=i, nan=int(probs[mask].isnan().sum()), tokens=int(mask.sum()),
        unique_ids=int(row['responses'].unique().numel()), ids=row['responses'][:16].tolist(),
        probs=probs[:16].tolist(),
        meta={k: str(v[i])[:180] for k,v in data.non_tensor_batch.items() if k not in ('raw_prompt','multi_modal_inputs')})))
