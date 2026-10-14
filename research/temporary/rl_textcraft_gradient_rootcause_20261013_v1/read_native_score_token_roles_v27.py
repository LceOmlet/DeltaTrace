"""Read exact token identities behind existing score-force measurements only."""
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer
from verl import DataProto

torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
DEST=OUT/'matched-target-score-readout-v27'
account=json.loads((DEST/'native-target-score-accounting.json').read_bytes())
launch=json.loads((ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/launch.json').read_bytes())
tok=AutoTokenizer.from_pretrained(launch['options']['actor_rollout_ref.model.path'],local_files_only=True,trust_remote_code=True)
report=dict(unix=time.time(),scope=__doc__,cases=[],model_calls=0,parameter_backward=0,DT_calls=0,training_changed=False)
for case in account['cases']:
    data=DataProto.load_from_disk(case['native_probability_carrier'])
    mask=data.batch['diagnostic_target_mask'].bool()
    entries=[]
    for token,values in sorted(case['score_by_token_ID'].items(),key=lambda x:x[1]['score_lower_squared'],reverse=True):
        locations=(mask[:15]&(data.batch['responses'][:15]==int(token))).nonzero()
        samples=[]
        for row,pos in locations[:4].tolist():
            before=data.batch['responses'][row,max(0,pos-30):pos].tolist()
            after=data.batch['responses'][row,pos:pos+16].tolist()
            samples.append(dict(row=row,position=pos,native_log_probability=float(data.batch['old_log_probs'][row,pos]),
                preceding_native_text=tok.decode(before,skip_special_tokens=False),
                native_text_starting_at_token=tok.decode(after,skip_special_tokens=False)))
        entries.append(dict(token_id=int(token),decoded_token=tok.decode([int(token)],skip_special_tokens=False),**values,samples=samples))
    report['cases'].append(dict(case=case['case'],token_roles=entries))
assert not torch.cuda.is_initialized()
path=DEST/'native-score-token-identities.json';assert not path.exists()
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(path),report=report)))
