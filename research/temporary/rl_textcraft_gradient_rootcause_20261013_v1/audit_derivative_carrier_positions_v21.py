"""Compare diagnostic carrier positions with the original VERL position owner."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import torch
from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

torch.set_num_threads(4)
OUT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-gradient-rootcause-20261013-v1')
cases=[]
for name in ('matched-task-target-derivatives-v11','matched-action-observation-target-derivatives-v20'):
    manifest=json.loads((OUT/name/'case-manifest.json').read_bytes())
    for entry in manifest['cases']:
        data=DataProto.load_from_disk(entry['path'])
        expected=compute_position_id_with_mask(data.batch['attention_mask'])
        actual=data.batch['position_ids']
        valid=data.batch['attention_mask'].bool()
        difference=(actual!=expected)&valid
        width=data.batch['responses'].shape[-1]
        targets=data.batch['diagnostic_target_mask'].bool()
        cases.append(dict(name=name,case=entry['name'],path=entry['path'],
            SHA256=hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest(),
            active_tokens=int(valid.sum()),mismatched_active_tokens=int(difference.sum()),
            mismatched_rows=int(difference.any(-1).sum()),
            mismatched_target_positions=int((difference[:,-width:]&targets).sum()),
            per_row_mismatched_active_tokens=difference.sum(-1).tolist(),
            actual_position_shape=list(actual.shape),owner_position_shape=list(expected.shape)))
owner=Path(inspect.getsourcefile(compute_position_id_with_mask))
actor=Path(os.environ['VERL_ROOT'])/'verl/workers/actor/dp_actor.py'
report=dict(unix=time.time(),scope=__doc__,cases=cases,
    position_owner=dict(path=str(owner),SHA256=hashlib.sha256(owner.read_bytes()).hexdigest(),source=inspect.getsource(compute_position_id_with_mask)),
    actor_owner=dict(path=str(actor),SHA256=hashlib.sha256(actor.read_bytes()).hexdigest(),
        position_use_lines=[line.strip() for line in actor.read_text().splitlines() if 'position_ids' in line]),
    source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),GPU_model_calls=0,optimizer_updates=0)
assert not torch.cuda.is_initialized()
dest=OUT/'derivative-carrier-position-audit-v21.json'
assert not dest.exists()
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
