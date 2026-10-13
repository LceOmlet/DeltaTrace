"""Read-only exact frozen backbone equality at native checkpoints5 and80."""
import json
from pathlib import Path
import time
import torch
torch.set_num_threads(4)
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
REL=Path('runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/textcraft-dt/checkpoints')
SHARED=Path('/mnt/si0021787ci2/default/lzq/deltatrace_backups/textcraft-20261011/runtime')/ROOT.relative_to('/')
report={'scope':__doc__,'unix':time.time(),'GPU_used':False,'ranks':[]}
for rank in (0,1):
 paths=[base/REL/f'global_step_{step}/actor/model_world_size_2_rank_{rank}.pt' for step,base in [(5,SHARED),(80,ROOT)]]
 models=[torch.load(p,map_location='cpu',weights_only=False,mmap=True) for p in paths]
 assert set(models[0])==set(models[1])
 names=[k for k in models[0] if 'lora_' not in k]
 different=[];elements=0
 for name in names:
  values=[m[name].to_local() if hasattr(m[name],'to_local') else m[name] for m in models]
  elements+=values[0].numel()
  if not torch.equal(*values):different.append({'name':name,'max_abs':(values[0].float()-values[1].float()).abs().max().item()})
 report['ranks'].append({'rank':rank,'source_files':[str(p) for p in paths],
                         'frozen_tensors':len(names),'local_elements':elements,'different':different,'all_exact_equal':not different})
 del models
path=OUT/'frozen-backbone-history-equality-v1.json';path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(path),'report':report}))
