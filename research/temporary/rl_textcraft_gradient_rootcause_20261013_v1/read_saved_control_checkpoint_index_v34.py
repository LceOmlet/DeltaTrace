"""Read completed saved DTPO/GRPO checkpoint availability, without model calls."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
SHARED=Path('/mnt/si0021787ci2/default/lzq/deltatrace_backups/textcraft-20261011/runtime')/ROOT.relative_to('/')
report=dict(unix=time.time(),scope=__doc__,algorithms={},GPU_model_calls=0,training_calls=0,GRPO_reruns=0)
for method in ('dt','grpo'):
    rel=Path(f'runs/textcraft-{method}-formal-native-graph-fresh-20261012-v1')
    launch=ROOT/rel/'launch.json'
    options=json.loads(launch.read_bytes())['options']
    bases=[]
    for base in (ROOT,SHARED):
        directory=base/rel/f'textcraft-{method}/checkpoints'
        marker=directory/'latest_checkpointed_iteration.txt'
        rows=[]
        for cp in sorted(directory.glob('global_step_*')):
            try:step=int(cp.name.removeprefix('global_step_'))
            except ValueError:continue
            files=[cp/'actor'/f'{prefix}_world_size_2_rank_{rank}.pt' for rank in (0,1) for prefix in ('model','optim','extra_state')]
            rows.append(dict(step=step,path=str(cp/'actor'),actor_state_files_present=all(p.is_file() for p in files)))
        bases.append(dict(path=str(directory),completed_marker=marker.read_text().strip() if marker.exists() else None,checkpoints=rows))
    report['algorithms'][method]=dict(launch=str(launch),launch_SHA256=hashlib.sha256(launch.read_bytes()).hexdigest(),
                                    relevant_options={k:v for k,v in options.items() if any(s in k for s in ('lora','actor.optim','ppo_mini','ppo_micro','grad_clip'))},bases=bases)
path=OUT/'saved-control-checkpoint-index-v34.json'
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(path),report=report)))
