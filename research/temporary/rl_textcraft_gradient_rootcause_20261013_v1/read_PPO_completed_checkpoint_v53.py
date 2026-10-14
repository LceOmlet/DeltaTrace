"""Read native PPO completed-checkpoint marker and files for resource planning."""
import json
from pathlib import Path
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN=ROOT/'runs/textcraft-ppo-control-fixed-verl-fresh-20261014-v2'
launch=json.loads((RUN/'launch.json').read_bytes())
directory=Path(launch['options']['trainer.default_local_dir'])
marker=directory/'latest_checkpointed_iteration.txt'
step=int(marker.read_text().strip()) if marker.exists() else None
checkpoint=directory/f'global_step_{step}' if step is not None else None
report=dict(unix=time.time(),run=str(RUN),native_checkpoint_directory=str(directory),
    marker=str(marker),completed_iteration=step,files=[dict(path=str(p),bytes=p.stat().st_size)
    for p in checkpoint.rglob('*') if p.is_file()] if checkpoint is not None else [],
    model_calls=0,optimizer_calls=0,stop_signals=0)
dest=OUT/'PPO-completed-checkpoint-v53.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
