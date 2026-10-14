"""Stage the bounded native diagnostic source without choosing or using GPUs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
spec=importlib.util.spec_from_file_location('strict_entry',HERE.parent/'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
dest=entry.ROOT+'/receipts/textcraft-gradient-rootcause-20261013-v1/native-common-update-feedback-v47/source'
subprocess.run(entry.SSH+['mkdir','-p',dest],check=True)
files=[HERE/'observe_common_update_feedback_v47.py',HERE/'observe_native_gradient_chain_v4.py',
       HERE.parent/'rl_upstream_alignment_20260929/observe_native_optimizer_minibatch.py',
       HERE.parent/'rl_upstream_alignment_20260929/observe_native_actor_loss_gradients.py']
for path in files:subprocess.run(entry.SCP+[str(path),entry.SSH[-1]+':'+dest+'/'+path.name],check=True)
receipt=dict(status='source_staged_only_no_GPU_job',destination=dest,
             source_SHA256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
             native_update_calls_planned=2,native_optimizer_steps_planned=8,model_calls_executed=0,
             formal_training_changed=False)
(HERE/'common-update-feedback-source-v47.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
