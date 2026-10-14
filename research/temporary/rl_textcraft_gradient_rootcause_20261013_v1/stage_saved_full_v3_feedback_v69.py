"""Stage the finite full-v3 regression sources; no device selection or use."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('strict_entry', HERE.parent / 'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
dest = entry.ROOT + '/receipts/textcraft-gradient-rootcause-20261013-v1/native-full-current-v3-feedback-v69/source'
files = [HERE / name for name in ('observe_saved_full_v3_feedback_v69.py',
    'observe_common_update_feedback_v47.py', 'observe_native_gradient_chain_v4.py')]
files += [HERE.parent / 'rl_upstream_alignment_20260929' / name for name in (
    'observe_native_optimizer_minibatch.py', 'observe_native_actor_loss_gradients.py')]
subprocess.run([*entry.SSH, 'mkdir', '-p', dest], check=True)
for path in files:
    subprocess.run([*entry.SCP, str(path), entry.SSH[-1]+':'+dest+'/'+path.name], check=True)
record = dict(status='source_staged_only_no_GPU_job', destination=dest,
    source_SHA256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    native_update_calls_planned=2, native_optimizer_steps_planned=8,
    native_DT_request_rows_planned=128, generation_calls_planned=0,
    validation_score_calls_planned=0, GRPO_model_reruns_planned=0, formal_training_changed=False)
(HERE / 'full-current-v3-feedback-source-v69.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record))
