"""User-authorized GRPO stop; reuse the existing protected process-group stop."""
from pathlib import Path
source=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')/'receipts/textcraft-gradient-rootcause-20261013-v1/stop_PPO_for_authorized_diagnostic_v54.py'
code=source.read_text()
changes={
 "OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'": "OUT=ROOT/'receipts/textcraft-grpo-to-ppo-20261015-v1'\nOUT.mkdir(exist_ok=True)",
 "PPO-temporary-stop-for-v47-v54.json": "GRPO-user-stopped.json",
 "job=json.loads((OUT/'native-PPO-control-v9-job.json').read_bytes())": "job=next(j for j in json.loads((ROOT/'formal-training.json').read_bytes())['jobs'] if j.get('task')=='TextCraft' and j.get('method')=='grpo')\njob=dict(job,run=str(Path(job['output']).parent),birth=job['observed_process_created_unix'])\nassert (job['pid'],job['birth'])==(3174448,1791793629.73)",
 "for role in ('actor','critic'):": "for role in ('actor',):",
 "[(1419849,1791977138.31),(3174448,1791793629.73),(2614045,1791949034.31)]": "[(3140624,1791993570.92),(2666718,1792028884.24),(2614045,1791949034.31)]",
 "User explicitly chose temporary PPO stop and full native restore after the bounded diagnostic.": "User requested stopping GRPO and continuing the existing PPO through its native complete checkpoint.",
 "temporarily_stopped_native_full_restore_required": "stopped_GRPO_at_user_request_checkpoints_and_logs_preserved",
}
for old,new in changes.items():
    assert old in code,old
    code=code.replace(old,new)
exec(compile(code,str(source),'exec'),globals())
