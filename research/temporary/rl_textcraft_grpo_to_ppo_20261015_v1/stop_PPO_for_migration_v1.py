"""Reuse the protected PPO stop after its native checkpoint and migration are prepared."""
import json
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
planned = json.loads((OUT/'PPO-native-pause-for-GPU45-migration-v117.json').read_bytes())
prepared = json.loads((OUT/'native-PPO-resume-GPU45-v117-prepared.json').read_bytes())
current = json.loads((OUT/'native-PPO-control-v9-job.json').read_bytes())
assert planned['status']=='planned_only_PPO_still_running'
assert prepared['checkpoint']==planned['checkpoint']
assert set(prepared['configuration_changes'])=={'trainer.resume_mode','trainer.resume_from_path'}
assert prepared['physical_gpus']==[4,5]
assert prepared['effective_environment']['CUDA_VISIBLE_DEVICES']=='4,5'
assert (current['pid'],current['birth'])==(planned['job']['pid'],planned['job']['birth'])
assert Path(planned['marker']).read_text().strip()==str(planned['completed_iteration'])
source = OUT/'stop_PPO_for_authorized_diagnostic_v54.py'
code = source.read_text()
changes = {
    'PPO-temporary-stop-for-v47-v54.json': 'PPO-stop-for-GPU45-migration-v117.json',
    '[(1419849,1791977138.31),(3174448,1791793629.73),(2614045,1791949034.31)]':
        '[(3140624,1791993570.92),(2614045,1791949034.31)]',
    'User explicitly chose temporary PPO stop and full native restore after the bounded diagnostic.':
        'User requested native full PPO continuation on GPU4/5 released by stopping GRPO; no new training method or diagnostic.',
}
for old, new in changes.items():
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
