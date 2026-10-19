"""Use the original bounded nonblocking observer on the actual negative2 job."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
owner=HERE.parent/'rl_dtpo_v9_clip_u_20261019_v1/profile_generation_phase.py'
code=owner.read_text(encoding='utf-8')
old='dtpo-v9-clip-u-postgroup-variance-20261019-v2'
assert code.count(old)==2
code=code.replace(old,'dtpo-v9-negative-clip2-postgroup-variance-20261019-v4')
old='runtime-overrides/actual-worker-imports-after-first-update-v1.json'
assert code.count(old)==1
code=code.replace(old,'runtime-overrides/actual-worker-imports-v9-negative2-eight-GPU-initial-v1.json')
exec(compile(code,str(owner),'exec'),globals())
