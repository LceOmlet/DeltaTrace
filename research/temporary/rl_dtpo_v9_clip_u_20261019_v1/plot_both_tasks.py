"""Reuse the accepted six-panel collector/renderer with the six v9 identities."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
owner=HERE.parent/'rl_dtpo_symmetric_probability_ratio_postgroup_variance_20261018_v8/plot_both_tasks.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='066fba6a3166dc2d07c69ecabc4461959ebfba559e67865f986a0c02b511441f'
code=owner.read_text(encoding='utf-8').replace(
    '-eight-gpu-symmetric-probability-ratio-postgroup-variance-20261018-v8',
    '-eight-gpu-v9-clip-u-20261019-v1')
code=code.replace('DTPO v8 对称比·std(g·u)（8卡）','DTPO v9 clip(u)·std(g·u)（8卡）')
exec(compile(code,str(owner),'exec'),globals())
