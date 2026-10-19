"""Compose the existing regression and native-transport CPU behavior suites."""
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner=ROOT/'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v1/source/run_cpu_checks.py'
code=owner.read_text().replace('dtpo-v9-clip-u-postgroup-variance-20261019-v1',
    'dtpo-v9-negative-clip2-postgroup-variance-20261019-v3')
old="('v9-clip',MODE,[AUDIT/'source/test_credit_v9.py'],[])]:"
assert code.count(old)==1
code=code.replace(old,"('v9-clip',MODE,[AUDIT/'source/test_credit_v9.py'],[]),\n        ('v9-negative2',MODE,[AUDIT/'source/test_credit_negative2.py'],[])]:")
anchor="        env['DT_CREDIT_MODE']=mode"
assert code.count(anchor)==1
code=code.replace(anchor,anchor+"\n        env['DT_EXPONENTIAL_CREDIT_NEGATIVE_CLIP']='2' if label=='v9-negative2' else '1'")
code=code.replace('clip_range=[-1,1]','clip_range=[-2,1]')
exec(compile(code,str(owner),'exec'),globals())
