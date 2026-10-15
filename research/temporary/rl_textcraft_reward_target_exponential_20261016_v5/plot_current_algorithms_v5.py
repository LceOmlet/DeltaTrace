"""Reuse the verified original collectors/parsers/plot with the v5 label."""
import hashlib
from pathlib import Path

source=Path(__file__).resolve().parent.parent/'rl_textcraft_weighted_reward_targets_20261015_v4/plot_current_algorithms_v4.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='78a85a43a5e0d938e6143e4c03b32b79c7f8d10e343df73650f6053e0dd4ba87'
code=source.read_text()
lines=code.splitlines()
matches=[i for i,line in enumerate(lines) if line.startswith('new="snapshot[\'DT_display_label\']')]
assert len(matches)==1
label="assert current_DT['version']=='textcraft-dt-reward-target-exponential-20261016-v5'; snapshot['DT_display_label']='DTPO v5（结算target，指数信用）'"
lines[matches[0]]='new='+repr(label)
code='\n'.join(lines)+'\n'
code=code.replace('DTPO v4尚无完成迭代，共同窗口为空','DTPO v5尚无完成迭代，共同窗口为空')
exec(compile(code,str(source),'exec'),globals())
