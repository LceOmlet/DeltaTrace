"""Original plot/collect/parse owners with stopped-GRPO and approved-v4 labels."""
from pathlib import Path
import hashlib

source=Path(__file__).resolve().parent.parent/'rl_textcraft_grpo_to_ppo_20261015_v1/plot_current_algorithms_v1.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='6d45d531542f7da4b0b30b2699a3ffd1140b50874b38c9575205325690c7ce52'
wrapper=source.read_text()
anchor="exec(compile(code, str(source), 'exec'), globals())"
extra='''old="snapshot['DT_display_label']=('DTPO v3（组乘子，仅方差）' if current_DT.get('variance_standardization')\\n                               else '新 DTPO（组乘子，无白化）')"
assert old in code
new="snapshot['DT_display_label']=('DTPO v4（结算target，线性信用）' if current_DT['version']=='textcraft-dt-weighted-reward-targets-20261015-v4' else ('DTPO v3（组乘子，仅方差）' if current_DT.get('variance_standardization') else '新 DTPO（组乘子，无白化）'))"
code=code.replace(old,new)
'''
assert anchor in wrapper
wrapper=wrapper.replace(anchor,extra+'\n'+anchor)
exec(compile(wrapper,str(source),'exec'),globals())
