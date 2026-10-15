"""Original plot owners with receipt-verified stopped controls and v4 status."""
from pathlib import Path
import hashlib

source=Path(__file__).resolve().parent.parent/'rl_textcraft_grpo_to_ppo_20261015_v1/plot_current_algorithms_v1.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='6d45d531542f7da4b0b30b2699a3ffd1140b50874b38c9575205325690c7ce52'
wrapper=source.read_text()
anchor="exec(compile(code, str(source), 'exec'), globals())"
extra='''old="snapshot['DT_display_label']=('DTPO v3（组乘子，仅方差）' if current_DT.get('variance_standardization')\\n                               else '新 DTPO（组乘子，无白化）')"
assert old in code
new="snapshot['DT_display_label']=(('DTPO v4（已撤回）' if current_DT.get('status')=='stopped_method_equivalence_not_established' else 'DTPO v4（结算target，线性信用）') if current_DT['version']=='textcraft-dt-weighted-reward-targets-20261015-v4' else ('DTPO v3（组乘子，仅方差）' if current_DT.get('variance_standardization') else '新 DTPO（组乘子，无白化）'))"
code=code.replace(old,new)
old="    else:\\n        assert j['pid_alive']"
assert old in code
new="    elif j['method']=='dt' and not j['pid_alive']:\\n        withdrawn_path=ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4/DTPO-v4-method-withdrawn.json'\\n        withdrawn=json.loads(withdrawn_path.read_bytes())\\n        assert withdrawn['status']=='stopped_method_equivalence_not_established' and not withdrawn['survivors']\\n        assert (j['pid'],j['observed_process_created_unix'])==(withdrawn['job']['pid'],withdrawn['job']['birth'])\\n        j['training_status']='withdrawn_no_training'\\n    else:\\n        assert j['pid_alive']"
code=code.replace(old,new)
old="assert last['dt']>0,'Current DTPO has no completed native iteration yet; source snapshot retained.'"
assert old in code
code=code.replace(old,"# A fresh approved run has no completed metrics until its first native iteration.")
old="        if ax.lines:ax.legend(frameon=False,fontsize=8)"
assert old in code
code=code.replace(old,"        if last['dt']==0 and limit is None:\\n            ax.plot([],[],color=styles['dt']['color'],linestyle=styles['dt']['linestyle'],label=styles['dt']['label']+'（无完成迭代）')\\n"+old)
old="else:ax.text(.5,.5,'原日志无此指标',ha='center',va='center',transform=ax.transAxes)"
assert old in code
code=code.replace(old,"else:ax.text(.5,.5,('DTPO v4尚无完成迭代，共同窗口为空' if common==0 else '原日志无此指标'),ha='center',va='center',transform=ax.transAxes)")
'''
assert anchor in wrapper
wrapper=wrapper.replace(anchor,extra+'\n'+anchor)
exec(compile(wrapper,str(source),'exec'),globals())
