"""Extend the existing original-parser/renderer adapter with one independent v10 run."""
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
owner=HERE.parent/'rl_textcraft_v5_two_gpu_reproduction_20261021_v1/plot_hourly.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='f738c84901b81faba4babe218f8f4436a62f01ef7b0c8061feba1743008cdcf1'
code=owner.read_text(encoding='utf-8')
anchor='print(json.dumps(snapshot))'
assert code.count(anchor)==1
code=code.replace(anchor,'''v10a=r/'receipts/textcraft-v10-target-logprob-20261021-v1'
v10run=r/'runs/textcraft-dtpo-v10-target-logprob-2gpu-formal-20261021-v2'
if (v10run/'source.json').is_file():
 extra=owner['collect'](str(v10a),[str(v10run/'textcraft-dt')])
 snapshot['jobs'].extend(extra['jobs'])
 snapshot['v10_collected_at']=extra['collected_at']
else:
 snapshot['jobs'].append(dict(task='TextCraft',method='dt',run_dir=str(v10run/'textcraft-dt'),
  budget=330,devices=[2,3],metrics=[],records=[],errors=[],pid=None,pid_alive=False,
  phase_label='prepared_waiting_native_verification_or_formal_launch'))
snapshot['v10_configuration_differences']=json.loads((v10a/'configuration-differences.json').read_bytes())
print(json.dumps(snapshot))''')
anchor="exec(compile(source,str(template),'exec'),globals())"
assert code.count(anchor)==1
extension=r'''
before="jobs=[old,new,reproduction]"
assert source.count(before)==1
source=source.replace(before,"v10=copy.deepcopy(next(j for j in current['jobs'] if Path(j['run_dir']).parent.name=='textcraft-dtpo-v10-target-logprob-2gpu-formal-20261021-v2'))\nv10.update(method='dt10',norm_scope='first_returned_rank_local_preclip')\njobs=[old,new,reproduction,v10]")
source=source.replace("output=PLOTS/'textcraft-v5-original-two-gpu-reproduction-vs-original-and-eight-20261021-v1'",
 "output=PLOTS/'textcraft-v10-target-logprob-two-gpu-vs-v5-original-reproduction-and-eight-20261021-v1'")
source=source.replace("DT_display_label='DTPO v5：原两卡 / 两卡正式复现 / 八卡'",
 "DT_display_label='DTPO v10双卡 / v5原两卡 / v5两卡复现 / v5八卡'")
source=source.replace("audit=dict(reproduction_original_configuration_changes=",
 "audit=dict(v10_configuration_differences=current['v10_configuration_differences'],v10_method_change='Target token native endpoint log-prob difference through original exponential Q/V/A; source remains v5',reproduction_original_configuration_changes=")
source=source.replace("reproduction_two_gpu=means[2],",
 "reproduction_two_gpu=means[2],v10_two_gpu=means[3],")
source=source.replace("('dt2','dt2new') if method=='dt2'", "('dt2','dt2new','dt10') if method=='dt2'")
before="'dt2new':dict(label='v5（本次两卡正式复现）',color='#009E73',marker='^',linestyle=':')}"
after="'dt2new':dict(label='v5（本次两卡正式复现）',color='#009E73',marker='^',linestyle=':'),\n    'dt10':dict(label='v10（target逐token log-prob差，两卡）',color='#8B5CF6',marker='D',linestyle='-.')}"
assert source.count(before)==1
source=source.replace(before,after)
source=source.replace('TextCraft · 原两卡 / 两卡正式复现 / 八卡：共同前',
 'TextCraft · v10双卡 / v5原两卡 / v5双卡复现 / v5八卡：共同前')
source=source.replace('TextCraft · 原两卡 / 两卡正式复现 / 八卡：各自完整历史',
 'TextCraft · v10双卡 / v5原两卡 / v5双卡复现 / v5八卡：各自完整历史')
'''
code=code.replace(anchor,extension+'\n'+anchor)
exec(compile(code,str(owner),'exec'),globals())
