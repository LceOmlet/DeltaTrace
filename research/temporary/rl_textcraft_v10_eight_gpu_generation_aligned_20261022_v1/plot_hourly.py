"""Add the fresh DP8 v10 identity to the original-parser/renderer adapter.

Historical runs stay separate. Only this new run is current; local shard and
measured global gradient norms remain on their existing separate axes.
"""
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
owner = HERE.parent / 'rl_textcraft_v10_target_logprob_20261021_v1/plot_hourly.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '753316bb57e851c358e2fba467ebec03eb31a050ea29b686d3a0d24ff77f7643'
source = owner.read_text(encoding='utf-8')

remote_extension = r'''
v10eight_audit=r/'receipts/textcraft-v10-eight-gpu-generation-aligned-20261022-v1'
v10eight_run=r/'runs/textcraft-dtpo-v10-target-logprob-8gpu-formal-20261022-v1'
if (v10eight_run/'source.json').is_file():
 extra=owner['collect'](str(v10eight_audit),[str(v10eight_run/'textcraft-dt')])
 snapshot['jobs'].extend(extra['jobs'])
 snapshot['v10_eight_collected_at']=extra['collected_at']
else:
 snapshot['jobs'].append(dict(task='TextCraft',method='dt',run_dir=str(v10eight_run/'textcraft-dt'),
  budget=330,devices=list(range(8)),metrics=[],records=[],errors=[],pid=None,pid_alive=False,
  phase_label='prepared_only_waiting_full_generation_review'))
config_diff=v10eight_audit/'configuration-differences.json'
snapshot['v10_eight_configuration_differences']=json.loads(config_diff.read_bytes()) if config_diff.is_file() else None
stop=r/'receipts/textcraft-user-stop-generation-investigation-20261021-v1/stop-state.json'
if stop.is_file():
 snapshot['historical_user_stop_receipt']=dict(path=str(stop),SHA256=hashlib.sha256(stop.read_bytes()).hexdigest())
 for job in snapshot['jobs']:
  if Path(job['run_dir']).parent.name in ('textcraft-dtpo-v5-original-2gpu-formal-20261021-v1',
                                        'textcraft-dtpo-v10-target-logprob-2gpu-formal-20261021-v3'):
   job['original_reader_pid_alive']=job.get('pid_alive')
   job.update(pid_alive=False,historical_snapshot=True,phase_label='historical_user_stopped_20261021',
              stop_receipt=str(stop))
'''

render_extension = r'''
before='jobs=[old,new,reproduction,v10]'
assert source.count(before)==1
source=source.replace(before,"v10eight=copy.deepcopy(next(j for j in current['jobs'] if Path(j['run_dir']).parent.name=='textcraft-dtpo-v10-target-logprob-8gpu-formal-20261022-v1'))\nv10eight.update(method='dt10eight',norm_scope='measured_global_preclip_native_DTensor_full_tensor')\njobs=[old,new,reproduction,v10,v10eight]")
source=source.replace("output=PLOTS/'textcraft-v10-target-logprob-two-gpu-vs-v5-original-reproduction-and-eight-20261021-v1'",
 "output=PLOTS/'textcraft-v10-eight-gpu-generation-aligned-vs-v5-and-two-gpu-history-20261022-v1'")
source=source.replace("DT_display_label='DTPO v10双卡 / v5原两卡 / v5两卡复现 / v5八卡'",
 "DT_display_label='DTPO v10八卡 / 独立双卡与v5历史'")
source=source.replace("audit=dict(v10_configuration_differences=",
 "audit=dict(v10_eight_configuration_differences=current['v10_eight_configuration_differences'],historical_scope='Descriptive independent runs; historical sampling protocols differ',v10_configuration_differences=")
source=source.replace('v10_two_gpu=means[3],','v10_two_gpu=means[3],v10_eight_gpu=means[4],')
source=source.replace("'reproduction_two_gpu','eight_minus_two'",
 "'reproduction_two_gpu','v10_two_gpu','v10_eight_gpu','eight_minus_two'")
before="'dt10':dict(label='v10（target逐token log-prob差，两卡）',color='#8B5CF6',marker='D',linestyle='-.')}"
after="'dt10':dict(label='v10（两卡，已停止）',color='#8B5CF6',marker='D',linestyle='-.'),\n    'dt10eight':dict(label='v10（本次八卡）',color='#D81B60',marker='v',linestyle='-')}"
assert source.count(before)==1
source=source.replace(before,after)
source=source.replace("label='v5（本次两卡正式复现）'", "label='v5（双卡复现，已停止）'")
source=source.replace("label='v5（八卡）'", "label='v5（八卡，已停止）'")
source=source.replace("else ('dt8',))]", "else ('dt8','dt10eight'))]")
source=source.replace('TextCraft · v10双卡 / v5原两卡 / v5双卡复现 / v5八卡：共同前',
 'TextCraft · v10八卡与独立双卡、v5历史：共同前')
source=source.replace('TextCraft · v10双卡 / v5原两卡 / v5双卡复现 / v5八卡：各自完整历史',
 'TextCraft · v10八卡与独立双卡、v5历史：各自完整迭代')
source=source.replace('相同goal不等于相同题面；本图是描述性比较。训练回报不是独立评估；长度含工具返回，KL 0.000不表示精确零。',
 '历史生成协议不同，本图为描述性比较。训练回报非独立评估；长度含工具返回，KL 0.000非精确零。')
'''

anchor = r"code=code.replace(anchor,extension+'\n'+anchor)"
assert source.count(anchor) == 1
source = source.replace(anchor, 'extension += ' + repr(render_extension) + '\n' + anchor)
anchor = "exec(compile(code,str(owner),'exec'),globals())"
assert source.count(anchor) == 1
source = source.replace(anchor,
    "assert code.count('print(json.dumps(snapshot))')==1\n"
    "code=code.replace('print(json.dumps(snapshot))'," +
    repr(remote_extension + '\nprint(json.dumps(snapshot))') + ")\n" + anchor)
exec(compile(source, str(owner), 'exec'), globals())
