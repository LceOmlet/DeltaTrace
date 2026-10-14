"""Plot current approved DTPO and existing controls from original owner logs.

Composes the existing collect/read_job_logs/parse_logs entry points. No new
training logger, model call, interpolation or missing-value substitution.
"""
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('strict_plot_transport',
    HERE.parent/'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)

remote = r'''
import hashlib,json,psutil,runpy,time
from pathlib import Path
ROOT=Path(ROOT_VALUE)
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
parser_path=OUT/'original_plot_training_progress.py'
owner=runpy.run_path(str(parser_path),run_name='native_algorithm_comparison_parser')
manifest=json.loads((ROOT/'formal-training.json').read_bytes())
chosen=[j for j in manifest['jobs'] if j.get('task')=='TextCraft' and j.get('method') in ('dt','grpo')]
assert len(chosen)==2 and {j['method'] for j in chosen}=={'dt','grpo'}
outputs=[j['output'] for j in chosen]
snapshot=owner['collect'](str(ROOT),outputs)
assert {j['pid']:j['observed_process_created_unix'] for j in snapshot['jobs']}=={
    j['pid']:j['observed_process_created_unix'] for j in chosen}
assert all(j['pid_alive'] for j in snapshot['jobs'])
current_DT=next(j for j in chosen if j['method']=='dt')
snapshot['DT_version']=current_DT['version']
snapshot['DT_display_label']=('DTPO v3（组乘子，仅方差）' if current_DT.get('variance_standardization')
                               else '新 DTPO（组乘子，无白化）')
job_path=OUT/'native-PPO-control-v9-job.json'
job=json.loads(job_path.read_bytes());p=psutil.Process(job['pid'])
assert p.create_time()==job['birth'] and p.status()!=psutil.STATUS_ZOMBIE
run=Path(job['run'])
chain=[];identity=job;maximum=None;seen=set()
while True:
    assert identity['run']==job['run'] and identity['pid'] not in seen
    seen.add(identity['pid']);chain.append((identity,maximum))
    previous=identity.get('previous_job_identity')
    if previous is None:break
    maximum=identity['resumed_completed_iteration'];identity=previous
metrics_by_step={};records=[];errors=[];sessions=[]
for identity,maximum in reversed(chain):
    found=list(Path('/tmp/ray').glob('session_*_'+str(identity['pid'])))
    assert len(found)==1
    session=found[0];sessions.append(str(session))
    lines,errs=owner['read_job_logs']({'ray_session':str(session)})
    values,_=owner['parse_logs'](lines)
    minimum=identity.get('resumed_completed_iteration',0)
    if maximum is not None:values=[m for m in values if minimum<m['step']<=maximum]
    else:assert all(m['step']>minimum for m in values)
    metrics_by_step.update({m['step']:m for m in values})
    records.extend(lines);errors.extend(errs)
checkpoint=run/'textcraft-ppo/checkpoints/latest_checkpointed_iteration.txt'
snapshot['jobs'].append(dict(task='TextCraft PPO',environment_task='TextCraft',method='ppo',
    pid=job['pid'],observed_process_created_unix=job['birth'],pid_alive=True,
    run_dir=str(run/'textcraft-ppo'),budget=330,devices=job['physical_gpus'],gpu=job['physical_gpus'][0],
    checkpoint_step=int(checkpoint.read_text()) if checkpoint.exists() else None,
    metrics=[metrics_by_step[k] for k in sorted(metrics_by_step)],records=records,errors=errors,
    phases=[],ray_sessions=sessions,source_job=dict(path=str(job_path),SHA256=hashlib.sha256(job_path.read_bytes()).hexdigest()),
    resume_scope='Same original run and explicit native full-checkpoint resume chain; only completed retained intervals are combined.'))
snapshot['source_label']=snapshot['DT_display_label']+'; original GRPO/PPO controls'
snapshot['original_parser_source']=dict(path=str(parser_path),SHA256=hashlib.sha256(parser_path.read_bytes()).hexdigest())
snapshot['collection_model_calls']=0
print(json.dumps(snapshot))
'''.replace('ROOT_VALUE', repr(entry.ROOT))
script = ('source '+shlex.quote(entry.ENTRY+'/metax-entry.env.sh')+
          '\nCUDA_VISIBLE_DEVICES= "'+'$VENV_PYTHON'+'" - <<\'PY\'\n'+remote+'\nPY\n')
result = subprocess.run(entry.SSH+['bash', '-s'], input=script.encode(),
                        capture_output=True, check=True, timeout=120)
snapshot = json.loads(result.stdout)
stamp = dt.datetime.fromtimestamp(snapshot['collected_at'],dt.timezone(dt.timedelta(hours=8)))
output = Path('F:/DeltaTraceBackups/plots')/snapshot['DT_version']/stamp.strftime('%Y%m%d-%H%M%S')
output.mkdir(parents=True,exist_ok=True)
snapshot_path=output/'source-snapshot.json'
snapshot_path.write_text(json.dumps(snapshot,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,
    'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold',
    'figure.facecolor':'#f8fafc','axes.facecolor':'white','text.color':'#0f172a'})
styles={
    'dt':dict(label=snapshot['DT_display_label'],color='#0072B2',marker='o',linestyle='-'),
    'grpo':dict(label='原生 GRPO',color='#D55E00',marker='s',linestyle='--'),
    'ppo':dict(label='原生 PPO',color='#009E73',marker='^',linestyle=':')}
panels=[('episode/reward/mean','平均训练回报','原环境奖励单位'),
        ('actor/entropy_loss','策略熵','原框架日志单位'),
        ('actor/grad_norm','梯度范数（原生更新、裁剪前）','每迭代日志值'),
        ('actor/kl_loss','相对参考策略的 KL','原框架日志单位'),
        ('response_length/mean','平均响应段长度（含工具返回）','token / 轨迹'),
        ('episode/tool_call_count/mean','平均工具调用数（原接口交互计数）','次 / 轨迹')]
jobs=snapshot['jobs']
last={j['method']:max((m['step'] for m in j['metrics']),default=0) for j in jobs}
assert last['dt']>0,'Current DTPO has no completed native iteration yet; source snapshot retained.'
common=min(last.values())
files=[snapshot_path]

def draw(selected,limit,title,name):
    fig,axes=plt.subplots(3,2,figsize=(13.5,10.2))
    fig.suptitle(title,x=.075,y=.98,ha='left',fontsize=18,weight='bold')
    subtitle=stamp.strftime('%Y-%m-%d %H:%M:%S UTC+8')+'  ·  '+', '.join(
        styles[j['method']]['label']+f" {last[j['method']]}/330" for j in selected)
    fig.text(.075,.942,subtitle,color='#475569',fontsize=9.5)
    for ax,(key,heading,unit) in zip(axes.flat,panels):
        for job in selected:
            values=[m for m in job['metrics'] if limit is None or m['step']<=limit]
            xs=[m['step'] for m in values]
            ys=[m['values'].get(key,float('nan')) for m in values]
            ys=[v if math.isfinite(v) else float('nan') for v in ys]
            if any(math.isfinite(v) for v in ys):
                ax.plot(xs,ys,linewidth=1.7,markersize=4,**styles[job['method']])
        ax.set_title(heading,loc='left',fontsize=10.5,pad=9)
        ax.set_xlabel('已完成的完整训练迭代');ax.set_ylabel(unit)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True));ax.grid(alpha=.18)
        ax.margins(x=.04,y=.12)
        if ax.lines:ax.legend(frameon=False,fontsize=8)
        else:ax.text(.5,.5,'原日志无此指标',ha='center',va='center',transform=ax.transAxes)
    fig.text(.075,.038,'仅原生日志中的实际已完成迭代；缺失不填零、不平滑。梯度为原 console 三位小数日志。',fontsize=8.5,color='#475569')
    fig.text(.075,.018,'KL 的 0.000 不表示精确零；响应长度包含工具观察，工具调用数沿用原 TextCraft 交互计数。',fontsize=8.5,color='#475569')
    fig.subplots_adjust(left=.075,right=.98,top=.89,bottom=.10,hspace=.54,wspace=.27)
    for extension in ('png','svg'):
        path=output/(name+'.'+extension);fig.savefig(path,dpi=165);files.append(path)
    plt.close(fig)

draw(jobs,common,f'TextCraft · 三算法前 {common} 次完整训练迭代对比','algorithm-comparison-common-window')
draw(jobs,None,'TextCraft · 当前各算法全部已完成迭代','algorithm-comparison-full')
csv_path=output/'metrics-with-sources.csv'
with csv_path.open('w',newline='',encoding='utf-8-sig') as stream:
    writer=csv.writer(stream);writer.writerow(['algorithm','iteration','metric','value','original_sources'])
    for job in jobs:
        for m in job['metrics']:
            for key,value in m['values'].items():
                writer.writerow([styles[job['method']]['label'],m['step'],key,value,json.dumps(m['sources'],ensure_ascii=False)])
files.append(csv_path)
report=dict(output=str(output),files=[str(p) for p in files],completed_iterations=last,
    common_window=common,original_snapshot=str(snapshot_path),new_training_changes=0,model_calls=0,
    primary_figure=str(output/'algorithm-comparison-common-window.png'),
    full_history_figure=str(output/'algorithm-comparison-full.png'))
(output/'plot-receipt.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
