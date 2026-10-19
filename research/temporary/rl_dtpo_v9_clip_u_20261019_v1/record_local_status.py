"""Record native progress/import receipts without changing training or its parser."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess

from transport import run_python, HERE

data = json.loads(run_python(r'''
import hashlib,json,psutil,runpy,time
from pathlib import Path
R=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
A=R/'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v1'
parser=R/'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'
assert hashlib.sha256(parser.read_bytes()).hexdigest()=='c20559e38e651284859837735b8e7d15fe7ca59b090c62b7bc651228f6ba3607'
owner=runpy.run_path(str(parser),run_name='v9_native_deployment_record')
plan=json.loads((A/'six-combination-manifest.json').read_bytes())
segments=[]
for segment in plan['segments']:
 prepared=json.loads(Path(segment['prepared']).read_bytes())
 state_path=Path(segment['prepared']).with_name('campaign-state.json')
 state=json.loads(state_path.read_bytes()) if state_path.exists() else None
 rows=[]
 for original in prepared['jobs']:
  matches=[j for j in (state or {}).get('jobs',[]) if j['method']==original['method']]
  j=matches[0] if matches else original
  row=dict(task=j['task'],method=j['method'],run=j['run'],version=j['version'],
   pid=j.get('pid'),birth=j.get('birth'),source_commit=j['local_patch_commit'],
   upstream_commit=j['upstream_commit'],source_receipt=j.get('source_receipt'),
   checkpoints=j['checkpoints'],mode=j['effective_environment'].get('DT_CREDIT_MODE'),
   source_bindings=j['source_bindings'],effective_configuration=j['options'],
   effective_config_SHA256=j['effective_config_SHA256'],status='queued',completed=0,
   actual_worker_bindings=[])
  if row['pid']:
   alive=psutil.pid_exists(row['pid']) and psutil.Process(row['pid']).create_time()==row['birth']
   snapshot=owner['collect'](str(R),[j['output']]);current,=snapshot['jobs']
   path=state_path.with_name(j['method']+'-current-source-snapshot.json')
   path.write_text(json.dumps(snapshot,indent=2)+'\n')
   row.update(alive=alive,status='running' if alive else 'exited',phase=current.get('phase_label'),
    completed=owner['completed'](current),checkpoint_step=current.get('checkpoint_step'),
    latest_metrics=current['metrics'][-1] if current['metrics'] else None,
    source_snapshot=str(path))
   override=Path(j['run'])/'runtime-overrides'
   for p in sorted(override.glob('actual-worker-imports-*.json')):
    row['actual_worker_bindings'].append(dict(path=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),record=json.loads(p.read_bytes())))
  rows.append(row)
 segments.append(dict(prepared=str(segment['prepared']),state=str(state_path) if state else None,jobs=rows))
report=dict(collected=time.time(),controller=json.loads((A/'controller-job.json').read_bytes()),
 manifest=str(A/'six-combination-manifest.json'),segments=segments,
 additional_model_or_optimizer_test_calls=0)
(A/'latest-deployment-observation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
'''))

result = Path('experiments/rl/results_dtpo_v9_clip_u_20261019_v1.json')
record = json.loads(result.read_text(encoding='utf-8'))
now = dt.datetime.fromtimestamp(data['collected'], dt.timezone(dt.timedelta(hours=8))).isoformat()
rows = [j for segment in data['segments'] for j in segment['jobs']]
assert len(rows) == 6
assert all(j['mode'] == 'reward_target_exponential_clipped_v9' for j in rows if j['method'] == 'dt')
assert {j['source_commit'] for j in rows} == {'a88d479c199f9b3171a7d29e8a5840dc4759a6ea'}
actual = [x for j in rows for x in j['actual_worker_bindings']]
for item in actual:
    results = item['record']['results']
    assert len(results) == 8
    assert all(r['DT_CREDIT_MODE'] == 'reward_target_exponential_clipped_v9' and r['DP_world_size'] == 8 for r in results)
    assert not any(m.get('matches_prepared_source') is False for r in results for m in r['modules'].values())
previous=record.get('deployment')
if previous and previous['controller']['pid']!=data['controller']['pid']:
    record.setdefault('deployment_history',[]).append(previous)
failed=[j for j in rows if j['status']=='exited' and j['completed']<200]
record.update(status=('native_v9_queue_needs_attention_no_failed_combination_skipped' if failed
        else 'native_v9_six_run_queue_in_progress'), recorded_at=now, deployment=data,
    actual_binding_scope='Only naturally imported modules; absent lazy modules remain pending.',
    hourly=dict(id='textcraft',status='ACTIVE',entry=str(HERE/'plot_hourly.py'),
        entry_SHA256=hashlib.sha256((HERE/'plot_hourly.py').read_bytes()).hexdigest()),
    goal_status='active',old_histories_spliced=False)
result.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(HERE/'latest-deployment-observation.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

summary=dict(recorded_at=now,status=record['status'],method='v5+clip(u,[-1,1]) v9; post-group variance, no centering',
    source_commit=rows[0]['source_commit'],upstream_commit=rows[0]['upstream_commit'],controller=data['controller'],
    manifest=data['manifest'],jobs=rows,result_file=str(result),verification=dict(CPU_tests=41,
        native='two ranks, B4/rank, actual DT multi-target and exact32768; zero optimizer updates'),
    hourly=record['hourly'],goal_status='active')
key='20261019_DTPO_v9_clip_u_six_eight_GPU_current'
def update_json(text):
    rendered=json.dumps(summary,indent=2,ensure_ascii=False).replace('\n','\n  ')
    match=re.search(r'^  '+re.escape(json.dumps(key))+r': ',text,re.M)
    if match:
        start=match.end();_,end=json.JSONDecoder().raw_decode(text[start:])
        return text[:start]+rendered+text[start+end:]
    start=text.index('{')+1
    return text[:start]+'\n  '+json.dumps(key)+': '+rendered+','+text[start:]

first=rows[0]
title='DTPO v9：v5指数信用逐事件裁剪，六组八卡正式队列'
bindings=('八rank原只读元数据RPC已核对v9模式/DP8及实际已导入源码；未自然导入的lazy DT模块不冒称已验证。' if actual
    else '实际八rank导入RPC已提交，待初始化自然完成；不以初始化或占卡称健康训练。')
failure_note=('当前首组首次DT原生exit1、0轮、未进入optimizer更新；MetaX明确报mcHostRegister页面锁定内存过多。控制器停在失败组合，其他五组未启动。该平台分配错误与v9信用裁剪分开记录，不能称已完成健康训练。' if failed else '')
header=f'''## {now} {title}

{failure_note}

人类明确两个任务DTPO均为v5+clip v9，另外四组保持原PPO/GRPO，替代goal里过时的symmetric标签。u=clip(-expm1(-d),-1,1)，先逐事件裁剪，再乘既定带符号信用b_k并汇总；单事件为原官方已归一化GRPO g。Q0=sum b_k、V0=sum b_k*min(exp(-d),2)、A0=sum b_k*u，原self/观察/padding/future mask保持。使用固定VERL原masked_var对最终保留policy位置汇总后的A0计算一个共同尺度，Q/V/A仅方差缩放一次、不减均值，不在乘g前缩放u、不裁剪g或最终gu。原DT、actor/core/optimizer、官方任务负载和效率路径保持。

方法与执行源{first['source_commit']}已提交推送，上游VERL {first['upstream_commit']}。21项原v5 CPU回归和20项v9行为通过，实际双rank每rank B4、多target及精确32768 DT通过，0 optimizer更新；六配置由原生Hydra逐section核对。此双rank验收不冒称八rank正式容量或训练质量。{bindings} controller {data['controller']['pid']}/{data['controller']['birth']}，首组driver {first['pid']}/{first['birth']}；本次完成{first['completed']}/200，实际phase={first.get('phase')}。实际导入路径/SHA、配置、PID出生和原日志来源见results_dtpo_v9_clip_u_20261019_v1.json及远端对应receipt。

顺序TextCraft DTPO v9→原PPO→原GRPO→AppWorld DTPO v9→原PPO→原GRPO，每组基础权重/0轮独占物理0–7、200完整迭代。TextCraft32组×8/global64/epochs1/4更新/save5/keep2；AppWorld作者40组×6/global32/epochs2/原取消、残批、test5、loop_sampler/共享save2/keep2；32k/B4/LoRA8/16、seed/采样/LR保持。原process owner f71ff4d8仅原生exit0、日志1..200与八rank共享完整checkpoint200齐全后交接，失败不跳过。其余五组当前queued；以各自原生状态后续更新。共享检查点在/mnt/si0021787ci2/default/lzq/deltatrace_backups/{{textcraft,appworld}}-8gpu-20261017/<对应20261019-v1 run>/checkpoints，日志/rollout和receipt保持共享ROOT路径。

旧八卡v5自身exit1、0轮，首次DT原pinned capture报CUDA invalid argument，根因未确定，未向旧进程发停止信号；原日志及证据保留，不称clip修复了该环境错误。所有v1–v8及旧v5不拼入新v9。小时textcraft已ACTIVE切换新统一入口，分别保存两任务正式三算法图，另保留v9与历史PPO105/GRPO191对照；后者不当新队列进度。新全局范数实测与原local分别保存；历史显示仅按批准√2均衡分片假设估算并明确标注，原值与来源保留。goal持续至六组最终训练、共享完整checkpoint200和最终图全部完成，旧梯度/SQL等任务不恢复。

'''
env_title='v9复用现有运行环境与已验收效率路径'
env_header=f'''## {now} {env_title}

六组正式队列仅两DTPO隔离源接入已批准v9公式，PPO/GRPO沿原owner；不重建环境、不重装、不下载、不清缓存。严格SSH31146/HostKeyAlias32036，source原metax-entry.env.sh后使用原VENV_PYTHON并合并launch环境，MACA_PATH/LD_LIBRARY_PATH/PYTHONPATH保持。原host-register8分配配置、TP1/defaultgraph、MRoPE、DT runner6a112224、actor acd3c8a5、corefc2f992b/worker198f509c及native_host_memory4419c2e3保持。启动前八卡空闲，原TextCraft作者服务36005复用，无新服务替代。上次v5首次DT错误与本次v9双rank32k验收分别记录，不能将测试通过写成旧错误根因已修复。部署/实际导入/后续原生进度见RUNTIME_RECORD最新v9条及结果文件。

'''
def prepend(text,block,marker):
    if text.startswith('## ') and marker in text.splitlines()[0]:
        end=text.find('\n## ',1);text=text[end+1:] if end>=0 else ''
    return block+text
for path,transform in [(Path('experiments/rl/current_runtime.json'),update_json),
        (Path('experiments/rl/RUNTIME_RECORD.md'),lambda s:prepend(s,header,title)),
        (Path('experiments/rl/REMOTE_ENVIRONMENT.md'),lambda s:prepend(s,env_header,env_title))]:
    path.write_text(transform(path.read_text(encoding='utf-8')),encoding='utf-8')
    head=subprocess.check_output(['git','show','HEAD:'+path.as_posix()]).decode('utf-8')
    blob=subprocess.check_output(['git','hash-object','-w','--stdin'],input=transform(head).encode('utf-8')).decode().strip()
    subprocess.run(['git','update-index','--add','--cacheinfo','100644',blob,path.as_posix()],check=True)
print(json.dumps(dict(result=str(result),completed={j['task']+':'+j['method']:j['completed'] for j in rows},
    phase=first.get('phase'),actual_binding_receipts=len(actual),unrelated_worktree_changes_staged=False),ensure_ascii=False))
