"""Record actual deployment and add only this task's metadata to shared records."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
from transport import run_python,HERE

data=json.loads(run_python(r'''
import hashlib,json,psutil,runpy,time
from pathlib import Path
R=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');A=R/'receipts/textcraft-v5-eight-gpu-20261019-v1'
s=json.loads((A/'campaign-state.json').read_bytes());j=s['jobs'][0]
p=psutil.Process(j['pid']);assert p.create_time()==j['birth']
parser=R/'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'
assert hashlib.sha256(parser.read_bytes()).hexdigest()=='c20559e38e651284859837735b8e7d15fe7ca59b090c62b7bc651228f6ba3607'
owner=runpy.run_path(str(parser),run_name='v5_native_deployment_record')
snapshot=owner['collect'](str(R),[j['output']]);current,=snapshot['jobs']
(A/'dt-current-source-snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
bindings=[]
for q in sorted((Path(j['run'])/'runtime-overrides').glob('actual-worker-imports-*.json')):
 v=json.loads(q.read_bytes())
 bindings.append(dict(path=str(q),SHA256=hashlib.sha256(q.read_bytes()).hexdigest(),record=v))
report=dict(collected=time.time(),controller=dict(pid=s['controller_pid'],birth=s['controller_birth']),
 job={k:j.get(k) for k in ['task','method','version','pid','birth','run','output','checkpoints','source_receipt']},
 state=s['status'],phase=current.get('phase_label'),completed=owner['completed'](current),checkpoint_step=current.get('checkpoint_step'),
 mode=j['effective_environment']['DT_CREDIT_MODE'],upstream_commit=j['upstream_commit'],
 execution_source_commit=j['local_patch_commit'],method_source_commit=j['method_source_commit'],execution_parent_commit=j['execution_parent_commit'],
 expected_source_bindings=j['source_bindings'],effective_configuration=j['options'],effective_config_SHA256=j['effective_config_SHA256'],
 actual_worker_bindings=bindings,workers=[dict(pid=q.pid,birth=q.create_time(),phase=q.name()) for q in p.children(recursive=True) if 'WorkerDict' in q.name()],
 cpu_verification=json.loads((A/'CPU-behavior-verification.json').read_bytes()),
 service=json.loads((A/'official-service.json').read_bytes()),platform_restoration=json.loads((A/'restored-platform-files.json').read_bytes()),
 prior_runtime_observation=json.loads((A/'prior-runtime-observed.json').read_bytes()),
 latest_metrics=current['metrics'][-1] if current['metrics'] else None,additional_model_or_optimizer_test_calls=0,
 manifest=str(A/'six-combination-manifest.json'),native_source_snapshot=str(A/'dt-current-source-snapshot.json'))
(A/'latest-deployment-observation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
'''))
now=dt.datetime.fromtimestamp(data['collected'],dt.timezone(dt.timedelta(hours=8))).isoformat()
record=dict(recorded_at=now,status='native_v5_training_submitted',**data,
    method_specification='experiments/rl/PLAN.md',budget=200,
    official_method='v5 original per-event exponential EOS Q/V/A, official normalized GRPO, original post-group advantage variance scaling',
    algorithm_code_changes=0,prior_histories_spliced=False,old_v7_goal_status='paused',
    hourly_automation='textcraft',hourly_plot_entry=str(HERE/'plot_hourly.py'),
    hourly_plot_entry_SHA256=hashlib.sha256((HERE/'plot_hourly.py').read_bytes()).hexdigest())
if data['actual_worker_bindings']:
    latest=data['actual_worker_bindings'][-1]['record']
    assert len(latest['results'])==8
    assert all(r['DT_CREDIT_MODE']=='reward_target_exponential_v5' and r['DP_world_size']==8 for r in latest['results'])
    assert not any(m.get('matches_prepared_source') is False for r in latest['results'] for m in r['modules'].values())
    record['status']='native_v5_eight_rank_mode_and_loaded_imports_verified_training_in_progress'
    record['actual_worker_binding_scope']='Eight-rank mode/DP and already imported modules only; lazy DT modules remain pending their natural first use.'
    record['actual_DT_owner_imports_status']=('pending_natural_DT_lazy_imports' if any(
        r['modules'].get('reward_readout',{}).get('imported') is False for r in latest['results'])
        else 'observed_after_natural_import')
result=Path('experiments/rl/results_textcraft_v5_eight_gpu_20261019_v1.json')
result.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
(HERE/'latest-deployment-observation.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
key='20261019_TextCraft_v5_eight_GPU_current'
def update_json(text):
    rendered=json.dumps(record,indent=2,ensure_ascii=False).replace('\n','\n  ')
    match=re.search(r'^  '+re.escape(json.dumps(key))+r': ',text,re.M)
    if match:
        start=match.end();_,end=json.JSONDecoder().raw_decode(text[start:])
        return text[:start]+rendered+text[start+end:]
    start=text.index('{')+1
    return text[:start]+'\n  '+json.dumps(key)+': '+rendered+','+text[start:]
title='TextCraft v5 八卡正式训练：原指数信用，负尾方案暂缓'
binding_note=('八rank只读元数据RPC已核对实际v5模式/DP8及已加载源码；尚未自然导入的DT模块不冒称已实际核验，未追加模型或optimizer调用。'
    if data['actual_worker_bindings'] else '实际worker导入RPC尚待初始化自然完成，当前不称已通过实际绑定。')
header=f'''## {now} {title}

用户要求先八卡训练v5，再讨论负尾。独立run {data['job']['version']} 从基础权重/0轮提交，200迭代；原官方GRPO、逐事件原EOS d和指数Q/V/A及原v5优势仅方差缩放保持。不在乘g前缩放u，不采用对称概率比，不增加负尾裁剪。TextCraft32组×8/global64/epochs1/4更新、32k/B4/LoRA8/16、save5/keep2与原采样/LR保持；实际效率源继承8790a1b7，DT runner6a112224、actor acd3c8a5全局/分片分别记录、core fc2f992b/worker198f509c不改。

执行提交{data['execution_source_commit']}，方法原始提交4545c558，上游VERL {data['upstream_commit']}。21项原v5 CPU行为测试通过；旧绑定测试因硬编码旧actor/runner SHA而排除，当前验收来源逐文件哈希核对，未改数值测试。{binding_note}controller {data['controller']['pid']}/{data['controller']['birth']}，driver {data['job']['pid']}/{data['job']['birth']}；当前完成{data['completed']}/200，实际phase={data['phase']}。初始化/占卡不称已完成健康迭代。

连接时原v8控制器和driver均不存在，最后记录76轮/检查点75，退出原因未建立，未发送停止信号。历史日志/rollout/检查点保留，新八卡v5不拼接旧双卡v5。原TextCraft CPU服务复用作者入口恢复；三个已证实回退的cumem/gpu_worker/MRoPE文件恢复到既有验收SHA，不重装、不清缓存。记录含实际环境/源路径SHA、配置、PID出生、服务及恢复回执，见{result.name}和远端receipts/textcraft-v5-eight-gpu-20261019-v1。

共享检查点：{data['job']['checkpoints']}。原native process owner f71ff4d8只执行本次TextCraft v5，正常完整200轮及八rank原生checkpoint200齐全才结束，不自动重启或跳过失败。旧v7/梯度goal保持暂停，其他组合不自动启动。小时绘图textcraft保持ACTIVE并已切换新run；原PPO105/GRPO191为独立已停止历史对照，梯度沿已批准的明确标注√2估算规则，当前新八卡v5用全局实测，CSV保留原值/local及来源。

'''
env_header=f'''## {now} 本次v5八卡启动所需环境恢复

原v8控制器/driver不存在且8卡空闲，原TextCraft服务36005无监听；复用既有作者服务入口恢复，服务PID{data['service']['pid']}/出生{data['service']['birth']}。cumem.py/gpu_worker.py/gpu_model_runner.py实际SHA与记录的未修补版本完全匹配，分别恢复到既有验收SHA 7e557a8f/37db3830/6ed94945。新回执ROOT/receipts/textcraft-v5-eight-gpu-20261019-v1/{{official-service.json,restored-platform-files.json}}保留精确前后SHA与原owner来源。无安装、下载或缓存清理。严格SSH31146/HostKeyAlias32036，source原入口后合并实际launch环境，继续原VENV_PYTHON，不调用解析后的系统Python。方法/运行/实际导入与环境恢复分开记录；当前正式身份见RUNTIME_RECORD最新v5八卡条目。

'''
def prepend(text,block,marker):
    if text.startswith('## ') and marker in text.splitlines()[0]:
        end=text.find('\n## ',1)
        text=text[end+1:] if end>=0 else ''
    return block+text
runtime=Path('experiments/rl/current_runtime.json')
history=Path('experiments/rl/RUNTIME_RECORD.md')
environment=Path('experiments/rl/REMOTE_ENVIRONMENT.md')
for path,transform in [(runtime,update_json),(history,lambda s:prepend(s,header,title)),
        (environment,lambda s:prepend(s,env_header,'本次v5八卡启动所需环境恢复'))]:
    path.write_text(transform(path.read_text(encoding='utf-8')),encoding='utf-8')
    head=subprocess.check_output(['git','show','HEAD:'+path.as_posix()]).decode('utf-8')
    blob=subprocess.check_output(['git','hash-object','-w','--stdin'],input=transform(head).encode('utf-8')).decode().strip()
    subprocess.run(['git','update-index','--add','--cacheinfo','100644',blob,path.as_posix()],check=True)
print(json.dumps(dict(result=str(result),status=record['status'],completed=data['completed'],phase=data['phase'],
    worker_count=len(data['workers']),actual_binding_receipts=len(data['actual_worker_bindings']),unrelated_worktree_changes_staged=False),ensure_ascii=False))
