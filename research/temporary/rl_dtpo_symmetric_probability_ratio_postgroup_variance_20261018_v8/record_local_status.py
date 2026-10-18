"""Save observed v8 deployment status; surgically stage only this runtime entry."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
RECEIPTS=HERE/'receipts'
summary=json.loads((RECEIPTS/'submission-summary.json').read_bytes())
initial_path=RECEIPTS/'textcraft-actual-worker-imports-initial.json'
initial=json.loads(initial_path.read_bytes())
assert len(initial['results'])==8
assert all(r['DT_CREDIT_MODE']=='reward_target_symmetric_probability_ratio_postgroup_variance_v8'
    and r['DP_world_size']==8 for r in initial['results'])
assert not any(m.get('matches_prepared_source') is False for r in initial['results'] for m in r['modules'].values())
now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
automation=tomllib.loads(Path('C:/Users/Administrator/.codex/automations/textcraft/automation.toml').read_text(encoding='utf-8'))
plots={}
for task in ('textcraft','appworld'):
    folder=Path('F:/DeltaTraceBackups/plots')/(task+'-eight-gpu-symmetric-probability-ratio-postgroup-variance-20261018-v8')
    current=max(p for p in folder.iterdir() if (p/'plot-receipt.json').exists())
    snap=json.loads((current/'source-snapshot.json').read_bytes())
    assert 'global preclip' in snap['gradient_norm_scope']
    plots[task]={name:str(current/name).replace('\\','/') for name in (
        'algorithm-comparison-common-window.png','algorithm-comparison-full.png',
        'algorithm-comparison-full.svg','metrics-with-sources.csv','source-snapshot.json','plot-receipt.json')}
record=dict(recorded_at=now,status='v8_native_six_run_queue_submitted',
    upstream_commit='20bd331bdbc9026a5668e11362178e10ab7400c8',
    method_patch_commit='12cd52e16e944c6e9ea7fdcd9a336c098b186d9f',
    execution_source_commit=summary['source_commit'],controller=summary['controller'],
    manifest=summary['manifest'],current_progress=summary['jobs'],
    initial_worker_binding_receipt=initial_path.as_posix(),
    initial_worker_binding_SHA256=hashlib.sha256(initial_path.read_bytes()).hexdigest(),
    method_specification='experiments/rl/PLAN.md',method_version='DTPO symmetric-probability-ratio postgroup-variance v8',
    normalization='One original VERL masked_var scale on post-group/event-sum advantages, no mean subtraction; Q/V/A common scale',
    actor_gradient_norm='actor/grad_norm: native DTensor.full_tensor global preclip; actor/grad_norm_local: first-returned local shard',
    historical_shard_values_not_converted_or_directly_compared=True,
    source_and_CPU_verification=(RECEIPTS/'CPU-behavior-verification.json').as_posix(),
    source_and_CPU_tests=43,official_norm_world_sizes=[1,2,8],extra_model_quality_tests=0,
    first_formal_complete_iteration_verified=summary['jobs'][0]['completed_iterations']>=1,
    old_training_stopped=summary['superseded_stop'],prior_histories_spliced=False,
    latest_plots=plots,hourly_automation_status=automation['status'],old_v7_goal_status='paused',
    training_quality_improvement_claim=False)
after=RECEIPTS/'textcraft-actual-worker-imports-after-update.json'
if after.exists():
    value=json.loads(after.read_bytes())
    assert len(value['results'])==8
    assert all(r['DT_CREDIT_MODE']=='reward_target_symmetric_probability_ratio_postgroup_variance_v8'
        and r['DP_world_size']==8 and r['actor_scheduler_epoch']>=1 for r in value['results'])
    assert not any(m.get('matches_prepared_source') is False for r in value['results'] for m in r['modules'].values())
    record['after_natural_update_binding_receipt']=after.as_posix()
    record['actual_scheduler_epochs']=[r['actor_scheduler_epoch'] for r in value['results']]
    record['actual_optimizer_entries']=[r['actor_optimizer_entries'] for r in value['results']]
    values=summary['jobs'][0]['last_metric']['values']
    assert 'actor/grad_norm' in values and 'actor/grad_norm_local' in values
    record['first_observed_grad_norm_global']=values['actor/grad_norm']
    record['first_observed_grad_norm_local']=values['actor/grad_norm_local']
result_path=Path('experiments/rl/results_dtpo_symmetric_probability_ratio_postgroup_variance_20261018_v8.json')
result_path.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
key='20261018_symmetric_probability_ratio_postgroup_variance_v8_current'
runtime=Path('experiments/rl/current_runtime.json')
def update_json(text):
    old=json.loads(text);old.pop(key,None)
    return json.dumps({key:record,**old},indent=2,ensure_ascii=False)+'\n'
runtime.write_text(update_json(runtime.read_text(encoding='utf-8')),encoding='utf-8')
completed=summary['jobs'][0]['completed_iterations']
header=f'''## {now} DTPO v8：std(g·u) 与统一全局范数记录

方法提交12cd52e1，正式执行源{summary['source_commit']}，上游VERL 20bd331b。逐事件原EOS d→tanh(d/2)，先乘原官方已归一化g并汇总，再按原完整训练批次保留policy mask调用官方masked_var，用一个共同尺度缩放Q/V/A，不减均值、不预先缩放u。原PPO/GRPO方法、backward/clip/optimizer计算及已验收DT/资源效率路径保持；三算法都只把范数日志改为原DTensor.full_tensor全局裁剪前范数，另存actor/grad_norm_local。原ConsoleLogger仅这两个键用9位有效数字；原解析器支持科学计数法。旧两卡/八卡首rank分片历史不补估为全局，不与新全局曲线直接比较。

43项CPU行为检查通过；同一整数FP64梯度的原clip_grad_norm_返回值，经DTensor.full_tensor物化后在2/8 rank与单进程完全一致。此处零容差仅验证同一精确可表示平方和的记录接口，不是模型数值容差或质量测试。六组按各自实际父源隔离，PPO的critic/offload与所有原计算owner保持；各任务原生Hydra配置逐section核对工作量一致，32k/B4/LoRA8/16、200轮及TextCraft save5/AppWorld save2保持。原v7 controller和driver已由原stop owner退出，无非zombie存活，日志与原完整检查点保留。

新controller {summary['controller']['pid']}/{summary['controller']['birth']}，首组driver {summary['jobs'][0]['pid']}/{summary['jobs'][0]['birth']}。实际八rank初始绑定符合v8/DP8，当前原日志完成{completed}/200；未完成迭代不能称健康更新。原生自动顺序TextCraft DTPO v8→PPO→GRPO→AppWorld DTPO v8→PPO→GRPO，均从基础权重/0轮独占GPU0–7；仅完整exit0、1..200日志和八rank共享checkpoint200后交接，失败不跳过。实际来源、配置、PID出生、后续自然更新及图像路径见results_dtpo_symmetric_probability_ratio_postgroup_variance_20261018_v8.json。小时任务状态{automation['status']}，旧v7 goal仍暂停，不假报已完成。

'''
history=Path('experiments/rl/RUNTIME_RECORD.md')
def replace_current_block(text):
    if text.startswith('## ') and 'DTPO v8：std(g·u)' in text.splitlines()[0]:
        at=text.find('\n## ',1)
        return text[at+1:] if at>=0 else ''
    return text
history.write_text(header+replace_current_block(history.read_text(encoding='utf-8')),encoding='utf-8')
def stage_bytes(path,data):
    digest=subprocess.check_output(['git','hash-object','-w','--stdin'],input=data).decode().strip()
    subprocess.run(['git','update-index','--add','--cacheinfo','100644',digest,path],check=True)
head_runtime=subprocess.check_output(['git','show','HEAD:'+runtime.as_posix()]).decode('utf-8')
head_history=subprocess.check_output(['git','show','HEAD:'+history.as_posix()]).decode('utf-8')
stage_bytes(runtime.as_posix(),update_json(head_runtime).encode('utf-8'))
stage_bytes(history.as_posix(),(header+replace_current_block(head_history)).encode('utf-8'))
print(json.dumps(dict(result=str(result_path),completed=completed,unrelated_runtime_worktree_changes_staged=False)))
