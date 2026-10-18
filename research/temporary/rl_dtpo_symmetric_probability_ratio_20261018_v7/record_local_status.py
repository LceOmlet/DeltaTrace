"""Record this deployment without staging unrelated runtime-document edits."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tomllib
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
RECEIPTS=HERE/'receipts'
summary=json.loads((RECEIPTS/'submission-summary.json').read_bytes())
initial=json.loads((RECEIPTS/'textcraft-actual-worker-imports-initial.json').read_bytes())
assert len(initial['results'])==8
assert all(r['DT_CREDIT_MODE']=='reward_target_symmetric_probability_ratio_v7' for r in initial['results'])
assert all(r['DP_world_size']==8 and r['actor_optimizer_entries']==r['actor_scheduler_epoch']==0 for r in initial['results'])
automation=tomllib.loads(Path('C:/Users/Administrator/.codex/automations/textcraft/automation.toml').read_text(encoding='utf-8'))
assert automation['status']=='ACTIVE' and 'rl_dtpo_symmetric_probability_ratio_20261018_v7/plot_both_tasks.py' in automation['prompt']
now=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
plots={}
for task in ('textcraft','appworld'):
    folder=Path('F:/DeltaTraceBackups/plots')/(task+'-eight-gpu-symmetric-probability-ratio-20261018-v7')
    current=max(p for p in folder.iterdir() if (p/'plot-receipt.json').exists())
    plots[task]={name:str(current/name).replace('\\','/') for name in (
        'algorithm-comparison-common-window.png','algorithm-comparison-full.png',
        'algorithm-comparison-full.svg','metrics-with-sources.csv','source-snapshot.json','plot-receipt.json')}
record=dict(recorded_at=now,status='symmetric_v7_six_run_goal_active',
    upstream_commit='20bd331bdbc9026a5668e11362178e10ab7400c8',
    local_patch_commit=summary['source_commit'],controller=summary['controller'],
    manifest=summary['manifest'],current_progress=summary['jobs'],
    initial_worker_binding_receipt=str(RECEIPTS/'textcraft-actual-worker-imports-initial.json').replace('\\','/'),
    initial_worker_binding_SHA256=hashlib.sha256((RECEIPTS/'textcraft-actual-worker-imports-initial.json').read_bytes()).hexdigest(),
    actual_initial_DP_world_sizes=[r['DP_world_size'] for r in initial['results']],
    actual_initial_optimizer_entries=[r['actor_optimizer_entries'] for r in initial['results']],
    actual_initial_scheduler_epochs=[r['actor_scheduler_epoch'] for r in initial['results']],
    method_specification='experiments/rl/PLAN.md',method_version='DTPO symmetric-probability-ratio v7',
    credit_variance_normalization=False,official_GRPO_group_normalization_preserved=True,
    source_and_CPU_verification=str(RECEIPTS/'CPU-behavior-verification.json').replace('\\','/'),
    source_and_CPU_tests=41,first_natural_DT_imports_verified=False,
    first_formal_complete_iteration_verified=False,extra_rollouts=0,
    stop_receipt=summary['superseded_stop'],prior_histories_spliced=False,
    physical_phase_report=str(RECEIPTS/'native-utilization-phase-summary.json').replace('\\','/'),
    hourly_automation={k:automation[k] for k in ('id','name','status','rrule','target_thread_id')},
    latest_plots=plots,local_submission_receipt=str(RECEIPTS/'submission-summary.json').replace('\\','/'),
    completion='All six native 200-iteration runs, final eight-rank shared checkpoint200 and both task final figures',
    goal_status='active',training_quality_improvement_claim=False)
for phase,name in (('after_DT','textcraft-actual-worker-imports-after-DT.json'),('after_update','textcraft-actual-worker-imports-after-update.json')):
    path=RECEIPTS/name
    if path.exists():
        value=json.loads(path.read_bytes())
        record['actual_'+phase+'_receipt']=str(path).replace('\\','/')
        record['actual_'+phase+'_scheduler_epochs']=[r['actor_scheduler_epoch'] for r in value['results']]
        if phase=='after_DT':
            assert all(r['modules']['counterfactual'].get('matches_prepared_source') for r in value['results'])
            record['first_natural_DT_imports_verified']=True
        if phase=='after_update':
            record['first_formal_complete_iteration_verified']=summary['jobs'][0]['completed_iterations']>=1
result_path=Path('experiments/rl/results_dtpo_symmetric_probability_ratio_20261018_v7.json')
result_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

KEY='20261018_symmetric_probability_ratio_v7_six_current'
OLD='20261017_six_fresh_threads8_eight_GPU_200_current'
def put_value(text,key,value):
    match=re.search(r'^  '+re.escape(json.dumps(key))+r':\s*',text,re.M)
    encoded=json.dumps(value,ensure_ascii=False,indent=2).replace('\n','\n  ')
    if match:
        _,length=json.JSONDecoder().raw_decode(text[match.end():])
        return text[:match.end()]+encoded+text[match.end()+length:]
    start=text.index('{')+1
    return text[:start]+'\n  '+json.dumps(key)+': '+encoded+','+text[start:]

runtime=Path('experiments/rl/current_runtime.json')
def update_json(text):
    data=json.loads(text.lstrip('\ufeff'))
    prior=dict(data[OLD])
    prior.update(status='superseded_by_symmetric_probability_ratio_v7',active=False,
        historical_driver_stopped=True,stop_receipt=summary['superseded_stop'],replacement_key=KEY)
    return put_value(put_value(text,OLD,prior),KEY,record)
runtime.write_bytes(update_json(runtime.read_bytes().decode('utf-8')).encode('utf-8'))
head_runtime=subprocess.check_output(['git','show','HEAD:'+runtime.as_posix()]).decode('utf-8')

completed=summary['jobs'][0]['completed_iterations']
header=f'''## {now} 对称概率比 DTPO v7 已提交；八卡六组从零自动队列已启动

用户批准的纯 DT 对称概率比 v7 已按唯一 PLAN 实现并推送，方法提交673b337fa1c78bf4772290966189e80a2b568d8e，上游 VERL 20bd331bdbc9026a5668e11362178e10ab7400c8。原 d→tanh(d/2)，取消信用乘组优势前后的均值/方差缩放，官方 GRPO 组内归一化保持；原多事件、观察/未来/self mask、DT计算、actor/core/采样和资源效率版本保持，无新增采样。41项CPU检查通过；32k CPU张量覆盖不冒称新GPU容量或质量验收。候选完整绑定与配置见本版本source-prepared及six-combination-manifest。

旧v6 controller2004205/1792263400.01及driver2004329/1792263400.35已依原stop owner退出，确认无非zombie存活；旧日志/检查点保留，不拼接。新controller2854542/1792312285.09，首组TextCraft DTPO driver2854659/1792312285.4；八个实际worker均v7/DP8，初始optimizer条目和scheduler均0。当前原日志完成{completed}/200；初始化与进程存活不称已完成健康更新。初始绑定中lazy DT尚未导入，是否已自然执行及后续绑定以results_dtpo_symmetric_probability_ratio_20261018_v7.json实际回执为准。

顺序TextCraft DTPO对称v7→原PPO→原GRPO→AppWorld DTPO对称v7→原PPO→原GRPO；六组均基础权重、独占0–7、200完整迭代。六份原生Hydra配置通过任务内工作量核对，TextCraft32×8/global64/epochs1/4更新/save5，AppWorld40×6/global32/epochs2/原残批/test5/共享save2保持；原32k/B4/LoRA8/16、seed/LR/采样/keep2保持。复用原f71ff4d8队列，仅原生exit0、日志1..200及八rank完整checkpoint200齐全后切换，失败不跳过。hourly textcraft已ACTIVE切换v7两任务三算法同图，缺失不补0，初始两任务共同窗口均空。goal保持active，六组最终训练、共享完整checkpoint200和两任务最终图完成前不标完成。

'''
history=Path('experiments/rl/RUNTIME_RECORD.md')
history.write_bytes(header.encode('utf-8')+history.read_bytes())
head_history=subprocess.check_output(['git','show','HEAD:'+history.as_posix()])

def stage_bytes(path,data):
    digest=subprocess.check_output(['git','hash-object','-w','--stdin'],input=data).decode().strip()
    subprocess.run(['git','update-index','--add','--cacheinfo','100644',digest,path],check=True)
stage_bytes(runtime.as_posix(),update_json(head_runtime).encode('utf-8'))
stage_bytes(history.as_posix(),header.encode('utf-8')+head_history)
print(json.dumps(dict(result=str(result_path),completed_iterations=completed,goal_status='active',
    unrelated_runtime_worktree_changes_staged=False),ensure_ascii=False))
