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
        assert len(value['results'])==8
        assert all(r['DT_CREDIT_MODE']=='reward_target_symmetric_probability_ratio_v7' and r['DP_world_size']==8 for r in value['results'])
        assert not any(m.get('matches_prepared_source') is False for r in value['results'] for m in r['modules'].values())
        record['actual_'+phase+'_receipt']=str(path).replace('\\','/')
        record['actual_'+phase+'_scheduler_epochs']=[r['actor_scheduler_epoch'] for r in value['results']]
        record['actual_'+phase+'_optimizer_entries']=[r['actor_optimizer_entries'] for r in value['results']]
        if phase=='after_DT':
            assert all(r['modules']['counterfactual'].get('matches_prepared_source') for r in value['results'])
            record['first_natural_DT_imports_verified']=True
        if phase=='after_update':
            assert all(r['actor_scheduler_epoch']>=1 and r['actor_optimizer_entries']>0 for r in value['results'])
            record['first_formal_complete_iteration_verified']=summary['jobs'][0]['completed_iterations']>=1
owner_path=RECEIPTS/'native-verification-owners.json'
if owner_path.exists():
    record['native_checkpoint_metadata_verification_owner']=json.loads(owner_path.read_bytes())
norm_path=RECEIPTS/'native-norm-once-v7-v1/completed-accounting.json'
scale_path=RECEIPTS/'native-readout-advantage-scale-first-three-v1.json'
if norm_path.exists() and scale_path.exists():
    native_norm=json.loads(norm_path.read_bytes())
    scale=json.loads(scale_path.read_bytes())
    assert native_norm['original_methods_restored_all_eight_ranks']
    assert native_norm['training_math_changes']==0
    assert all(r['value']['completed_iteration_before_update']==2 for r in native_norm['captures'])
    record['native_iteration3_gradient_magnitude_observation']=dict(
        receipt=str(norm_path).replace('\\','/'),SHA256=hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        native_local_norm_mean_by_rank=native_norm['native_local_norm_mean_by_rank'],
        materialized_global_norm_mean=native_norm['materialized_global_norm_mean'],
        original_methods_restored_all_eight_ranks=True,added_model_backward_optimizer_calls=0)
    record['existing_native_readout_advantage_scale']=dict(
        receipt=str(scale_path).replace('\\','/'),SHA256=hashlib.sha256(scale_path.read_bytes()).hexdigest(),
        scope=scale['scope'],summary=scale['summary'],
        coefficient_RMS_ratio_is_not_gradient_norm_ratio=True,
        full_gradient_gap_causal_decomposition_completed=False,method_or_hyperparameter_changes=0)
historical_path=RECEIPTS/'historical-gradient-magnitude-analysis-v1.json'
if historical_path.exists():
    historical=json.loads(historical_path.read_bytes())
    record['historical_v1_through_v7_gradient_audit']=dict(
        receipt=str(historical_path).replace('\\','/'),
        SHA256=hashlib.sha256(historical_path.read_bytes()).hexdigest(),
        native_norm_scope=historical['norm_scope'],
        version_identity_note=historical['version_identity_note'],
        summaries=historical['summaries'],
        same_eight_GPU_v6_to_v7_step3_ratio=historical['comparison']['observed_ratio_v6_to_v7'],
        native_actor_core_unchanged=True,added_model_backward_optimizer_calls=0,
        method_or_hyperparameter_changes=0,
        all_magnitude_gap_components_quantified=False)
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
milestone='首轮自然DT和原生更新已完成' if record['first_formal_complete_iteration_verified'] else '八卡六组从零自动队列已启动'
if 'native_iteration3_gradient_magnitude_observation' in record:
    milestone='第3轮原生梯度数值及已有优势尺度记录'
if 'historical_v1_through_v7_gradient_audit' in record:
    milestone='历史v1至v7原生日志梯度尺度核对'
progress_note=''
if record['first_formal_complete_iteration_verified']:
    metric=summary['jobs'][0]['last_metric']
    values=metric['values']
    phases=sorted({r['phase'] for r in summary['jobs'][0]['workers']})
    checkpoint=summary['jobs'][0].get('checkpoint_step')
    progress_note=f"自然DT结束及首次更新后两份只读RPC回执均覆盖八rank，实际counterfactual/reward_readout来源符合v7；首次更新后各rank scheduler=1、optimizer条目496。原TaskRunner完整指标来源为{metric['sources']}，当前环境训练回报{values['episode/reward/mean']}、熵{values['actor/entropy_loss']}、梯度范数日志{values['actor/grad_norm']:.3f}、KL日志{values['actor/kl_loss']:.3f}，后两者三位小数不表示精确零；单轮总耗时{values['timing_s/step']}秒。本次采集worker阶段{phases}，原保存标记{checkpoint}。未追加DT/model/optimizer诊断调用，也不以首轮结果宣称质量改进；原save5保持。"
if 'native_iteration3_gradient_magnitude_observation' in record:
    progress_note+='\n\n用户明确问题为梯度严重低于历史GRPO/PPO的0.02–0.04，而非是否精确零。复用原一次性native rank metadata observer，仅绑定当前PID出生/八rank/原local minibatch8；第3轮原本计划的4次更新返回值已记录，八rank update_policy和_optimizer_step均自动恢复，未追加模型、反向、优化器、scheduler或collective计算。八rank原裁剪前分片范数均值0.000371570–0.000466658；逐次按原NormPartial norm_type2定义合并八片后，全球范数均值0.001174096。此值另记，不覆盖原三位小数曲线或混入旧两卡全局范数。已有readout原生JSON报告前三批显示source优势RMS/同token广播g的RMS=0.043322、0.044294、0.043140；含self target后=0.088468、0.094661、0.090908。self target只占约0.85%–0.88% token。范围为非零g请求的完整policy source及target，含原DP padding，并非最终保留训练minibatch；不能把系数RMS比率直接当作梯度比率。原actor/core/归约未改变，现有证据确认优势信号压缩，尚未将全部梯度差距作分量因果分解，不改方法或超参。'
if 'historical_v1_through_v7_gradient_audit' in record:
    progress_note+='\n\n按用户要求，仅复用原read_job_logs/parse_logs读取历史正式身份并归约已有回执：原v1完成85轮，grad中位0.078、最大1.743；组乘子无白化中间版实际run后缀v1（不虚构独立v2部署）完成6轮，中位0.004；v3完成45轮，中位0.025；v4撤回0轮无梯度；v5完成79轮，中位0.021；v6两卡5轮中位0.013、八卡64轮中位0.007。v7第3轮原rank0分片0.000379238，对同八卡v6第3轮0.007约小18.46倍。旧真实输入标准差0.201244、原v3方差缩放4.9691倍，v3首实际PPO输入缩放5.40242倍；支持优势尺度为关键变化。v3第24轮单个-173112.765625信用占已记录向量平方和99.999907%，属于极端项支配方差的另一现象。v7源信用系数很小且不再缩放，不能把这两种小梯度机制混为一谈；现有数据未精确拆分18.46倍的每项贡献。此次未重启训练、重跑历史模型/反向/优化器或改信用公式。'
header=f'''## {now} 对称概率比 DTPO v7：{milestone}

用户批准的纯 DT 对称概率比 v7 已按唯一 PLAN 实现并推送，方法提交673b337fa1c78bf4772290966189e80a2b568d8e，上游 VERL 20bd331bdbc9026a5668e11362178e10ab7400c8。原 d→tanh(d/2)，取消信用乘组优势前后的均值/方差缩放，官方 GRPO 组内归一化保持；原多事件、观察/未来/self mask、DT计算、actor/core/采样和资源效率版本保持，无新增采样。41项CPU检查通过；32k CPU张量覆盖不冒称新GPU容量或质量验收。候选完整绑定与配置见本版本source-prepared及six-combination-manifest。

旧v6 controller2004205/1792263400.01及driver2004329/1792263400.35已依原stop owner退出，确认无非zombie存活；旧日志/检查点保留，不拼接。新controller2854542/1792312285.09，首组TextCraft DTPO driver2854659/1792312285.4；八个实际worker均v7/DP8，初始optimizer条目和scheduler均0。当前原日志完成{completed}/200；初始化与进程存活不称已完成健康更新。初始绑定中lazy DT尚未导入，是否已自然执行及后续绑定以results_dtpo_symmetric_probability_ratio_20261018_v7.json实际回执为准。

{progress_note}

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
