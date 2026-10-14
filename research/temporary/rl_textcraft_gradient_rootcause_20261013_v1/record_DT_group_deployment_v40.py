"""Record the authorized deployment separately from unresolved root-cause work."""
import datetime
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
job=json.loads((HERE/'DT-group-multiplier-v40-job.json').read_text(encoding='utf-8'))
verification=json.loads((HERE/'verify_DT_group_multiplier_v40-dt.json').read_text(encoding='utf-8'))
status_path=HERE/'read_DT_group_status_v40-dt.json'
status=json.loads(status_path.read_text(encoding='utf-8'))['report'] if status_path.exists() else None
worker_path=HERE/'verify_DT_group_workers_v40-dt.json'
workers=json.loads(worker_path.read_text(encoding='utf-8')) if worker_path.exists() else None
stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
report=dict(observed_at=stamp,job=job,interface_verification=verification,
    actual_status=status,actual_workers=workers,rootcause_repair_verified=False,
    semantic_changes=['Call fixed native GRPO for the trajectory multiplier; replace raw reward multiplier, not multiply twice.',
                      'Remove DT masked_whiten, with no replacement normalization.'],
    official_actor_core_optimizer_scheduler_checkpoint_unchanged=True,
    configuration='32k, LoRA8/16, B4/card, global64,4 native optimizer updates/iteration,330 iterations,save5,keep2 unchanged',
    fresh_base_model=True,old_DTPO_resumed=False,GRPO_PPO_rerun=False,hourly_automation='PAUSED',
    rootcause_scope='Actual common-return component -> actual optimizer update pattern -> cumulative internal directional propagation gain. No noise classification, no validation-score one-update trial.',
    existing_internal_evidence='Fixed same input and exact same incoming BF16 adjoint: parameter VJP at checkpoints5/40/80 =0.06040356/0.08634082/1.10390448. Directional sensitivity only; not a universal norm or accepted root cause.',
    counterevidence='On the full original late firstglobal64 coefficients, native total PG40/80 only2.06208/2.21412. Five-iteration net parameter displacement decreases late; neither same-direction step accumulation nor universal18x amplification is established.')
saved=HERE/'saved-fixed-seed-parameter-growth-v41.json'
if saved.exists():
    detail=json.loads(saved.read_text(encoding='utf-8'))
    report['additional_existing_evidence']=dict(receipt=str(saved),model_calls=0,
        layer_parameter_VJP_ratio_40_to_80={k:detail['growth']['40_to_80']['layers'][k]['norm_ratio'] for k in ('0','15','24','29','30','31')},
        lower16_gradient_squared_norm_fraction={cp:sum(v['energy_fraction'] for layer,v in case['layers'].items() if int(layer)<16)
                                               for cp,case in detail['cases'].items()},
        interpretation='Same incoming adjoint: amplification spreads to lower layers; not confined to the final projection. This localizes the sensitive path, but does not identify which historical updates produced it.')
report['goal_app_status_last_observed']='paused; latest user requested continued research; assistant has not marked it complete or changed its status'
dest=REPO/'experiments/rl/results_textcraft_DT_group_multiplier_20261014_v1.json'
dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
runtime=REPO/'experiments/rl/current_runtime.json'
current=json.loads(runtime.read_text(encoding='utf-8'))
key='20261014_DT_native_GRPO_multiplier_no_whiten_v1'
current[key]=dict(observed_at=stamp,**{k:report[k] for k in ['job','semantic_changes','rootcause_repair_verified','hourly_automation','rootcause_scope']},
                 actual_worker_receipt=workers.get('receipt') if workers else None,
                 detailed_receipt=str(dest),additional_existing_evidence=report.get('additional_existing_evidence'),
                 goal_app_status_last_observed=report['goal_app_status_last_observed'])
current={key:current.pop(key),**current}
runtime.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
phase='原worker身份/初始状态待回读' if workers is None else '原worker实际SHA、optimizer0/scheduler0及既有被动记录已回读'
entry=f'''## {stamp} 用户批准官方GRPO乘子与去DT白化，新DTPO已提交

版本{job['version']}，PID{job['pid']}/出生{job['birth']}，physical0/1，基础权重新开330迭代。方法patch ea2a470c，启动transport修正及部署commit {job['local_patch_commit']}；不是旧DTPO续训。{phase}；尚不把初始化或进程存活当完整更新通过。

仅两项语义改动：原完整batch调用固定VERL官方GRPO，原uid/traj_uid/reward/mask/std算法直接给乘子g；source=g*(-expm1(-d))、target=g、O/pad0，原环境奖励保持。取消DT出口masked_whiten，不加别的归一化或裁剪。两个实际变更文件是ray_trainer.py SHA63b0d623与dt_training_batch.py SHAa4e1f70b；actor3a65e173/corefc2f992b/worker9dfd6ce7/checkpointf65b6538及DT数值/内存版本保持，未加入分母候选。实际旧256行接口检查通过：128个官方零组乘子行在原DT解析路径全部为零；35个raw reward0但组乘子非零的行不能跳过DT；不是模型训练/根因验收。32k/LoRA8/16/B4/global64/4更新/330/save5/keep2及完整checkpoint状态合同保持。

原PPO2/3与GRPO4/5保持，6/7另有占用；小时检查PAUSED。只继续梯度范数加速上升根因，不研究共同分量是不是噪声；用户已明确要看共同分量如何影响更新模式与累积增强。固定seed内部增益证据与实际全PG反证都保留，新方法训练提交不等于根因已证实或修好。详细本地回执：experiments/rl/results_textcraft_DT_group_multiplier_20261014_v1.json。

补充仅CPU读已有v33/v35参数分片记录：同输入/同BF16输出adjoint的40→80，layer0/15/24/29/30/31参数VJP倍率13.206/8.230/9.899/5.517/3.489/1.622；所有层完整明细见saved-fixed-seed-parameter-growth-v41.json。下半16层梯度平方和占总平方和从40的86.42%升至80的95.05%，这是参数坐标分区，不是因果贡献比例。未新增模型调用或训练实验，尚不能指认历史哪个更新分量导致变化。get_goal实际仍返回paused；用户最新已要求继续研究，本轮已执行，未由助手完成/暂停/重设goal，不把旧文档的active当应用状态。

'''
for name in ('RUNTIME_RECORD.md','REMOTE_ENVIRONMENT.md'):
    p=REPO/'experiments/rl'/name
    text=p.read_text(encoding='utf-8')
    if text.startswith('## ') and '用户批准官方GRPO乘子与去DT白化' in text.split('\n',1)[0]:
        text=text[text.index('\n## ',1)+1:]
    p.write_text(entry+text,encoding='utf-8',newline='\n')
print(json.dumps(dict(receipt=str(dest),worker_readback=workers is not None,job=job['pid']),ensure_ascii=False))
