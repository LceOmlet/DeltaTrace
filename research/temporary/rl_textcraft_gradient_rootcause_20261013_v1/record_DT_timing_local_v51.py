"""Persist observed DT audit and current diagnostic scope in existing runtime records."""
from datetime import datetime
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
now=datetime.now().astimezone().isoformat()
remote=json.loads((HERE/'record_DT_timing_binding_v51-dt.json').read_text(encoding='utf-8'))
audit=remote['report']
stop=json.loads((HERE/'stop_PPO_for_authorized_diagnostic_v54-dt.json').read_text())['report']
diagnostic=json.loads((HERE/'common-update-feedback-v47-job.json').read_text())
record=dict(observed_at=now,DT_timing_audit=remote,rootcause_goal_status='active',
    hourly_automation='PAUSED',accepted_rootcause_repair=None,
    PPO_temporarily_stopped=stop,PPO_native_resume_prepared=str(HERE/'resume_native_PPO_after_diagnostic_v55-dt.json'),
    finite_internal_gain_diagnostic=diagnostic,
    diagnostic_scope='Conditional old checkpoint80 native updates followed by same-input/same-BF16-adjoint VJP; not an entire-history causal proof. No validation-score, DT or GRPO model call.',
    user_authorized='Temporarily stop PPO, perform the finite diagnostic on2/3, then restore PPO with original full native checkpoint15.')
path=ROOT/'experiments/rl/results_textcraft_DT_timing_version_audit_20261014_v1.json'
path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
runtime=ROOT/'experiments/rl/current_runtime.json'
old=json.loads(runtime.read_text(encoding='utf-8-sig'))
key='20261014_DT_actual_historical_timing_audit_v51'
value=dict(observed_at=now,status=audit['status'],detailed_receipt=str(path),remote_runtime_override=remote['receipt'],
    DT_first_iteration=audit['first_iteration'],DT_timing_ratios=audit['ratios'],
    actual_DT_imports=audit['unchanged_DT_imports_verified'],actual_worker_identities=[{k:w[k] for k in ('pid','birth','rank')} for w in audit['actual_workers']],
    identical_original_first_rollout_bytes=True,rootcause_goal_status='active',hourly_automation='PAUSED',rootcause_repair_verified=False,
    PPO_status=stop['status'],PPO_checkpoint=stop['checkpoint'],finite_native_gain_diagnostic=diagnostic)
runtime.write_text(json.dumps({key:value,**{k:v for k,v in old.items() if k!=key}},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
text=f'''## {now} DT历史版本与当前实际导入完成核验；PPO按用户决定暂时停下

原10月12首轮与本次首轮的256行原生rollout文件SHA31ba5b7e完全相同，151个reward1、105个reward0。实际DT请求151→236；保留135条正乘子请求的原IDs/masks逐字节相同，16条旧正回报请求不再需要DT，新增101条负乘子请求。旧请求实际causal tokens299296，新934911（3.12370倍）；新增负乘子请求平均6634.32token，正乘子请求平均1961.81token。两rank补齐B4处理量322728→977096（3.02761倍），原TaskRunner timing_s/adv357.424→1095.178秒（3.06409倍）；每rank原批次秒数约318→1039。原日志路径/行号、原逐token输入文件及SHA均在v48/v50/v52原回执，完整证据{path.as_posix()}。

当前两worker1426979/1791977192.37和1429032/1791977210.63通过原Ray只读RPC实际回读：GDN3f51f5f7、runner198f78db、decoder44153c17、retainedcapture5ad0bf9b、acceleratedcapture993ae72a，与10月12已运行版本对齐；capture_backend仍accelerated.qwen35.qwen35_code_local_capture，offload/compiledGDN/pinnedhost/FA与GDNsuffix/compact/prefix开关为True。当前retain方法实际包含原gate_destination CPU保留分支，未退回历史慢版或旧溢出路径。正式DT仍是用户批准的官方GRPO乘子/no-whiten版本de8b6b27与canonical01bf6fa5；本次只读审计未改变训练代码或参数。新增长请求解释主要工作量增幅，token计数不是精确耗时模型，未据此声称底层各算子性能完全相同；两次first B4组成不相同且物理卡不同。远端完成覆盖{remote['receipt']}已链接ROOT三份manifest，旧launch/source保持。

goal实际active（覆盖较早paused读数），小时检查仍PAUSED，梯度范数根因未完成/未验收。用户明确批准暂时停下PPO并在有限诊断后原生完整恢复；原2671393/1791949551.27已停止、无存活成员，checkpoint15的actor/critic双rank model/optimizer/extra state/data及marker保留，未制造或删除checkpoint。原生resume_path恢复命令已准备，仅改变resume_mode/resume_from_path，尚未恢复。

有限内部增益诊断v47 PID{diagnostic['pid']}/出生{diagnostic['birth']}，physical2/3，诊断commit{diagnostic['diagnostic_commit']}，main source61e1f086、原observer3cbefaf3。从旧80检查点读取原模型/optimizer/scheduler及原81输入/RNG，只对已测同正回报组PG分量做原生更新前后传播增益的条件比较；2臂各4笔原更新，不是验证分数试验或旧正式DTPO续训。当前初始化/加载阶段，无完成结果；当前正式DT0/1、GRPO4/5和其他项目6/7保持。此有限比较也不能直接证明85迭代完整历史因果或新方法从头已修好。

'''
for name in ('RUNTIME_RECORD.md','REMOTE_ENVIRONMENT.md'):
    p=ROOT/'experiments/rl'/name
    p.write_text(text+p.read_text(encoding='utf-8-sig'),encoding='utf-8')
print(str(path))
