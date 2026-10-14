"""Record measured localization and withdraw the unmatched numerical pair."""
import json
from pathlib import Path
from datetime import datetime

HERE=Path(__file__).resolve().parent
RL=HERE.parents[2]/'experiments/rl'
audit=json.loads((HERE/'native-PG-localization-audit-v2.json').read_bytes())
white=json.loads((HERE/'whitening-temporal-analysis-v2.json').read_bytes())
now=datetime.now().astimezone().isoformat()
entry={'observed_local':now,'status':'native_PG_localized_temporal_rootcause_unresolved',
    'goal_tool_observation':{'status':'active','updatedAt':1791937457,'goal_completed':False},
    'user_scope':'Continue autonomously until the accelerating gradient-norm cause is repaired, unless an actual user judgment is required. Existing-data quantitative accounting precedes expensive trials.',
    'localization_receipt':str((HERE/'native-PG-localization-audit-v2.json').resolve()),
    'localization_source_commit':audit['diagnostic_source_commit'],
    'localization_source_SHA256':audit['diagnostic_source_SHA256'],
    'group_norms':audit['group_gradient_statistics']['norms'],
    'reconstructed_PG_norm':audit['reconstructed_sum_PG_norm'],
    'signed_projection_on_sum':audit['signed_projection_on_sum'],
    'factual_point':audit['factual_point'],
    'earlier_pair_correction':audit['withdrawn_evidence'],
    'whitening_receipt':str((HERE/'whitening-temporal-analysis-v2.json').resolve()),
    'whitening_inferred_iterations':white['usable_inferred_iterations'],
    'whitening_excluded_truncated_iterations':white['excluded_truncated_success_iterations'],
    'whitening_calibration81':white['snapshot81_calibration'],
    'whitening_findings':white['findings'],
    'rootcause_resolved':False,'accepted_repair':None,'deployed_repair':None,
    'normalizer_candidate_acceptance':'Confirmed denominator/reference defect, temporal sufficiency not established; candidate stopped, not accepted/deployed.',
    'hourly_automation':'PAUSED','GRPO_rerun':False,'old_DTPO_resumed':False,
    'diagnostic_driver':{'pid':1230244,'birth':1791935966.16,'physical_GPUs':[0,1],
        'phase':'complete_gradient_localization_not_repair','latest_observed_alive':False},
    'limits':audit['limits']+['Historical B4 positions/denominators and temporal parameter directions still require data-based accounting; no new long training has been submitted.']}
for filename in ['current_runtime.json','results_textcraft_gradient_rootcause_20261013_v1.json']:
    path=RL/filename
    document=json.loads(path.read_bytes())
    if filename=='current_runtime.json':
        document={'20261014_gradient_localization_v3':entry,**document}
        document['20261013_focused_gradient_goal_v2'].update(status='active',completed=False,rootcause_resolved=False,latest_user_constraint=entry['user_scope'])
    else:
        document.update(observed_local=now,status=entry['status'],goal_active=True,goal_status_from_tool='active',rootcause_fully_resolved=False,
            accepted_repair=None,deployed_repair=None,repair_deployed=False,latest_goal_tool_observation=entry['goal_tool_observation'],gradient_localization=entry)
    path.write_text(json.dumps(document,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
note=f'''## {now} 原生PG分组定位与旧配对证据纠正，goal继续active

用户要求不因根因尚未完成而结束goal；当前工具返回active（updatedAt1791937457）。小时检查仍PAUSED，旧DTPO和过早候选仍停止，不重做GRPO。原物理0/1的有界定位1230244/1791935966.16已完成退出，无optimizer/scheduler/rollout/DT/validation调用；原两rank496个trainable参数摘要与scheduler80保持。

在原iteration81第一global64、原checkpoint80、原B4分母及完整loss mask下，三个互斥token系数分组之和精确等于原优势。FP64原分片mesh梯度范数：successful target2.342079886、successful source0.682939703、zero-reward policy0.075419341；保留交叉内积后总PG2.208345150。source对总PG整体抵消。该结论只定位晚期批次，不证明85轮加速原因；全分片范数与日志原native preclip返回均值分开。源commit {audit['diagnostic_source_commit']}，harness SHA{audit['diagnostic_source_SHA256']}。

重要纠正：历史legacy PG0.9583064与candidate0.1620889两次测量的rank1实际entropy/KL/clip不同，未处于同一前向/PPO点；撤回5.9倍作为配对修复效果或因果证据。原记录保留，独立完整global64 scalar/dL/dlogp分母参考验算仍成立；这项旧证据不一致不能冒充原训练根因。

只读原被动记录恢复raw-credit moments，并用原三位小数max的舍入区间推断82轮retained policy数量；11/21/29轮存在成功历史截断而排除。iteration81原raw A全部选定值精确对应，N推断误差约9 token落在舍入区间内。raw标准差从早期约0.1565升到晚期0.3678，不支持白化分母塌缩。政策token数量、target概率/密度变化需要结合实际B4权重及参数方向继续验算，不能把系数或输出logit范数的倍率直接当参数梯度因果比例。

证据：{entry['localization_receipt']}；{entry['whitening_receipt']}。accepted/deployed repair均无，根因目标未完成，继续自主排查。

'''
path=RL/'RUNTIME_RECORD.md'
path.write_text(note+path.read_text(encoding='utf-8'),encoding='utf-8')
path=RL/'REMOTE_ENVIRONMENT.md'
env_note=f'''## {now} 原生PG定位诊断完成退出

1230244/1791935966.16的物理0/1有界原生PG分组定位已完成退出，无optimizer或scheduler更新；旧DTPO与过早候选保持停止，正式GRPO未改，小时检查PAUSED、修复goal active。不是根因已验收或新正式部署。后续用设备前重新检查实际物理占用。源码与回执见RUNTIME_RECORD顶部。

'''
path.write_text(env_note+path.read_text(encoding='utf-8'),encoding='utf-8')
(HERE/'gradient-localization-record-v3.json').write_text(json.dumps(entry,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'status':entry['status'],'goal':'active','old_pair_withdrawn':True,'rootcause_resolved':False}))
