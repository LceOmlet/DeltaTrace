"""Record the withdrawn conjecture, candidate stop, and existing-data accounting."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RL = HERE.parents[2]/'experiments/rl'
analysis_path = HERE/'gradient-growth-accounting-v2.json'
analysis = json.loads(analysis_path.read_bytes())
stop_path = HERE/'stop_premature_temporal_candidate_v1.json'
stop = json.loads(stop_path.read_bytes())['report']
observed = analysis['observed_local']
entry = {
    'observed_local': observed,
    'status': 'premature_temporal_candidate_stopped_causal_claim_withdrawn_existing_data_analysis',
    'user_scope': 'Only explain and repair persistent, increasingly steep DTPO gradient-norm growth. Quantitatively check existing data before trusting a conjecture or committing to expensive training. Arithmetic, fits, approximate evidence and code inspection have distinct scopes; no one calculation is mandatory or sufficient.',
    'candidate_stop': stop,
    'analysis_report': str(analysis_path.resolve()),
    'plot': analysis['artifacts']['plot'], 'CSV': analysis['artifacts']['CSV'],
    'trend_conclusion': analysis['trend_conclusion'],
    'component_budget': analysis['saved_late_component_budget'],
    'denominator_accounting': analysis['denominator_accounting'],
    'fixed_input_parameter_location': analysis['fixed_input_parameter_location'],
    'evidence_gap': analysis['remaining_evidence_gap'],
    'rootcause_resolved': False, 'accepted_repair': None, 'deployed_repair': None,
    'hourly_automation': 'PAUSED', 'GRPO_rerun': False, 'old_DTPO_resumed': False,
    'goal_tool_observation': {'status':'blocked','updatedAt':1791908298,
        'goal_completed':False,'status_changed_by_this_correction':False,
        'note':'Read-only goal tool still returns blocked; manual user-directed investigation continued. No completed root cause or external impasse is asserted.'},
    'normalizer_candidate': {'version':'textcraft-global-token-mean-20261013-v3',
        'patch_commit':'56e69825fc45aa6cd5f3ef6a3bb48680bb22843d',
        'actor_SHA256':'ec0924e6abe561542ddd863bcf7040b995b19a34e500394a0604db6fc1dc5941',
        'worker_SHA256':'b64b196e8079c38ea4b94cd9d911211f7aa7fa13434966d18c649eaab558e8ce',
        'accepted_as_temporal_rootcause_repair':False},
}

path = RL/'current_runtime.json'
runtime = json.loads(path.read_bytes())
runtime = {'20261014_gradient_accounting_correction_v2':entry, **runtime}
old = runtime['20261014_DT_temporal_candidate_v1']
old['job']['status'] = stop['status']
old['job']['hypothesis_status'] = 'withdrawn_before_analytical_magnitude_check'
old.update(status=stop['status'], latest_observed_alive=False,
    latest_observed_unix=stop['finished_unix'], stop_receipt=str(stop_path.resolve()),
    goal_status='blocked as returned by goal tool; objective unfinished',
    repair_acceptance='not accepted; withdrawn before temporal data collection')
goal = runtime['20261013_focused_gradient_goal_v2']
goal.update(status='blocked as returned by goal tool; user-directed investigation ongoing',
    candidate_initial_binding='historically verified; candidate now stopped',
    latest_user_constraint=entry['user_scope'], rootcause_resolved=False, completed=False)
path.write_text(json.dumps(runtime,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

path = RL/'results_textcraft_gradient_rootcause_20261013_v1.json'
result = json.loads(path.read_bytes())
result.update(observed_local=observed,status=entry['status'],goal_active=False,
    goal_status_from_tool='blocked',rootcause_fully_resolved=False,
    accepted_repair=None,deployed_repair=None,repair_deployed=False,
    analytical_correction=entry,
    next_evidence=[entry['evidence_gap']])
if 'running_temporal_candidate' in result:
    result['running_temporal_candidate'].update(status=stop['status'],
        latest_observed_alive=False,stop_receipt=str(stop_path.resolve()))
if 'new_focused_goal' in result:
    result['new_focused_goal'].update(status='blocked as returned by goal tool; objective unfinished',
        latest_user_constraint=entry['user_scope'],candidate_status=stop['status'])
result['latest_goal_tool_observation'] = entry['goal_tool_observation']
path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

note = f'''## {observed} 梯度增长验算v2：撤回过早因果链并停止候选

用户明确：新增假设先用既有信息检验是否足以解释梯度范数增长的量级和加速；不能把算出的数或拟合函数当成精确、充分或唯一的验算方法；不通过两三天一轮的暴力训练试错，不偏离本异常。

已核对PID出生与独立进程组，停止提前提交的候选2510247/1791907554.02。原现场/source/launch/运行覆盖保留，无活跃成员残留、未制造检查点，正式GRPO3174448/1791793629.73未受影响。候选launch记为stopped_pending_analytical_norm_accounting；ROOT三正式manifest未改。旧有缺陷DTPO保持停止，小时检查PAUSED，不重做GRPO、不调超参或改信用方法。

仅用既有85轮原TaskRunner记录分析：10轮中位数从0.035、0.041、0.0495、0.0685、0.0855、0.1045、0.181到0.335，最后81–85五轮为0.502。整体水平与离散程度均增长。前50/70轮二次拟合不能解释后段幅度；全85轮二次项系数相对前50轮增加约8倍，指数拟合参数也随窗口改变。现有数据不能认定固定二次或指数规律，拟合不作为根因证据。

原后期第一global64分项梯度记录：PG0.9583064、已加权entropy0.0013804、已加权KL0.0017162。保留交叉内积后总范数0.9576443；两辅助项能影响范数的三角上界仅PG的0.323%。该点由PG主导，不冒充85轮每一更新分项。原调用链是preclip L2返回并取均值，不是跨迭代累计平方。现有固定输入checkpoint5/40/80观察中LoRA B梯度主导，80时A仅占平方范数0.655%；参数/Adam矩增长不能直接替代当前梯度证据。

原分母错误确实影响某一批；既有PG配对0.9583064→0.1620889只证明该点，不解释整段增长。CPU只读原iteration81数据，四个global64在old-policy点的logprob梯度系数L2倍率5.143/1.921/2.588/2.493；这是系数向量，不是参数范数。完整256白化系数二阶矩约0.9999926。均匀缩短N_j不会改变B4相对权重；“长度下降/target占比上升所以持续放大”因果链撤回。方向、相关项及其时间变化需证据，不能从scalar loss、原始信用RMS或单点范数直接推出；也不要求只有精确Gram矩阵这一种后续分析途径。

源码与完整SHA及量级/拟合局限见 {analysis_path.resolve()}；图 {analysis['artifacts']['plot']}；原日志路径/行号CSV {analysis['artifacts']['CSV']}；停止回执 {stop_path.resolve()}。当前normalizer candidate仍56e69825/ec0924e6/b64b196e，确认分母缺陷与容量/参考通过不等于持续异常根因已修复。accepted/deployed repair均无，目标未完成。goal工具最新仍返回blocked(updatedAt1791908298)，本轮未设置blocked/complete，手动按用户最新指令继续分析；不把历史active标签当当前工具状态。

'''
path = RL/'RUNTIME_RECORD.md'
path.write_text(note+path.read_text(encoding='utf-8'),encoding='utf-8')
path = RL/'REMOTE_ENVIRONMENT.md'
env_note = f'''## {observed} 提前提交的DT候选已撤回停止

PID2510247/1791907554.02的独立候选进程组已核对身份后停止，无活跃成员残留，正式GRPO3174448/1791793629.73保持。原因是尚未完成梯度加速增长的量级验算即提交，不能记为训练失败或修复验收。原source/launch/运行覆盖及日志保留，候选未写入ROOT正式三manifest。旧DTPO仍停止，小时检查PAUSED；本次仅CPU分析已有数据，无新训练或GPU调用。源码版本和停止/分析回执见RUNTIME_RECORD最新条目。以后使用设备前仍需核对实时占用。

'''
path.write_text(env_note+path.read_text(encoding='utf-8'),encoding='utf-8')
(HERE/'gradient-accounting-correction-v2.json').write_text(json.dumps(entry,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({'status':entry['status'],'rootcause_resolved':False,'candidate_stopped':True}))
