"""Persist the actual withdrawn status; no training or method changes."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import tomllib

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
R = HERE / 'receipts'
VERSION = 'textcraft-dt-weighted-reward-targets-20261015-v4'
ROOT = '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
AUDIT = ROOT + '/receipts/textcraft-weighted-reward-targets-20261015-v4'
PLOT = Path('F:/DeltaTraceBackups/plots') / VERSION / '20261016-004831'


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def identity(path):
    path = Path(path)
    return dict(path=str(path), SHA256=hashlib.sha256(path.read_bytes()).hexdigest())


stop = read(R/'DTPO-v4-method-withdrawn.json')
job = read(R/'DTPO-v4-job.json')
assert stop['status'] == job['status'] == 'stopped_method_equivalence_not_established'
assert not stop['survivors'] and stop['completed_iterations_from_original_logs'] == 0
assert stop['checkpoint_step_preserved'] is None and job['accepted_for_training'] is False
plot = read(PLOT/'plot-receipt.json')
snapshot = read(PLOT/'source-snapshot.json')
assert plot['completed_iterations'] == {'dt': 0, 'grpo': 191, 'ppo': 50}
assert snapshot['DT_display_label'] == 'DTPO v4（已撤回）'
auto = tomllib.loads(Path('C:/Users/Administrator/.codex/automations/textcraft/automation.toml').read_text(encoding='utf-8'))
sha = hashlib.sha256((HERE/'plot_current_algorithms_v4.py').read_bytes()).hexdigest()
assert auto['status'] == 'ACTIVE' and sha in auto['prompt'] and 'v4已撤回' in auto['prompt']
automation = dict(id=auto['id'], kind=auto['kind'], status=auto['status'], rrule=auto['rrule'],
    target_thread_id=auto['target_thread_id'], updated_at=auto['updated_at'],
    entry=identity(HERE/'plot_current_algorithms_v4.py'), local_TOML_readback_verified=True,
    current_DT_status='withdrawn; do not restart', scope='Original plots and existing logs only')
write(HERE/'hourly-automation-v4.json', automation)
checks = read(R/'interface-verification.json')
report = dict(version=VERSION, recorded_unix=time.time(), status=stop['status'],
    accepted_for_training=False, candidate_source_patch_commit='804d2f67c9a98a90705b91ca840587e2f6312b36',
    reason='The implementation replaced exponential Q/V advantage with g times normalized finite allocation. Linear seed/attribution equivalence and budget conservation do not establish the original Q/V or expected policy-gradient equivalence.',
    user_clarification='取消指数变换了？为什么要这样做？这不就破坏等价性质了吗',
    method_specification='experiments/rl/PLAN.md only; latest correction supersedes historical v4 deployment approval wording',
    actual_deployment=job, actual_withdrawal=stop, completed_formal_iterations=0,
    formal_optimizer_updates=0, formal_checkpoints=0, replacement_training_started=False,
    original_PPO_preserved=dict(pid=3946004,birth=1792041120.19,completed_at_plot=50,physical_gpus=[4,5]),
    stopped_v3=dict(completed=45,checkpoint=45,receipt=identity(R/'previous-DTPO-v3-stop.json')),
    stopped_GRPO=dict(completed=191,checkpoint=190),
    behavior_checks=checks,
    behavior_check_scope='12 unique passing interface/owner tests; genuine zero-settlement targets retained; default path and weighted seed behavior checked. These do not prove the removed-exponent estimator is equivalent.',
    native_receipts=[identity(R/n) for n in ('native-small-rank0.json','native-small-rank1.json','capacity-32768-rank0.json','capacity-32768-rank1.json','initial-state-and-existing-passive-records-v4.json')],
    capacity_scope='B4 per rank, dual B8, exact 32768 artificial capacity input; no optimizer or quality evaluation. Native finite about 383.86/383.96 seconds. Numeric differences reported without an invented tolerance.',
    target_filter_status='User requirement remains: only native reward-settlement action targets, including settlement value 0. Implemented and tested in retained candidate, not deployed as an accepted running method.',
    old_gradient_goal='paused; no new goal or unrelated diagnostics',
    hourly_plot=dict(output=str(PLOT),receipt=identity(PLOT/'plot-receipt.json'),automation=automation),
    remote_withdrawal_receipt=AUDIT+'/DTPO-v4-method-withdrawn.json',
    exact_receipts=[identity(p) for p in sorted(R.iterdir()) if p.is_file()])
path = WORKSPACE/'experiments/rl/results_textcraft_weighted_reward_targets_20261015_v4.json'
write(path,report)
summary = dict(collected_at=snapshot['collected_at'],output=str(PLOT),completed=plot['completed_iterations'],
    increments_since_previous_success={'dt':0,'grpo':0,'ppo':1},
    previous_successful_output=str(PLOT.parent/'20261016-003639'),DT_status='withdrawn at 0 complete iterations',
    common_window=0,training_changes_during_collection=0,model_calls_during_collection=0,automation=automation)
write(PLOT/'hourly-summary.json',summary)
runtime_path = WORKSPACE/'experiments/rl/current_runtime.json'
runtime = read(runtime_path)
current = dict(version=VERSION,status=stop['status'],accepted_for_training=False,pid=4022617,birth=1792081757.39,
    completed_formal_iterations=0,formal_optimizer_updates=0,formal_checkpoints=0,
    receipt=identity(path),withdrawal_receipt=identity(R/'DTPO-v4-method-withdrawn.json'),
    remote_withdrawal_receipt=AUDIT+'/DTPO-v4-method-withdrawn.json',
    original_PPO_identity=dict(pid=3946004,birth=1792041120.19),candidate_patch='804d2f67c9a98a90705b91ca840587e2f6312b36')
runtime={'20261016_DTPO_v4_withdrawn_exponential_QV_not_preserved':current,**runtime}
runtime['latest_textcraft_method_deployment']=current
runtime['latest_actual_training_plot_snapshot']=str(PLOT)
runtime['textcraft_hourly_plot_latest']={**summary,'summary':str(PLOT/'hourly-summary.json'),
    'receipt':str(PLOT/'plot-receipt.json'),'goal_status':'paused','scope':'Original three-algorithm plots and existing logs only'}
write(runtime_path,runtime)
stamp=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()
header=f'## {stamp} DTPO v4已撤回：取消指数不保留原Q/V等价，0轮时停止，PPO继续'
body=f'''用户质疑取消指数破坏等价性质。实际v4把原g*(1-exp(-d))替换为g*C，其中C是带权归一有限分配；原seed转置线性/单位信用守恒不能证明这项策略优势替代保留原Q/V或期望策略梯度等价。该部署决定撤回，不将行为/容量检查作为其理论验收。唯一PLAN已加最新纠正，以下较早批准v4实施文字作为历史。

v4 driver4022617/1792081757.39与其完整成员组已复用原保护停止owner撤下，无survivor；原collect/read_job_logs/parse_logs确认0完整迭代、0正式optimizer更新、无checkpoint。源码804d2f67、12项唯一行为检查及双rank精确32768/B4容量回执保留为未接受候选，不自动启训，也没有恢复旧v3。旧v3完成45轮/完整45保留；PPO3946004/1792041120.19及两worker和其他6/7受保护，继续同run，最新50轮；GRPO停止191/完整190历史保持。原奖励结算动作target（含0）的过滤要求仍有效，已实现测试的候选不等于正确优势组合已部署。

textcraft小时绘图仍ACTIVE，入口SHA{sha}按明确撤回回执标“DTPO v4（已撤回）”；0轮共同窗口为空、不拼旧曲线、保留原PPO/GRPO全历史。新图和原CSV/来源快照在{PLOT.as_posix()}。旧梯度goal仍paused，不新增诊断、重启或超参变化。完整事实与源码/配置/测试/停止回执见experiments/rl/results_textcraft_weighted_reward_targets_20261015_v4.json及远端{AUDIT}/DTPO-v4-method-withdrawn.json。
'''
for name in ['RUNTIME_RECORD.md','REMOTE_ENVIRONMENT.md']:
    p=WORKSPACE/'experiments/rl'/name
    old=p.read_text(encoding='utf-8');assert header not in old
    p.write_text(header+'\n\n'+body+'\n'+old,encoding='utf-8')
print(json.dumps(dict(status=stop['status'],completed=0,source_patch=report['candidate_source_patch_commit'],
                     report=str(path),plot=str(PLOT),automation='ACTIVE'),ensure_ascii=False))
