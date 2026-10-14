"""Reconcile saved norm-reporting evidence without training computation."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RESULTS = REPO / 'experiments/rl'
ROOT = '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
OUT = ROOT + '/receipts/textcraft-gradient-rootcause-20261013-v1'
now = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def saved(name):
    value = read(HERE / name)
    return value.get('report', value)

def witness(name):
    path = HERE / name
    return dict(path=str(path), SHA256=hashlib.sha256(path.read_bytes()).hexdigest())

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

native = saved('record_native_rank_metadata_completion_v76-dt.json')
cpu = saved('verify_native_dtensor_norm_cpu_v75-dt.json')
replication = saved('compare_completed_original_feedback_v73-dt.json')
probes = saved('saved-native-probe-reducer-accounting-v74.json')
checkpoint = read(HERE / 'v3-first-native-checkpoint5-state.json')
reference = read(HERE / 'official-global-norm-materialization-reference-20261015.json')
plot_dir = Path('F:/DeltaTraceBackups/plots/textcraft-dt-grpo-multiplier-variance-only-20261014-v3/20261015-040157')
snapshot = read(plot_dir / 'source-snapshot.json')
plots = read(plot_dir / 'plot-receipt.json')
backup = read(Path('F:/DeltaTraceBackups/textcraft-20261011/latest-backup.json'))
coordinate_cases = read(RESULTS / 'results_textcraft_gradient_coordinates_20261015_v1.json')['cases']
fixed_norms = [math.hypot(*coordinate_cases[str(step)]['coordinate_norms'].values()) for step in (5, 40, 80)]
assert native['original_methods_restored_both_ranks']
assert cpu['status'] == 'native_Torch_CPU_norm_return_and_clipping_contract_verified'
assert checkpoint['latest_completed_checkpoint_marker'] == 5

scope = dict(
    field='actor/grad_norm', native_actor_SHA256='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c',
    meaning='Original returned-rank local NormPartial.item(), before clipping; native DataProto.concat retains the first returned metadata dictionary.',
    first_returned_rank_not_assumed_global_rank0_for_all_historical_runs=True,
    materialized_global_norm_missing_from_old_console=True,
    old_values_not_replaced_by_sqrt2_estimates=True,
    clipping_error_not_inferred_from_reporting_error=True,
    official_actor_core_training_math_unchanged=True,
)

probe_rows = []
for row in probes['cases']:
    probe_rows.append(dict(diagnostic=row['diagnostic'], case=row['case'],
        local_native_clip_returns=[r['native_clip']['native_clip_return'] for r in row['ranks']],
        global_preclip_norm_from_saved_gradient_shards=row['full_preclip_norm_from_native_shards'],
        sources=[dict(path=r['source'], SHA256=r['SHA256']) for r in row['ranks']]))

report = dict(
    recorded_at=now, status='native_partial_norm_reporting_confirmed_real_historical_growth_remains',
    goal_status='active', hourly_automation='PAUSED', rootcause_repair_verified=False,
    native_metric_scope=scope,
    actual_CUDA_training_observation=dict(
        completed_iteration=6, version='DTPO v3', driver_pid=native['driver_pid'], driver_birth=native['driver_birth'],
        native_local_norms_by_rank=native['native_local_norms_by_rank'],
        native_local_norm_mean_by_rank=native['native_local_norm_mean_by_rank'],
        global_norm_per_original_optimizer_step=native['materialized_global_norm_per_optimizer_step'],
        global_norm_mean=native['materialized_global_norm_mean'],
        calculation=native['calculation'], original_methods_restored_both_ranks=True,
        additional_model_backward_optimizer_scheduler_calls=0, additional_tensor_collectives_arithmetic=0,
        receipt=OUT + '/live-native-rank-metadata-once-v76/completed-accounting.json',
        original_capture_bindings=[dict(path=x['path'], SHA256=x['SHA256']) for x in native['captures']]),
    native_CPU_contract=dict(
        status=cpu['status'], torch_version=cpu['ranks'][0]['torch_version'],
        local_item=[r['original_item_return'] for r in cpu['ranks']],
        global_norm=cpu['ranks'][0]['materialized_global_norm'],
        original_clipped_gradient=cpu['ranks'][0]['original_clipped_gradient'],
        official_unsharded_reference_gradient=cpu['ranks'][0]['official_unsharded_reference_gradient'],
        original_torch_testing_assert_close_passed=all(r['official_assert_close_passed'] for r in cpu['ranks']),
        CUDA_transfer_tested=False, GPUs_used=0, model_backward_optimizer_calls=0,
        source=cpu['ranks'][0]['source'], receipt=OUT + '/native-dtensor-norm-contract-cpu-v75/result.json'),
    official_reference=reference,
    saved_GPU_probe_norm_accounting=probe_rows,
    historical_fixed_seed_global_VJP=dict(
        scope='Same factual late input and saved BF16 block31 adjoint, original all-rank parameter-gradient shards. Directional parameter VJP, not whole PPO gradient or operator norm.',
        completed_checkpoint=[5, 40, 80],
        global_norm=fixed_norms,
        rank0_norm_squared_share_rounded=[0.4666500, 0.4703735, 0.4732188],
        checkpoint80_over_5=fixed_norms[-1] / fixed_norms[0],
        conclusion='The actual global directional sensitivity still increases about18.28x while rank energy share stays nearly constant; local metric scope does not explain this growth.',
        source_report='experiments/rl/results_textcraft_gradient_coordinates_20261015_v1.json'),
    repeated_original_update_audit=dict(
        configuration_differences=replication['effective_configuration_differences'],
        before_parameter_elements_identical=replication['before_parameter_elements_identical'],
        parameter_displacement=replication['parameter_displacement']['total'],
        first_eight_returned_rank_metrics_identical={k:v['first_eight_identical'] for k,v in replication['native_first_update_metrics'].items()},
        first_eight_all_rank_forward_identity_established=False,
        original_fixed_seed_global_VJP_changes_percent=[7.2281, 1.0772],
        decision='The original conditional effect magnitude varies;7.23% is not a stable precise causal rate. Actual net parameter displacement also differs, so a scalar logging correction alone cannot explain this variation.',
        receipt=OUT + '/completed-original-feedback-comparison-v73.json', rootcause_of_repeat_variation_resolved=False),
    current_DTPO_v3=dict(
        canonical_method_patch='f726ee72', actual_submission='e05e23cb1bc1cccade732ad192ce15b7e9dc6a1e',
        trainer_SHA256='b4b0187ef747d1ffb91dbfb8d1b8f914f93d01d90ff58d24e386f2cb1b7eb943',
        official_PPO_reduction='L=(1/16)*sum_j(S_j/N_j)',
        official_GRPO_multiplier=True, variance_only_no_mean_subtraction=True,
        global_denominator_candidate_deployed=False, first_native_checkpoint=checkpoint),
    latest_plots=dict(directory=str(plot_dir), receipt=plots, source_snapshot_SHA256=hashlib.sha256((plot_dir/'source-snapshot.json').read_bytes()).hexdigest()),
    backup=dict(status=backup['status'], started=backup['started'],
        completed_checkpoints=backup.get('completed_checkpoints'), files=backup.get('files'), bytes=backup.get('bytes'),
        local_full_sha256_verified=backup.get('local_full_sha256_verified', False),
        shared_backup=backup.get('shared_backup'), latest_receipt='F:/DeltaTraceBackups/textcraft-20261011/latest-backup.json',
        native_schema_compatibility_patch='3a4db894', training_changes=0,
        shared_mount_is_same_NFS_fault_domain=True),
    local_evidence=[witness(n) for n in ('record_native_rank_metadata_completion_v76-dt.json',
        'verify_native_dtensor_norm_cpu_v75-dt.json', 'compare_completed_original_feedback_v73-dt.json',
        'saved-native-probe-reducer-accounting-v74.json', 'v3-first-native-checkpoint5-state.json')],
    limits=[
        'Reporting materialization has not been deployed; official actor/core SHA remain unchanged.',
        'A CPU/Gloo clipping contract does not establish every live CUDA clipping detail.',
        'An early stable v3 norm, conditional decrease or first full checkpoint is not acceptance of the historical gradient-growth repair.',
        'The15 matched-history28.27x target-gradient contrast and the fixed-input18.28x weight-state sensitivity contrast are distinct controlled comparisons.',
    ],
)
report_path = RESULTS / 'results_textcraft_native_norm_reporting_20261015_v1.json'
write(report_path, report)

v3_path = RESULTS / 'results_textcraft_DTPO_variance_v3_20261015.json'
v3 = read(v3_path)
v3.update(recorded_at=now, native_metric_scope=scope,
          first_completed_native_checkpoint=checkpoint,
          actual_both_rank_iteration6_metadata=report['actual_CUDA_training_observation'],
          latest_plots=report['latest_plots'], backup=report['backup'])
v3['latest_progress'] = {j['method']:max((x['step'] for x in j['metrics']), default=0) for j in snapshot['jobs']}
v3['latest_native_metric'] = {j['method']:j['metrics'][-1] if j['metrics'] else None for j in snapshot['jobs']}
v3['latest_metric_observed_unix'] = snapshot['collected_at']
write(v3_path, v3)

feedback_path = RESULTS / 'results_textcraft_full_v3_feedback_regression_20261015_v1.json'
feedback = read(feedback_path)
feedback['native_grad_norm_scope_correction'] = scope
feedback['repeated_original_update_reliability_audit'] = report['repeated_original_update_audit']
feedback['native_grad_norm_mean_ratio_is_global_norm_ratio'] = False
feedback['global_fixed_seed_parameter_VJP_values_remain_valid'] = True
write(feedback_path, feedback)

upper_path = RESULTS / 'results_textcraft_upper_propagation_20261015_v1.json'
upper = read(upper_path)
upper['native_temporal_gradient_norm_scope'] = scope
write(upper_path, upper)

record = dict(recorded_at=now, status=report['status'], public_report=str(report_path),
    goal='active', hourly_checks='PAUSED', rootcause_repair_verified=False,
    official_actor_core_unchanged=True, current_DTPO_v3=report['current_DTPO_v3'],
    actual_CUDA_training_observation=report['actual_CUDA_training_observation'],
    latest_progress=v3['latest_progress'], plots=str(plot_dir), backup=report['backup'])
runtime_path = RESULTS / 'current_runtime.json'
runtime = read(runtime_path)
runtime = {'20261015_native_partial_norm_reporting_and_v3_checkpoint5':record, **runtime}
write(runtime_path, runtime)
paragraph = f'''## {now} 原生日志分片范数已确认；v3第5次完整检查点已核验，根因未验收

v3自然第6次更新的原返回对象在两个rank均为DTensor NormPartial；原actor直接.item()，原DataProto.concat只保留首返回rank元数据。四笔更新分别合并两个已保存rank范数后，全局均值0.03420001，原rank0均值0.02273131；两处原方法已自动恢复，额外模型/反向/optimizer/scheduler/张量运算及collective均0。原actor3a65e173/corefc2f992b保持，未部署日志修补。CPU原Torch两rank/Gloo合同通过默认assert_close：local3/4、global5，原裁剪与官方非分片参考一致；这不代表已测试CUDA转移或裁剪一定有错。

统计范围错误不能解释原异常：已有全部梯度分片的固定输入/BF16 seed参数VJP在checkpoint5/40/80为0.06040356/0.08634082/1.10390448，5→80真实全局增长18.28倍；rank0能量份额约0.467→0.473。v73仅CPU核对两次原更新：配置与21,639,168个before参数元素相同，但净位移差/旧位移0.2454、cos0.9694；原固定探针增幅7.23%/1.08%不同，不能作为精确稳定因果倍率，首返回rank前8个前向指标相同不能当作全部rank一致。根因goal active，未验收修复。完整来源/SHA/官方参考和边界见experiments/rl/results_textcraft_native_norm_reporting_20261015_v1.json。

DTPO v3仍3140624/1791993570.92、physical0/1，方法f726ee72/提交e05e23cb/trainer b4b0187e保持原官方1/16、官方GRPO组乘子和masked_var仅方差缩放不减均值；32k/LoRA8/16/B4/global64/330/save5/keep2不变。首原生checkpoint5两rank均496 optimizer states/step20、scheduler5，cpu/numpy/random/cuda训练RNG和rollout RNG、原data.pt均实际读取，model文件各8,997,781,971字节、optimizer各87,091,812字节；完整文件SHA仍待原备份。04:01:57原日志DTPO6/330、GRPO162/330、PPO22/330；PPO4128395/1792002965.29已完整恢复20并继续，GRPO3174448保持，小时检查PAUSED。三算法图/CSV/来源快照在{plot_dir}，梯度轴已注明首返回rank分片范数，未补估缺失全局值。

一次异机备份因旧schema entry/argv与覆盖引用类型在复制前失败，失败回执保留；仅原backup_current_training.py补读source/launch和路径引用，patch3a4db894，不改训练或rclone传输实现。04:16:25重交PID26128/出生1792008985.968，F盘启动剩余83,510,960,128字节，当前状态{backup['status']}，未声称双副本SHA校验完成；F盘是异机，同NFS共享副本不是独立故障域。

'''
for name in ('RUNTIME_RECORD.md', 'REMOTE_ENVIRONMENT.md'):
    path = RESULTS / name
    old = path.read_text(encoding='utf-8-sig')
    path.write_text(paragraph + old, encoding='utf-8')
print(json.dumps(dict(report=str(report_path), progress=v3['latest_progress'], backup_status=backup['status'], global_norm_mean=native['materialized_global_norm_mean']), ensure_ascii=False))
