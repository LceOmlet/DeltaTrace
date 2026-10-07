"""Record completed owner comparisons; no launch, method or runtime mutation."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def binding(path):
    return dict(path=path.as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


rule_path = REPO / 'experiments/rl/results_existing_memory_rule_20261008.json'
curve_path = REPO / 'experiments/rl/results_existing_memory_author_curves_20261008.json'
rule, curve = [json.loads(path.read_bytes()) for path in (rule_path, curve_path)]
memory_observation = json.loads(sorted(HERE.glob('memory-observation-*.json'))[-1].read_bytes())
curve_observation = json.loads(sorted(HERE.glob('memory-curve-observation-*.json'))[-1].read_bytes())
for observed in (memory_observation, curve_observation):
    assert observed['completed'] and not observed['driver']['same_birth']
    assert observed['textcraft_same_birth'] and not any(observed['textcraft_release_present'])
assert rule['baseline_original_symmetric_vector_equals_prior_replay']
assert all(curve['controls'][key] for key in
           ('same_original_source', 'same_source_positions',
            'all_views_share_factual_score', 'all_views_share_all_EOS_score'))
assert rule['source_sha256'] == curve['source_sha256']
assert not rule['credit_repaired'] and not curve['credit_repaired']
case = next(point for point in rule['sampled_points']
            if point['row'] == 3 and point['mode'] == 'most_negative')
current = dict(
    state='completed_existing_forward_memory_rule_not_accepted_no_credit_repair',
    source_sha256=rule['source_sha256'],
    DT_launch=json.loads((HERE/'memory-launch.json').read_bytes()),
    curve_launch=json.loads((HERE/'memory-curve-launch.json').read_bytes()),
    latest_observation_unix=curve_observation['unix'],
    both_diagnostic_drivers_exited=True,
    diagnostic_workers_complete=True,
    complete_DT_comparison=binding(rule_path),
    original_author_curves=binding(curve_path),
    original_author_function_sha256='583f4b7d0426407eb9a517f173365762860a1f4382f472dffb5c07de7d3e94a1',
    original_author_k=20,
    actual_native_B4_sha256=rule['original_native_sha256'],
    original_symmetric_full_vector_matches_previous_replay=True,
    actual_target_endpoint_arrays_equal=rule['exact_target_endpoint_arrays_equal'],
    selected_newline=case,
    original_native_target_decomposition=rule['newline_previous_native_target_decomposition'],
    native_fla_V_branch_order_diagnosis=rule['native_fla_V_branch_order_diagnosis'],
    original_author_metrics={name:{view:entry['author_return']
                                  for view,entry in profile['views'].items()}
                             for name,profile in curve['profiles'].items()},
    selected_B4_raw_advantage_moments=rule['selected_B4_raw_advantage_moments'],
    interpretation='The existing forward memory callback worsens the selected complete-vector outlier and this trajectory\'s original RISE/MAS. The local forward-order improvement is insufficient for a production switch. The reverse endpoint V component is the largest measured local discrepancy; this is a finite decomposition component, not its independent causal effect or a whole-token advantage.',
    selected_points_not_population_accuracy=True,
    A_squared_not_parameter_gradient_share=True,
    memory_fix=dict(commit='799224868e0a9c0f8031b6012bb71505ab801a35',
                    state='actual_failed_B4_and_two_32768_DT_capacity_verified_not_formally_deployed',
                    receipt=binding(REPO/'experiments/rl/results_memory_capacity_20261008.json')),
    formal=dict(TextCraft_pid=2833207, TextCraft_birth=1791370325.16,
                same_birth=True, update_release_present=[False, False],
                AppWorld_state='terminal_not_restarted',
                restore_checkpoint=False, production_profile_changed=False),
    unchanged=dict(lora_rank=8, lora_alpha=16,
                   actor_microbatch_per_gpu=4, DT_minibatch_per_gpu=4,
                   context_limit=32768, QVA_formula=True, PPO=True,
                   whitening=True, source_and_target_IDs=True),
    terminal_resources=dict(physical_mx_smi_by_GPU=curve_observation['physical_memory_mib']
                            if 'physical_memory_mib' in curve_observation else curve_observation['physical_mx_smi'],
                            cgroup_memory_usage_bytes=int(curve_observation['memory.usage_in_bytes'])),
    physical_peaks=dict(DT=rule['resources'], original_author_curves=curve['measurements']['physical_sampled_peak_mib']),
    numerical_tolerance_changed=False, correction_or_credit_clipping_added=False,
    official_accuracy_pass_claim=False, credit_repaired=False,
    formal_restart=False, TextCraft_update_released=False,
)
runtime = REPO/'experiments/rl/current_runtime.json'
raw = runtime.read_bytes()
marker = b'  "latest_existing_memory_rule_comparison":'
start = raw.index(marker)
assert list(json.loads(raw))[-1] == 'latest_existing_memory_rule_comparison'
tail = '\r\n'.join(json.dumps({'latest_existing_memory_rule_comparison':current},
                              ensure_ascii=False, indent=2).splitlines()[1:-1]).encode('utf8')
runtime.write_bytes(raw[:start] + tail + b'\r\n}\r\n')
json.loads(runtime.read_bytes())

ledger = REPO/'experiments/rl/RUNTIME_RECORD.md'
heading = '## 2026-10-08 完整memory对照与原作者曲线结束：单方向规则未被接受'
note = '''
完整DT诊断99ad9596/PID2175801/birth1791404173.84、作者曲线8987deac/
PID2233742/birth1791404715.28均已完成退出，物理4/5回到各859MiB。
同一真实B4、原对称向量逐值复现；仅改为原已有forward memory callback时，
newline的d从-4.415554变-4.698849，原始A从-61.295715变-81.615494。
先前原生单删除d=+23.089031，对应A约+0.75；主要来自下一code fence的概率
0.998526降至7.3486e-10。此大负值不符合当前定义的单删除文本反事实方向。
原作者函数SHA583f4b7d、k20和所有原数组保留；原对称/forward的signed RISE
为0.393269/0.403295，positive-view MAS为0.631998/0.644110，均未改善。
这是选定真实轨迹，不能当总体质量结论；两rank各42个原生forward，无新DT/
backward/optimizer/rollout/恢复。完整DT两模式各一次/rank的冷暖时差不称提速比。

已有原生FLA张量的CPU分项汇总进一步定位：真实single端点上原finite的V分支
为+12.016079；joint正向/反向系数乘同一single差为+14.112987/-3.095070。
反向V项差-15.111150，其余项合计补偿约+2.686，不能把分项称独立V干预或
全token优势。全native single输出效应+15.827504，原single finite为+15.825014；
未发明非零finite官方容差。已有单方向局部改善没有传递到完整向量与作者曲线，
故不部署，不组合扫描，不裁剪或加倍率，不修改Q/V/PPO与原白化。

App OOM的79922486存储生命周期补丁与信用质量分开：实际失败B4和包含原
async-vLLM生命周期的两次精确32768 DT容量已通过，峰值60.803/62.627GiB；
此为容量回执，不是32k任务/PPO更新/信用准确性验收，也尚未正式部署。
Text原PID2833207/birth1791370325.16继续hold，release不存在；App正式未重启。
所有实际路径/SHA、启动commit/PID创建时间、配置与回执绑定见current_runtime
末字段、results_existing_memory_rule_20261008.json及
results_existing_memory_author_curves_20261008.json。信用尚未修复。

'''
content = ledger.read_bytes()
assert heading.encode('utf8') not in content
first_newline = content.index(b'\n')+1
ledger.write_bytes(content[:first_newline] + ('\n'+heading+'\n'+note).encode('utf8') + content[first_newline:])
print(json.dumps(dict(runtime=binding(runtime), ledger=binding(ledger),
                      complete_DT=current['complete_DT_comparison'],
                      curves=current['original_author_curves']),ensure_ascii=False))
