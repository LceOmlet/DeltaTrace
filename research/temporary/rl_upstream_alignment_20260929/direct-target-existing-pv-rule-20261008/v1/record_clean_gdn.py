"""Record the completed, rejected numerical-profile diagnostic, not a deployment."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def binding(path):
    return dict(path=path.as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


path = REPO/'experiments/rl/results_existing_clean_GDN_20261008.json'
result = json.loads(path.read_bytes())
observed = json.loads(sorted(HERE.glob('clean-gdn-observation-*.json'))[-1].read_bytes())
assert observed['completed'] and observed['textcraft_same_birth']
assert not any(observed['textcraft_release_present'])
assert all(row['phase'] == 'complete' for row in observed['ranks'])
assert result['baseline_original_symmetric_vector_equals_prior_replay']
point = next(x for x in result['sampled_points'] if x['row'] == 3 and x['mode'] == 'most_negative')
assert point['existing_clean_gdn']['opposite_sign_to_previous_single_delete']
current = dict(
    state='complete_preserved_clean_GDN_comparison_not_accepted_no_credit_repair',
    receipt=binding(path), launch=json.loads((HERE/'clean-gdn-launch.json').read_bytes()),
    terminal_observation=binding(sorted(HERE.glob('clean-gdn-observation-*.json'))[-1]),
    workers_complete=True, physical_devices_returned_to_idle=True,
    baseline_full_vector_equals_previous=True,
    native_target_endpoint_arrays_equal=result['exact_target_endpoint_arrays_equal'],
    preserved_clean_owner=result['preserved_clean_owner'],
    selected_newline=point,
    selected_points_sign_disagreements=result['biased_sample_sign_disagreements'],
    selected_points_not_population_rate=True,
    physical_peak_and_PSS=result['resources'],
    original_author_curves_for_this_candidate='not_run: the candidate failed the already demonstrated extreme-token repair; no overall RISE/MAS quality claim is made',
    existing_baseline_original_author_curve_receipt=binding(REPO/'experiments/rl/results_existing_memory_author_curves_20261008.json'),
    disposition='Do not switch or combine profiles to repair a measured local sign. The owner-preserved clean profile worsens this same complete-target outlier; the original current profile is preserved. No clipping, multiplier, target modification or Q/V/A/PPO change.',
    formal=dict(TextCraft_pid=2833207, TextCraft_birth=1791370325.16,
                update_release_present=[False, False], AppWorld_state='terminal_not_restarted',
                checkpoint_restore=False, production_profile_changed=False),
    credit_repaired=False, tolerance_changed=False,
    memory_fix_state='79922486 verified on failed B4 and two exact32768 DT calls; current diagnostic also completed, no formal deployment',
)
runtime = REPO/'experiments/rl/current_runtime.json'
raw = runtime.read_bytes()
key = 'latest_preserved_clean_GDN_comparison'
assert key not in json.loads(raw) and raw.endswith(b'}\r\n')
tail = '\r\n'.join(json.dumps({key: current},ensure_ascii=False,indent=2).splitlines()[1:-1]).encode('utf8')
runtime.write_bytes(raw[:-3].rstrip(b'\r\n')+b',\r\n'+tail+b'\r\n}\r\n')
json.loads(runtime.read_bytes())
ledger = REPO/'experiments/rl/RUNTIME_RECORD.md'
heading = '## 2026-10-08 完整clean-v1 GDN数值规则对照完成：未修复极端信用，不部署'
note = '''
b5b7b391/PID2575252/birth1791407996.54两rank已完成，物理4/5回到各859MiB。
同一实际B4、当前存储/内核/分块/target/QVA/PPO不变，仅选择作者保留clean-v1的
空norm/gate与memory map。完整基线向量逐值复现，所有原target端点数组逐值相同。
换行符2883的原对称d=-4.415554/A=-61.295715，clean d=-8.588811/A=-4027.667480；
先前原生单删除d=+23.089031/A=+0.75，实际奖励0.75。因此未修复，候选拒绝。
此前偏置12点中的反号4变5，不是总体错误率。两rank全向量与QVA逐值相同。
完整DT冷基线105.63/105.52秒、暖clean70.21/70.25秒，不称加速比。
物理峰值均54196MiB，PSS峰值8.151/8.375GB，原79922486缓存释放仍生效。

此候选未跑新增作者曲线：它已经未能修复本次明确异常，不继续花费曲线计算；
不因此声称总体归因质量变差。原对称及memory-only的作者曲线回执保留，不被单token
对照替换。辅助clean曲线入口只是复用原owner的诊断选项，未启动/未部署。
没有裁剪、倍率、改target、改Q/V/PPO、修改容差或恢复检查点。Text仍hold，
App正式仍terminal。当前信用问题未修复，显存生命周期补丁验收单独保留。
所有源码SHA、实际helper、原任务配置及回执绑定见current_runtime末字段。
'''
raw = ledger.read_bytes()
assert heading.encode('utf8') not in raw
end = raw.index(b'\n')+1
ledger.write_bytes(raw[:end]+('\n'+heading+'\n\n'+note.lstrip()+'\n\n').encode('utf8')+raw[end:])
print(json.dumps(dict(receipt=binding(path), state=current['state'], selected_newline=point),ensure_ascii=False))
