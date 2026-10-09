"""Record the completed frozen PV diagnostic, without accepting a repair."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RUN = HERE/'attention-pv-textcraft-v2'


def ref(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    analysis = json.loads((RUN/'analysis.json').read_bytes())
    preserved = json.loads((RUN/'preservation.json').read_bytes())
    launch = json.loads((RUN/'launch.json').read_bytes())
    assert preserved['all_SHA256_verified'] and not preserved['driver_alive']
    for item in preserved['files']:
        actual = ref(item['local'])
        assert actual['sha256'] == item['sha256'] and actual['bytes'] == item['bytes']
    previous = analysis['comparison_with_previous_input_readout']
    assert len(previous) == 165
    assert all(p['DT_difference'] == p['native_difference'] == 0 for p in previous)
    points = analysis['points']
    assert all(all(p['final_FA_PV_ledger']['actual_owner'][
        'dense_original_view_identity_matches'].values()) for p in points)
    ranks = []
    for rank in (0,1):
        data = json.loads((RUN/f'rank{rank}.json').read_bytes())
        phases = [json.loads(s) for s in (RUN/f'preserved/results/rank{rank}-phases.jsonl').read_text().splitlines()]
        assert data['phase'] == 'complete'
        assert data['operations'] == dict(DT=6,native_forward=30,backward=0,optimizer=0,
            scheduler=0,rollout=0,checkpoint_restore=0)
        readouts = [b['attention_PV_readout'] for b in data['batches']]
        ranks.append(dict(rank=rank,pid=data['pid'],birth=data['birth'],operations=data['operations'],
            elapsed_seconds=data['elapsed_seconds'],actual_imports=data['owners'],finite_owner=data['finite_owner'],
            measured_DT_peak_allocated_bytes=max(b['DT_detail']['peak_allocated'] for b in data['batches']),
            sampled_phase_PSS_peak_bytes=max(p['process_pss_bytes'] for p in phases),
            sampled_phase_runtime_free_min_bytes=min(p['runtime_free_bytes'] for p in phases),
            added_public_FA_calls=sum(r['extra_public_FA_calls'] for r in readouts),
            additional_PV_readout_seconds=sum(r['seconds'] for r in readouts),
            artifact_write_and_hash_seconds=sum(r['artifact_write_and_hash_seconds'] for r in readouts),
            actual_FA_owner=readouts[0]['actual_owner']))
    keys = ('joint_QK_softmax_residual','factual_P_value_residual','PV_background_residual')
    strata = {}
    for name,group in analysis['repeatability_strata'].items():
        tail = group['robust_spurious_tail_from_original_run']
        strata[name] = dict(points=group['points'],historical_spurious_points=tail['points'],
            states=tail['states'],conditional_finite_sample_values={k:tail['conditional_finite_sample_values'][k] for k in keys},
            state_equal_largest_absolute_PV_term_frequency=tail['state_equal_largest_absolute_PV_term_frequency'],
            controls=tail['PV_controls'])
    operators = json.loads((RUN/'preserved/raw-operator-manifest.json').read_bytes())
    earlier = json.loads((REPO/'experiments/rl/results_attention_input_20261009.json').read_bytes())
    receipt = dict(scope='Completed passive final FA PV residual split, not an accepted numerical repair',
        observed_unix=preserved['unix'],launch=ref(RUN/'launch.json'),launch_identity=launch,
        source_commit=launch['base_commit'],upstream_VERL_commit='20bd331',
        original_PPO_debug=earlier['original_PPO_debug'],original_PPO_NaN_repaired=False,
        frozen_native_input_preservation=earlier['frozen_native_input_preservation'],
        frozen_native_archive=earlier['frozen_native_archive'],
        diagnostic_preservation=ref(RUN/'preservation.json'),diagnostic_saved_files=len(preserved['files']),
        diagnostic_saved_bytes=sum(x['bytes'] for x in preserved['files']),analysis=ref(RUN/'analysis.json'),
        analyzer=ref(HERE/'analyze_attention_gate.py'),protocol=ref(HERE/'attention-pv-protocol.json'),
        rank_checks=ranks,unique_points=165,previous_input_DT_and_native_exact_equal=True,
        original_historical_baseline_exact_equal_points=analysis['original_DT_exact_equal_points'],
        historical_rank1_drift='Original 68-point rank1 drift remains separate; new gate/input/PV values match exactly.',
        repeatability_strata=strata,
        observed_controls=dict(max_abs_decomposition_roundoff=max(abs(p['final_FA_PV_ledger']['decomposition_roundoff']) for p in points),
            max_original_FA_factual_replay_difference=max(p['final_FA_PV_ledger']['original_FA_factual_replay_maxabs'] for p in points),
            all_dense_operand_view_identities_match=True),
        original_operator_artifacts=dict(manifest=ref(RUN/'preserved/raw-operator-manifest.json'),
            files=len(operators['files']),bytes=operators['bytes'],copied_local=False,
            scope=operators['scope']),
        decomposition='QK=Cq+Ck-u*(O_FR-O_DR); value=Cv-u*(O_F-O_FD); '
            'PV=u*((O_FR-O_DR)-(O_FD-O_D)). Rounded native FA outputs are used.',
        interpretation='PV background is the largest absolute measured term in 8/9 historical spurious '
            'positions in each repeatability stratum; conditional distributions, states and exposure remain separate. '
            'The ledger includes native arithmetic and is not a causal share. No term is subtracted from credit. '
            'Saved operands enable operator-only follow-up without another full-model collection.',
        observer_failure=ref(REPO/'experiments/rl/results_attention_pv_observer_failure_20261009.json'),
        official_FA_FLA_tolerance_changed=False,official_tolerance_executed=False,
        QVA_whitening_PPO_changed=False,production_modified=False,candidate_deployed=False,
        extreme_negative_credit_repaired=False,formal_training_restarted=False,
        physical_snapshot=preserved['physical'],host_snapshot=preserved['host'])
    output = REPO/'experiments/rl/results_attention_pv_20261009.json'
    output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    snapshot_path = REPO/'experiments/rl/current_runtime.json'
    snapshot = json.loads(snapshot_path.read_bytes())
    snapshot.update(observed_unix=preserved['unix'],
        observed_utc=datetime.fromtimestamp(preserved['unix'],timezone.utc).isoformat())
    snapshot['latest_readonly_observation'] = dict(textcraft='Formal training stopped',
        appworld='Formal training stopped',diagnostic='Passive PV v2 completed and exited; no numerical candidate accepted',receipt=ref(output))
    snapshot['attention_PV_v2_completion_20261009'] = dict(receipt=ref(output),pid=launch['pid'],
        birth=launch['birth'],status='exited',extreme_credit_repaired=False)
    snapshot_path.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    note='''
## 2026-10-09 PV分解完成：误报负尾的主要FA余项是背景交互，尚未修复

被动诊断PID4148472/出生1791503396.84完成并退出，8文件绑定be4a5d0b；每rank
6原DT/30原native及60原public FA附加读出，零optimizer/rollout/恢复。两rank
497.07/496.66秒，其中PV读出7.40/6.43秒、文件写入和SHA247.00/258.87秒。
全部165位置与前一input诊断的DT d/native单删逐值一致；原历史rank1漂移仍单列。
原dense Q/K/V的view身份全数匹配，事实FA重放差0；这不是新官方容差测试。

历史97配对不变位置中的9个误报负尾（5状态），PV背景绝对余项在8/9位置最大，
状态等权频率.8；PV/QK-softmax/value绝对值条件中位数2.11926/.350870/.0106646。
另68历史漂移位置中的9个旧标记位置亦8/9为PV最大，独立保留，不混入总体尾部矩。
这些是同调用余项分解，包含原生算术，不是已隔离的因果份额、核bug或修复证明。
该PV余项比较单删除的P变化在联合参考V与单删V下的作用；后续定位同源交互，
不能直接扣余项、换输出倍率或按极值挑单删结果，既有失败条件行候选不重跑。

17项精确源码/协议/原日志/ledger共5,080,156字节已复制本机逐SHA核验；72项
原操作数41,814,827,240字节保留远端，写入时SHA和当前尺寸在manifest中，不称
已异机复制。后续算子定位复用这些张量，不必再采集一轮整网。原PPO44项与64
冻结输入的本机保存保持。FA/FLA原容差、Q/V/A、白化、PPO及正式入口未改。
两正式任务仍停止；原PPO NaN与极端负信用均未称修复。见results_attention_pv_20261009.json。

'''
    runtime = REPO/'experiments/rl/RUNTIME_RECORD.md'
    raw = runtime.read_bytes()
    if note.splitlines()[1].encode() not in raw:
        line = raw.index(b'\n')+1
        runtime.write_bytes(raw[:line]+note.encode()+raw[line:])
    print(json.dumps(dict(receipt=ref(output),ranks=[{k:r[k] for k in ('rank','elapsed_seconds',
        'measured_DT_peak_allocated_bytes','sampled_phase_PSS_peak_bytes','additional_PV_readout_seconds',
        'artifact_write_and_hash_seconds')} for r in ranks],remote_operator_files=len(operators['files']))))


if __name__ == '__main__':
    main()
