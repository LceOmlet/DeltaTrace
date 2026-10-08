"""Record completed passive input measurements; do not change a launcher."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RUN = HERE/'attention-input-textcraft'


def ref(path):
    path = Path(path)
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    analysis = json.loads((RUN/'analysis.json').read_bytes())
    preserved = json.loads((RUN/'preservation.json').read_bytes())
    assert preserved['all_SHA256_verified'] and not preserved['driver_alive']
    for item in preserved['files']:
        actual = ref(item['local'])
        assert actual['sha256'] == item['sha256'] and actual['bytes'] == item['bytes']
    previous = analysis['comparison_with_previous_gate_readout']
    assert len(previous) == 165
    assert all(p['DT_difference'] == p['native_difference'] == 0 for p in previous)
    launch = json.loads((RUN/'launch.json').read_bytes())
    earlier = json.loads((REPO/'experiments/rl/results_attention_gate_20261009.json').read_bytes())
    ranks = []
    for rank in (0, 1):
        data = json.loads((RUN/f'rank{rank}.json').read_bytes())
        phases = [json.loads(line) for line in (RUN/f'preserved/rank{rank}-phases.jsonl').read_text().splitlines()]
        assert data['phase'] == 'complete'
        assert data['operations'] == dict(DT=6,native_forward=30,backward=0,optimizer=0,
            scheduler=0,rollout=0,checkpoint_restore=0)
        ranks.append(dict(rank=rank,pid=data['pid'],birth=data['birth'],phase=data['phase'],
            elapsed_seconds=data['elapsed_seconds'],operations=data['operations'],
            actual_imports=data['owners'],finite_owner=data['finite_owner'],
            measured_DT_peak_allocated_bytes=max(b['DT_detail']['peak_allocated'] for b in data['batches']),
            sampled_phase_PSS_peak_bytes=max(p['process_pss_bytes'] for p in phases),
            additional_readout_seconds=sum(b['attention_branch_readout']['seconds'] for b in data['batches']),
            diagnostic_source_sha256=data['attention_branch_source_sha256']))
    strata = {}
    for key, group in analysis['repeatability_strata'].items():
        tail = group['robust_spurious_tail_from_original_run']
        strata[key] = dict(points=group['points'],old_robust_spurious_points=tail['points'],
            trajectories=tail['trajectories'],states=tail['states'],
            conditional_finite_sample_values={k:tail['conditional_finite_sample_values'][k]
                for k in ('input_projection_norm_RoPE_residual','finite_FA_core_residual')},
            state_equal_largest_absolute_core_term_frequency=tail['state_equal_largest_absolute_core_term_frequency'])
    receipt = dict(scope='Completed passive final FA input/QKV split and exact debug preservation; no repair',
        observed_unix=preserved['unix'],launch=ref(RUN/'launch.json'),launch_identity=launch,
        upstream_VERL_commit='20bd331',original_PPO_debug=earlier['original_PPO_debug'],
        original_PPO_NaN_repaired=False,
        frozen_native_input_preservation=earlier['frozen_native_input_preservation'],
        frozen_native_archive=earlier['frozen_native_archive'],native_input_files=64,
        diagnostic_preservation=ref(RUN/'preservation.json'),diagnostic_saved_files=len(preserved['files']),
        diagnostic_saved_bytes=sum(x['bytes'] for x in preserved['files']),
        analysis=ref(RUN/'analysis.json'),analyzer=ref(HERE/'analyze_attention_gate.py'),
        protocol=ref(HERE/'attention-input-protocol.json'),rank_checks=ranks,
        previous_gate_points=165,previous_gate_DT_and_native_exact_equal=True,
        original_historical_baseline_exact_equal_points=analysis['original_DT_exact_equal_points'],
        historical_rank1_drift='68 historical comparisons differ; original-run robust flags are not '
            'revalidated confidence bounds. New gate/input probes match each other exactly.',
        repeatability_strata=strata,
        decomposition='C_QKV=<dq,Delta Q>+<dk,Delta K>+<dv,Delta V>; '
            'input=mn1*Delta n1-mg*Delta gate-C_QKV; finite_FA_core=C_QKV-mc*Delta O.',
        interpretation='Core contraction includes finite FA propagation and native arithmetic. This '
            'localization is not an isolated kernel bug, causal share, method-quality or repair claim. '
            'No pooled body/tail moment; complete point/cell/exposure analysis is preserved.',
        actual_finite_FA_source=ref(HERE/'finite-FA-source.json'),
        token_contexts=dict(raw_display=ref(HERE/'saved-token-context.json'),
            joined_values=ref(HERE/'saved-token-context-with-values.json'),
            display_reader=ref(HERE/'read_saved_token_context.py'),
            trajectories=45,points=165,new_model_queries=0,new_DT=0,
            actual_readout_preparation_used=True,
            diagnostic_coordinate_error=dict(source=ref(HERE/'read_saved_token_context-failed-coordinate.py'),
                error=ref(HERE/'saved-token-context-failed-coordinate.stderr'),
                scope='First display-only reader confused original suffix mask slots with packed full-input slots. '
                    'It failed before producing output; corrected by the original suffix_positions mapping. '
                    'The production and native/DT diagnostic calls did not use this reader.')),
        official_FA_FLA_tolerance_changed=False,official_tolerance_executed=False,
        QVA_whitening_PPO_changed=False,production_modified=False,candidate_deployed=False,
        extreme_negative_credit_repaired=False,formal_training_restarted=False,
        physical_snapshot=preserved['physical'],host_snapshot=preserved['host'])
    output = REPO/'experiments/rl/results_attention_input_20261009.json'
    output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    snapshot_path=REPO/'experiments/rl/current_runtime.json'
    snapshot=json.loads(snapshot_path.read_bytes())
    snapshot.update(observed_unix=preserved['unix'],
        observed_utc=datetime.fromtimestamp(preserved['unix'],timezone.utc).isoformat())
    snapshot['latest_readonly_observation']=dict(textcraft='Formal training stopped',
        appworld='Formal training stopped',diagnostic='Passive FA input/QKV diagnostic complete and exited; '
            'no candidate or formal job launched',receipt=ref(output))
    snapshot['attention_input_completion_20261009']=dict(receipt=ref(output),
        PID3901349_status='exited',exact_local_saved_files=12,extreme_credit_repaired=False)
    snapshot_path.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    runtime=REPO/'experiments/rl/RUNTIME_RECORD.md'
    marker='## 2026-10-09 末层有限FA核心已由集合残差隔离；原现场完整保存'
    note='''## 2026-10-09 末层有限FA核心已由集合残差隔离；原现场完整保存

被动输入诊断PID3901349已结束，实际7文件绑定af05de9d；每rank6原DT/30原native、
零optimizer/rollout/恢复，完成231.77/231.46秒。与此前门控诊断全部165个DT d和
native单删逐值一致；与更早历史运行的rank1漂移仍单列，不按新整网阈值抹除。
12原文件3,773,592字节（源码/协议/原日志/完整逐位置ledger）已本机逐SHA保存。
PPO原44文件及64冻结native输入的无损本机副本保持；原NaN未称修复。

原dq/dk/dv的QKV收缩将core/input分成输入投影/QK归一化/RoPE与有限FA核心。
历史配对逐值不变的97位置中，9个旧稳健误报负尾跨5状态；全部9位置有限FA核心
绝对残差较输入部分更大，条件中位数-3.153664与-0.112901。历史漂移的68位置
另有9个旧标记位置，同样9/9核心较大；不混合成总体尾部矩或普遍因果结论。
实际加载.so SHA4f42c391及其原构建源码9ebcef18、build回执f2bbf6fc已只读核对，
不是据失败候选名称推测生产版本。仍需查有限FA背景/交互项，不能据此称核bug。

正式两组仍停止，Q/V/A、白化、PPO和原FA/FLA容差未改；无候选部署、无信用
裁剪/倍率纠偏。诊断峰值/导入路径/版本及分层见results_attention_input_20261009.json。
以下“已启动”条目保留为当时历史，不作为当前运行状态。


'''
    raw=runtime.read_bytes()
    if marker.encode() not in raw:
        line=raw.index(b'\n')+1
        runtime.write_bytes(raw[:line]+b'\n'+note.encode()+raw[line:])
    print(json.dumps(dict(receipt=ref(output),rank_checks=[{k:v for k,v in r.items()
        if k not in ('actual_imports','finite_owner')} for r in ranks])))


if __name__ == '__main__':
    main()
