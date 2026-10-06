"""Bind completed, separately scoped efficiency and official-operator evidence."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROW = HERE / 'individual-prefix-owner-candidate-v1'
REP = ROW / 'native-representation-candidate'
COMBINED = ROW / 'combined-storage-row-v1'


def ref(path):
    return dict(path=str(path.relative_to(REPO)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    completed_path = COMBINED / 'combined-b8-capacity-completed-summary-1791324705.json'
    completed = json.loads(completed_path.read_bytes())
    rows = completed['probe']['ranks']
    gdn_path = REP / 'gdn-normalizer-v2/result.json'
    gdn = json.loads(gdn_path.read_bytes())
    assert gdn['status'] == 'passed' and not gdn['tolerance_changed']
    fa_path = REP / 'offline-checks-v1/independent-fa-results/execution.json'
    fa = json.loads(fa_path.read_bytes())
    assert fa['status'] == 'passed' and len(fa['checks']) == 4
    phases = json.loads((REP / 'real-b8-v3-phase-1791323792.json').read_bytes())
    speed = {}
    for rank in ('0', '1'):
        original = phases['ranks'][rank]['values']['reports']['shared_warm']['total_wall_seconds']
        actual = rows[rank]['stages']['actual_b8']['reports']['shared_individual_rows_warm']
        capacity = rows[rank]['stages']['exact32768_b8']['reports']['shared_individual_rows_warm']
        assert rows[rank]['context_lengths'] == [32768] * 4
        speed[rank] = dict(original_actual_B4_hot_seconds=original,
            combined_actual_B4_hot_seconds=actual['total_wall_seconds'],
            same_input_observed_ratio=original / actual['total_wall_seconds'],
            combined_exact32768_B4_hot_seconds=capacity['total_wall_seconds'],
            actual_pss_bytes=actual['pss_bytes'], capacity_pss_bytes=capacity['pss_bytes'],
            capacity_torch_peak_allocated_bytes=capacity['peak_torch_allocated_bytes'])
    result = dict(status='isolated_B8_and_exact32768_DT_completed_production_wiring_prepared',
        no_checkpoint_restore=True, no_optimizer_step=True,
        original_request_and_official_config_preserved=True,
        resources=dict(lora_rank=8, lora_alpha=16, actor_microbatch_per_card=4,
                       DT_minibatch_per_card=4, total_submitted_batch=8, max_length=32768),
        measurements=speed,
        evidence=dict(completed=ref(completed_path),
            combined_CPU=ref(COMBINED / 'cpu-contracts.json'),
            official_native_varlen_and_finite=ref(fa_path),
            official_GDN=ref(gdn_path),
            failed_old_reference=ref(REP / 'offline-checks-v1/rank0-failure-inspection.stdout.txt'),
            canonical_reference_interface_fix=ref(ROW / 'gdn-checker-canonical-fix-v1/source-fix.json'),
            QVA_comparison=ref(REP / 'saved-qva-cpu-review/compact-qva-review.json'),
            raw_A_direction=ref(REP / 'saved-qva-cpu-review/real-b8-v3-rawA-direction.json'),
            production_prepared=ref(ROW / 'production-wiring-v1/prepared.json'),
            production_actual_CPU_imports=ref(ROW / 'production-wiring-v1/actual-imports-cpu.json')),
        limits=['Speed ratio covers this saved B8 only, not a full iteration or training-throughput ratio.',
                'The exact32768 call is DT capacity, not a new optimizer-update or vLLM-colocation test.',
                'FA original assertions cover native varlen forward and coincident finite derivative checks; nonzero row/scalar outputs are bitwise equal on the actual saved operands.',
                'GDN uses production official l2norm_fwd at the reference boundary, then unchanged official recurrence/o/ht assertions. The previous FP16 normalization failure remains recorded.',
                'Cross-bank Q/V/A differences are measured and retained, not assigned a new whole-network tolerance.',
                'Whole-batch official whitening remains enabled in the trainer; this evidence does not prove learning quality recovered.',
                'Current formal-v3 has no completed update/gen timing pair; no historical ratio is substituted.'])
    receipt = ROW / 'verification-index.json'
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    main_path = REPO / 'experiments/rl/results_appworld_efficiency_20261007.json'
    main_result = json.loads(main_path.read_bytes())
    main_result['individual_row_candidate_progress'] = result
    main_result['individual_row_verification_index'] = ref(receipt)
    main_result['fresh_v3_latest_phase'] = ref(COMBINED / 'formal-v3-phase-1791324705.json')
    main_result['status'] = 'Fresh formal-v3 source97e3 remains active; isolated row+storage B8/exact32768 and scoped original FA/FLA checks completed; production wiring prepared, not yet deployed.'
    main_path.write_text(json.dumps(main_result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    ledger_path = REPO / 'experiments/rl/RUNTIME_RECORD.md'
    ledger = ledger_path.read_text(encoding='utf8')
    title = '## 2026-10-07 06:20 逐行缓存与存储组合完成有界验收，生产接线仅prepared'
    if title not in ledger:
        note = '''
## 2026-10-07 06:20 逐行缓存与存储组合完成有界验收，生产接线仅prepared

正式v3仍1953903/birth1791321793.72、GPU4/5、source97e3cb75，没有导入旧训练
检查点。旧v2已实际执行整批官方masked_whiten；v3继续相同trainer d35ddd26，
06:11仍DT第二组44/46，无本次actor入口/完整gen-update timer。此前2495秒actor
属于旧padding版本，不能充作当前正常性能证明或本次更新时间。

隔离real-b8-v3已完成，非仍在运行：result f637f0e5，runner5f14/answerd473、
未压缩artifact50af/lease54ad，wrapper3e1d/library4f42。原same-input热调用
9.77/9.74秒→逐行3.93/3.94秒。实际FA varlen原断言通过，原7ff保存操作数的
每行7项coincident断言通过；实际非零B4五项输出与原scalar逐行bitwise相同。
GDN旧checker对FP16全零padding直接F.normalize产生NaN，实际native张量全finite。
新d107只接原FLA l2norm_fwd后执行不变reference/o-ht断言，两rank通过；本机
canonical checker c2853a15已同样修接口。旧5c192源/失败保留，没有nan_to_num或
放宽tol。上述均为对应算子范围，不是新整网容差。

组合artifact37a860/leaseb947复用已验证4a/6d存储owner，只存消费的边界行。
实际176请求CPU合同a3b469：原literal IDs/边界/last-write保持、HF Cache 32项
字节检查通过、CUDA未初始化。隔离combined-capacity-b8-v1原verify22bc、
diag9b1d、prepared0189，PID2235930/birth1791324452.14已自行完成并释放GPU2/3。
总体完成result87deb9be；原模型初始化顺序真实B8冷暖→exact32768冷暖，无CP/
optimizer step/profiler。真实热调用3.962秒，32768热调用8.30秒/每卡B4。
32768 torch allocated peak25351057920B是该DT容量范围，不是新PPO更新或vLLM
共存峰值验收。真实行PSS13289958400/13900460032B，32k PSS约14.4GB。

跨bank/layout的A最大差.02030/.03152、1012有效action的方向cosine.989422、
L2比.98455均保留；native端点已有.16量级变化，不能归咎单FA finite或把缓存
dtype转换当作全部原因。组合对未压缩row A/V最大差.00194/.00140，同bank
重复也有非零差；不加整网通过阈值。原Q/V/A/奖励/观察mask/PPO公式未改。

production-wiring-v1 prepared6c65，producer3e0c只给原prefix factory透传已有
individual_prefixes/boundary_row_storage参数，默认不进入；canonical路径直接
指向上述owner，不使用诊断class/sys.modules替换。environment4ff仅改库/sha与
两开关；原factory/profile、VERL/LOOP/trainer不变。禁CUDA真实14模块import
f1f481通过，未构造模型；此时仍仅prepared，不能当部署或正式DT已执行。
来源索引individual-prefix-owner-candidate-v1/verification-index.json；所有源、
实际dtype/原断言、失败和测量范围在对应冻结收据，PLAN没有修改。

'''
        end = ledger.index('\n') + 1
        ledger_path.write_text(ledger[:end] + note + ledger[end:], encoding='utf8')
    print(json.dumps(dict(receipt=ref(receipt), result=ref(main_path), measurements=speed)))


if __name__ == '__main__':
    main()
