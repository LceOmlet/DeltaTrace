"""Bind the completed current-operand run to its unchanged official assertions.

No numerical algorithm, tolerance, GPU operation or production change is added.
Original remote outputs are preserved, including the coordinate-name typo.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
AUDIT = HERE.parents[1]
OUT = HERE/'block-dtype-results'


def read(path):
    return json.loads(path.read_bytes())


def binding(path):
    raw = path.read_bytes()
    return dict(path=path.relative_to(REPO).as_posix(),bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


transport = read(OUT/'transport.json')
for item in transport:
    raw = (OUT/item['relative']).read_bytes()
    assert len(raw)==item['bytes']
    assert hashlib.sha256(raw).hexdigest()==item['sha256']
launch = read(OUT/'launch.json')
phase = read(OUT/'phase.json')
result = read(OUT/'official-result.json')
assert phase['phase']=='complete'
assert (phase['pid'],phase['birth'])==(launch['pid'],launch['birth'])
assert binding(OUT/'check_current_block_official_dtypes.py')['sha256']==launch['helper_sha256']
assert result['status']=='completed_actual_operand_comparison'
assert result['shape']==[1,64,32,128]
assert phase['shape']==[2,64,32,128]
assert phase['upstream_shape']==result['shape']
assert result['upstream_source']=='recorded finite consumer'
assert phase['initial_state_nonzero'] and not phase['cuda_initialized_before_original']

expected = {'native o':.005,'native dq':.008,'finite dq':.008,
    'native dk':.008,'finite dk':.008,'native dv':.008,'finite dv':.008,
    'native dbeta':.02,'finite dbeta':.02,'native dg':.02,'finite dg':.02}
for case in result['cases']:
    assert case['status']=='passed'
    assert {c['name']:c['threshold'] for c in case['checks']}==expected
    assert all(c['status']=='passed' for c in case['checks'])
assert {c['dtype'] for c in result['cases']}=={'torch.bfloat16','torch.float16'}

gdn = read(AUDIT/'direct-target-existing-pv-rule-20261008/v1/gdn-v2-results/results/rank0.json')
case = gdn['geometry']['candidate']
assert case['packed_slot']==2883 and case['token_id']==198
assert phase['absolute_block_start']==gdn['gdn_subops']['30']['single_capture_start']==128
assert [v['head_start'] for v in phase['source_artifacts']]==[0,8,16,24]
sources = read(AUDIT/'direct-target-existing-pv-rule-20261008/v1/joint-finite-sources.json')
selected = {v['module']:{k:v[k] for k in ('module','path','sha256')}
    for v in sources['files'] if v['module'] in
    ('qwen35_dense_finite_runner','finite_fla_gpu','profiles.qwen35_gdn_symmetric')}
source_audit = read(HERE/'official-dtype-source-audit/source.json')
assert source_audit['sha256']==phase['harness']['sha256']
assert binding(HERE/'official-dtype-source-audit/verify_saved_fla_dtypes.py')['sha256']==source_audit['sha256']

observations = [read(p) for p in sorted(HERE.glob('block-dtype-observation-*.json'))]
terminal = observations[-1]
assert terminal['phase']['phase']=='complete' and not terminal['same_birth']
assert terminal['textcraft_same_birth'] and terminal['textcraft_release_present']==[False,False]
live = next(v for v in observations if v['same_birth'])
assert live['textcraft_same_birth'] and live['textcraft_release_present']==[False,False]

receipt = dict(
    status='Completed: current actual GDN30 block satisfies the unchanged original FLA assertions in both tested dtypes.',
    case=case,launch=launch,formal_source_sha256=phase['source_sha256'],
    source_binding=dict(
        original_dtype_harness=phase['harness'],
        pinned_reference=dict(path=result['original_assertion_source'],
            sha256='35f28bf6d01f101f075309133929d1764ab540eb9a892f35eca92227e8768813'),
        finite_modules=selected,
        scope='Original saved harness is SHA pinned and executed unchanged using the source-selected sys.path. The existing source audit binds module files; this run did not emit a live imported-module path dump.'),
    inputs=dict(
        prepared=phase['prepared'],shape=phase['shape'],upstream_shape=phase['upstream_shape'],
        source_artifacts=phase['source_artifacts'],
        original_operand_dtypes=result['original_operand_dtypes'],
        recorded_initial_state_dtype=result['recorded_initial_state_dtype'],
        original_harness_initial_state_dtype=result['initial_state_dtype'],
        recorded_upstream_dtype=result['recorded_upstream_dtype'],
        upstream_source=result['upstream_source'],initial_state_nonzero=True,
        cuda_initialized_before_original=False,
        selection='Saved factual first single-intervention 64-token block; all four original eight-head groups concatenated. Factual rows equal the original single-intervention factual operands. The original harness forms coincident endpoints.'),
    coordinate_metadata_correction=dict(
        original_phase_field='absolute_block_start',original_value=128,
        correct_meaning='capture-local block offset, not original packed token position',
        token_packed_slot=2883,
        action='Original raw phase/helper preserved; future helper field renamed capture_local_block_start. No input selection, computed result or remote run was changed.'),
    official_checks=dict(cases=result['cases'],thresholds=expected,total_passed=22,total_failed=0,
        tolerance_changed=False,warning_exemption_enabled=False,
        scope='Original output and q/k/v/beta/g adjoint checks against the pinned original recurrent reference and fla.utils.assert_close. No ht/dh0 check was emitted by this harness.'),
    resources=dict(
        launch_to_completion_seconds=phase['unix']-launch['launched_unix'],
        case_seconds={c['dtype']:c['seconds'] for c in result['cases']},
        torch_peak_allocated_bytes=phase['torch_peak_allocated_bytes'],
        final_pss_bytes=phase['pss_bytes'],live_observed_pss_bytes=live['current_PSS_bytes'],
        physical_gpu4_observed_mib=1996,physical_gpu4_terminal_mib=859,
        scope='Physical memory is one live mx-smi observation, not a continuously sampled peak. Case time includes cold compilation/initialization and is not a throughput comparison.'),
    formal_state=dict(TextCraft_same_birth=True,TextCraft_update_release_present=[False,False],
        AppWorld_formal_terminal=True,formal_restart=False,credit_repaired=False),
    operations=dict(model=0,full_DT=0,rollout=0,optimizer=0,checkpoint_restore=0,
        operator_forward_and_autograd=True,finite_operator_adjoint=True,
        production_modified=False),
    limitations=[
        'Coincident-endpoint operator checks do not certify a nonzero whole-model finite attribution or factual single-deletion credit.',
        'Actual FP16 native adjoint RMS ratios can report zero while max_abs is nonzero; the original assertion was not replaced and this is not bitwise equality.',
        'This current 64-token block does not supersede historical failing checks on other recorded operands.',
        'The previously measured joint-versus-single-deletion disagreement and original author curve evidence remain unchanged. No credit correction was deployed.',
        'The independent consumed-FA-cache OOM repair has separate failed-B4 and exact32768 receipts; this operator test is not a capacity acceptance.'],
    evidence=[binding(OUT/'transport.json'),binding(OUT/'phase.json'),binding(OUT/'official-result.json'),
        binding(OUT/'launch.json'),binding(HERE/'block-dtype-observation-1791414754.json'),
        binding(HERE/'block-dtype-observation-1791414791.json'),
        binding(AUDIT/'direct-target-existing-pv-rule-20261008/v1/gdn-v2-results/results/rank0.json'),
        binding(AUDIT/'direct-target-existing-pv-rule-20261008/v1/joint-finite-sources.json'),
        binding(REPO/'experiments/rl/PLAN.md')])
target = REPO/'experiments/rl/results_current_extreme_official_block_dtype_20261008.json'
assert not target.exists()
target.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

runtime = REPO/'experiments/rl/current_runtime.json'
raw = runtime.read_bytes()
key = 'latest_current_extreme_official_GDN_block_dtype'
assert key not in json.loads(raw) and raw.endswith(b'}\r\n')
value = dict(receipt=binding(target),launch=launch,passed=22,failed=0,
    scope=receipt['official_checks']['scope'],resources=receipt['resources'],
    formal_state=receipt['formal_state'],coordinate_metadata_correction=receipt['coordinate_metadata_correction'])
tail='\r\n'.join(json.dumps({key:value},ensure_ascii=False,indent=2).splitlines()[1:-1]).encode('utf8')
runtime.write_bytes(raw[:-3].rstrip(b'\r\n')+b',\r\n'+tail+b'\r\n}\r\n')
json.loads(runtime.read_bytes())

ledger = REPO/'experiments/rl/RUNTIME_RECORD.md'
raw = ledger.read_bytes()
heading = '## 2026-10-08 当前AppWorld极端点GDN30实际块完成原FLA dtype对照，信用未修复'
assert heading.encode('utf8') not in raw
note = '''
9bb32def/PID3281630/birth1791414714.78，物理GPU4。
使用原保存的当前newline2883、GDN30首个single-intervention 64-token块，
四个8-head拼为32-head；实际q/k/v/beta FP16、raw_g FP32、原非零initial state，
原saved upstream，无随机替代。原verify_saved_fla_dtypes.py SHA386a389f...未修改，
调用原FLA 0.4.1 reference/assert_close，FP16与BF16对照各11项通过，未改原阈值。
这是重合端点的o/dq/dk/dv/dbeta/dg检查，不含ht/dh0，不证明非零有限整网归因准确。
FP16原native adjoint部分RMS比显示0但max_abs非零，不能称逐位相同。
此输入通过不覆盖历史不同输入的dk超差，也不覆盖整条PPO或归因质量。
约60.09秒含准备/冷编译；torch peak allocated 564818432B，结束PSS6415631360B，
live物理GPU4观测1996MiB，结束859MiB，物理观测不是连续峰值。
原phase字段absolute_block_start=128实际是capture-local offset；原回执逐字保留，
仅未来helper改名capture_local_block_start。未改输入/结果，也未为元数据重跑GPU。
首次launch引号导致远端Python parse失败，未创建测试进程/GPU；修正后只启动一次。
1791414791.8797884 terminal观测：TextCraft原PID/birth仍hold、双rank release不存在；
AppWorld正式terminal。信用问题仍未修复，无新精化/倍率/裁剪，无训练重启/恢复。
原AppWorld consumed-FA-cache生命周期修复和failed-B4/精确32k验收单独保留。
'''
end = raw.index(b'\n')+1
ledger.write_bytes(raw[:end]+('\n'+heading+'\n\n'+note.lstrip()+'\n\n').encode('utf8')+raw[end:])
print(json.dumps(dict(receipt=binding(target),passed=22,failed=0,
    source_module_count=len(selected),credit_repaired=False),ensure_ascii=False))
