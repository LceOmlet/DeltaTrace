"""Verify local stride addressing, then freeze the original-cache B1/B4 check."""
import ast,hashlib,json,statistics,subprocess,sys,zipfile
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
F=A/'snapshot${ARTIFACT_ROOT}/codex_fa_native_strides_operator_20260907_v1'
with zipfile.ZipFile(F/'review_bundle.zip') as z:
    for name in z.namelist():
        assert '/' not in name and '\\' not in name
        (F/name).write_bytes(z.read(name))
d=json.loads((F/'results.json').read_text());p=d['protocol']
assert p==json.loads((F/'protocol.json').read_text())
assert sha(F/'study.py')==p['study_sha256']
for n,h in p['sources'].items():assert sha(F/n)==h
assert d['status']=='native_strides_operator_exact_no_end_to_end_claim'
assert d['attributions']==d['native_model_forwards']==d['VJPs']==d['quality_queries']==0
assert len(d['calls'])==6
for repeat in range(3):
    old,new=[next(x for x in d['calls'] if x['repeat']==repeat and x['name']==n) for n in ['old','native_strides']]
    assert old['outputs']==new['outputs']
    assert all(x['exact'] and x['max_abs_difference']==0 for x in d['comparisons'][repeat].values())
    a=new['activity'];assert a['QKV_copy_bytes']==0
    assert a['calls_attempted']==a['calls_enqueued']==1
    for n in ['q0','k0','q1','k1','v0']:
        assert a['input_storage_reused'][n] and a['input_strides'][n]==a['kernel_input_strides'][n]
        assert d['tested_input_layouts'][n]['storage_offset']>0
times={n:[x['seconds'] for x in d['calls'] if x['name']==n and not x['warm']] for n in ['old','native_strides']}
summary={'status':'native_FA_stride_addressing_exact_on_captured_real_operands','raw_sha256':sha(F/'results.json'),
    'protocol_sha256':sha(F/'protocol.json'),'library_sha256':d['build_library']['sha256'],
    'operator_calls':6,'attributions':0,'quality_queries':0,'full_outputs_exact':True,
    'measured_seconds':times,'median_time_ratio':statistics.median(times['native_strides'])/statistics.median(times['old']),
    'input_layouts':d['tested_input_layouts'],'QKV_storage_reused':True,
    'scope':'Pinned actual NI0 layer35 values placed in the observed native FA layout with nonzero endpoint-like offsets. Layout validation, not a new benchmark or whole-model performance claim.'}
(A/'fa_native_strides_operator_summary_20260907.json').write_text(json.dumps(summary,indent=2))

s=(A/'fa_shared_mean_integration_20260907.py').read_text()
s=s.replace('from vendor_fa_finite_gqa_runtime import VendorFAFiniteP1CompactGQA',
    'from vendor_fa_finite_native_strides_runtime import VendorFAFiniteP1NativeStrides')
s=s.replace("compact_extension=VendorFAFiniteP1SharedMeanReuse(p['shared_mean_library'],p['shared_mean_library_sha256'])",
    "compact_extension=VendorFAFiniteP1NativeStrides(p['strided_library'],p['strided_library_sha256'])")
s=s.replace("old_extension=VendorFAFiniteP1CompactGQA(p['compact_library'],p['compact_library_sha256'])",
    "old_extension=VendorFAFiniteP1SharedMeanReuse(p['shared_mean_library'],p['shared_mean_library_sha256'])")
# Replace only mode/profile/status strings; baseline runtime/library names stay intact.
s=s.replace("'shared_mean'", "'native_strides'").replace('shared_mean_B4_profile','native_strides_B4_profile')
s=s.replace('FA_shared_mean_B1_and_real_B4_exact','FA_native_strides_B1_and_real_B4_exact')
s=s.replace('FA_SHARED_MEAN','FA_NATIVE_STRIDES')
ast.parse(s);(A/'fa_native_strides_integration_20260907.py').write_text(s)
p=json.loads((A/'fa_shared_mean_integration_protocol_20260907.json').read_text())
p.update(purpose='FA first: address-only native stride integration versus current shared-mean reuse. Original NI2 B1 and original NI0/3/6,MH1 true B4, one warm plus two alternating measured pairs per group,12 whole-model attributions. No MLP/head changes, generation, quality curves, FT or VJPs. Real model FA, signed finite P1, FP16/FP32 arithmetic and three finite passes unchanged.',
    study_sha256=sha(A/'fa_native_strides_integration_20260907.py'),
    wait_for_pid=177899,wait_for_script='${ARTIFACT_ROOT}/codex_fa_native_strides_operator_20260907_v1/study.py',
    runtime_source_parent='${ARTIFACT_ROOT}/codex_fa_shared_mean_integration_20260907_v1',
    strided_library='${ARTIFACT_ROOT}/codex_fa_native_strides_build_20260907_v1/libdeltatrace_fa_finite_native_strides.so',
    strided_library_sha256=summary['library_sha256'],
    stride_operator_raw_sha256=summary['raw_sha256'],
    attribution_batch_scope='B1 NI2 N1241 and B4 distinct NI0/3/6,MH1 N607; physical native endpoint batch2/8. One warm+two measured per method per group. Candidate B4 warm contains the profiler and is excluded from measured latency.12 total calls, no extra profile call.',
    predeclared_review={
        'numerics':'Address-only change: compare all same-job full signed values and endpoint scores; fail on mismatch. Does not establish independent quality or token-deletion signs.',
        'memory':'Matched shared-mean FA baseline, report full peak and removed temporary allocations; no fixed additional-memory ceiling.',
        'performance':'Two alternating measured pairs per shape; preserve all timings and caveats. Inspect actual B4 finite/native FA dispatch and preparation copies in the existing warm call. No cross-job speedup multiplication or FT claim.',
        'stop':'Stop on source/model/finite validity failure; no shape or parameter sweep.'})
n='vendor_fa_finite_native_strides_runtime.py';p['sources'][n]=sha(A/n)
for name,h in p['sources'].items():assert sha(A/name)==h
(A/'fa_native_strides_integration_protocol_20260907.json').write_text(json.dumps(p,indent=2))
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_fa_native_strides_integration_20260907_v1',
    'study.py='+str(A/'fa_native_strides_integration_20260907.py'),
    'protocol.json='+str(A/'fa_native_strides_integration_protocol_20260907.json'),n+'='+str(A/n),
    '--request',str(A/'launch_fa_native_strides_integration_20260907.json')],check=True)
assert len(json.loads((A/'launch_fa_native_strides_integration_20260907.json').read_text())['cmd'].encode())<100000
print(json.dumps(summary))
