"""Freeze a small real-operand layout check, including actual nonzero view offsets."""
import ast,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
F=A/'snapshot${ARTIFACT_ROOT}/codex_fa_native_strides_build_20260907_v1'
with zipfile.ZipFile(F/'review_bundle.zip') as z:
    for name in z.namelist():
        assert '/' not in name and '\\' not in name
        (F/name).write_bytes(z.read(name))
b=json.loads((F/'results.json').read_text())
assert b['status']=='finite_extension_compiled_not_executed'
assert b['vendor_sources_before']==b['vendor_sources_after']
assert sha(F/'study.py')==b['protocol']['study_sha256']
assert sha(F/'vendor_fa_finite_p1_native_strides.cu')==b['protocol']['extension_sha256']
(A/'fa_native_strides_build_summary_20260907.json').write_text(json.dumps({
    'status':b['status'],'raw_sha256':sha(F/'results.json'),
    'protocol_sha256':sha(F/'protocol.json'),'library_sha256':b['library']['sha256'],
    'compile_seconds':b['wall_seconds'],'native_vendor_sources_unchanged':True,
    'operator_calls':0,'attributions':0,'quality_queries':0},indent=2))
s=(A/'fa_shared_mean_operator_20260907.py').read_text()
begin=s.index("runpy.run_path(");end=s.index("os.environ['MACA_PATH']",begin)
s=s[:begin]+"b=json.loads(Path(p['build_result']).read_text())\nassert sha(Path(p['build_result']))==p['build_result_sha256']\nassert b['status']=='finite_extension_compiled_not_executed'\n"+s[end:]
s=s.replace('from vendor_fa_finite_shared_mean_runtime import VendorFAFiniteP1SharedMean',
    'from vendor_fa_finite_native_strides_runtime import VendorFAFiniteP1NativeStrides')
s=s.replace('from vendor_fa_finite_gqa_runtime import VendorFAFiniteP1CompactGQA',
    'from vendor_fa_finite_shared_mean_reuse_runtime import VendorFAFiniteP1SharedMeanReuse')
s=s.replace('old=VendorFAFiniteP1CompactGQA(', 'old=VendorFAFiniteP1SharedMeanReuse(')
s=s.replace("new=VendorFAFiniteP1SharedMean(A/'libdeltatrace_fa_finite_shared_mean.so',b['library']['sha256'])", """# Recorded default FA layout is [B,N,H,D], exposed to attribution as [B,H,N,D].
# Repack these same captured real values before timing; both methods get identical
# noncontiguous views with an endpoint-like nonzero storage offset. No new samples.
for name in ['q0','k0','q1','k1','v0','u']:
    value=compact[name];batch,heads,length,dim=value.shape
    backing=torch.empty((batch*2,length,heads,dim),device=value.device,dtype=value.dtype)
    view=backing[batch:].transpose(1,2);view.copy_(value)
    assert torch.equal(view,value) and view.storage_offset()>0 and not view.is_contiguous()
    compact[name]=view
new=VendorFAFiniteP1NativeStrides(p['library'],b['library']['sha256'])""")
s=s.replace('shared_mean','native_strides')
# Restore baseline import after descriptive mode replacement.
s=s.replace('vendor_fa_finite_native_strides_reuse_runtime','vendor_fa_finite_shared_mean_reuse_runtime')
s=s.replace("['study.py','protocol.json','results.json','build_results.json','compile.log']", "['study.py','protocol.json','results.json']")
s=s.replace('Shared-tile midpoint must preserve the existing coefficient and output quantization.',
    'Address-only FA input strides must preserve every finite output.')
s=s.replace("'calls':[],'comparisons':[],", "'calls':[],'comparisons':[],'tested_input_layouts':{k:{'shape':list(v.shape),'strides':list(v.stride()),'storage_offset':v.storage_offset()} for k,v in compact.items()},")
ast.parse(s);(A/'fa_native_strides_operator_20260907.py').write_text(s)
p=json.loads((A/'fa_shared_mean_operator_protocol_20260907.json').read_text())
p.update(purpose='Six finite operator calls on pinned actual NI0 layer35 values; both versions receive identical native-FA-layout noncontiguous views with nonzero storage offsets. Address-only change. Baseline is verified shared-mean reuse, not older compact GQA. Zero model or quality calls.',
    study_sha256=sha(A/'fa_native_strides_operator_20260907.py'),
    extension_sha256=sha(A/'vendor_fa_finite_p1_native_strides.cu'),
    old_library='${ARTIFACT_ROOT}/codex_fa_shared_mean_reuse_operator_20260907_v1/libdeltatrace_fa_finite_shared_mean_reuse.so',
    old_library_sha256='d5e0cac33f206fcbd0a3c31364fea06cc37b39165dea7c571dfd40519119665b',
    library='${ARTIFACT_ROOT}/codex_fa_native_strides_build_20260907_v1/libdeltatrace_fa_finite_native_strides.so',
    build_result='${ARTIFACT_ROOT}/codex_fa_native_strides_build_20260907_v1/results.json',build_result_sha256=sha(F/'results.json'),
    next_if_compiled='Six local operator checks only. If exact, bounded original NI2 B1 and original NI0/3/6,MH1 B4 integration; no automatic quality sweep.',
    input_layout_provenance={'source':'fa_shared_mean_integration_numeric_20260907.json',
        'sha256':sha(A/'fa_shared_mean_integration_numeric_20260907.json'),
        'path':'groups[0].runs[0].result.native_paired_replay_layouts[0]',
        'native_order':'[B,N,H,D]; finite input is transpose(1,2). Actual tensor values from old NI0 capture.'})
names=['vendor_fa_finite_shared_mean_reuse_runtime.py','vendor_fa_finite_native_strides_runtime.py']
p['sources']={n:sha(A/n) for n in names}
(A/'fa_native_strides_operator_protocol_20260907.json').write_text(json.dumps(p,indent=2))
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_fa_native_strides_operator_20260907_v1',
    'study.py='+str(A/'fa_native_strides_operator_20260907.py'),
    'protocol.json='+str(A/'fa_native_strides_operator_protocol_20260907.json'),
    *[n+'='+str(A/n) for n in names],
    '--request',str(A/'launch_fa_native_strides_operator_20260907.json')],check=True)
print(json.dumps({'build_verified':True,'library_sha256':b['library']['sha256'],'compile_seconds':b['wall_seconds'],'next_operator_calls':6}))
