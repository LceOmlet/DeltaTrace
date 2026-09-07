"""Source provenance/build audit only; no numerical/runtime equivalence claim."""
import hashlib,json
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
records=[]
for directory,study,protocol,suffix,status in [
 ('codex_vendor_fa_source_probe_20260907_v1','vendor_fa_source_probe_20260907.py','vendor_fa_source_probe_protocol_20260907.json','.cpp','unmodified_vendor_compile_failed'),
 ('codex_vendor_fa_source_gpu_language_20260907_v1','vendor_fa_source_gpu_language_probe_20260907.py','vendor_fa_source_gpu_language_protocol_20260907.json','.cu','compiled_unmodified_vendor_source')]:
    F=A/'snapshot/tmp'/directory;d=json.loads((F/'results.json').read_text());p=d['protocol']
    assert p==json.loads((A/protocol).read_text())
    assert sha(F/'study.py')==sha(A/study)==p['study_sha256']
    assert sha(F/'source_manifest.json')==p['source_manifest_sha256']
    m=json.loads((F/'source_manifest.json').read_text());assert m==p['source']
    assert d['source_files_verified']==len(m['files'])==78 and d['status']==status
    assert d['source_files_after']=={k:v['sha256'] for k,v in m['files'].items()}
    assert sha(F/('flash_bwd_hdim128_fp16_sm80'+suffix))==m['files']['csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu']['sha256']
    assert all(d[k]==0 for k in ['model_forwards','model_backwards','attribution_calls','GPU_kernel_launches'])
    if suffix=='.cu':assert d['compile_returncode']==0 and d['object']['bytes']>0
    else:assert d['compile_returncode']!=0 and 'argument unused during compilation' in (F/'compile.log').read_text()
    records.append({'directory':directory,'status':status,'result_sha256':sha(F/'results.json'),'compile_log_sha256':sha(F/'compile.log'),
      'wall_seconds':d['wall_seconds'],'source_files_verified':78,'object':d.get('object'),'compiler_command':d['compile_command']})
out={'status':'verified_source_build_diagnostics_only','source_repository':m['repository'],'commit':m['commit'],
 'source_version':m['declared_fa_version'],'installed_version':m['installed_fa_version'],
 'installed_binary_source_match_proven':False,'source_patches':0,'model_or_attribution_calls':0,
 'finite_extension_implemented':False,'runtime_default_changed':False,'records':records,
 'decision':'Preserve genuine source/build evidence. User prefers stable default FA public contracts and efficiency; do not adopt this historical-source fork as the default runtime.'}
(A/'vendor_fa_source_probes_summary_20260907.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'status':out['status'],'source_files':78,'source_patches':0,'build_statuses':[r['status'] for r in records]}))
