"""Build receipts plus bounded actual-tensor finite-FA operator validation."""
import ast,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
builds=[]
for version in [1,2,3,4]:
    F=A/f'snapshot${ARTIFACT_ROOT}/codex_fa_two_sweeps_build_20260907_v{version}'
    with zipfile.ZipFile(F/'review_bundle.zip') as z:
        for n in z.namelist():
            assert '/' not in n and '\\' not in n
            (F/n).write_bytes(z.read(n))
    b=json.loads((F/'results.json').read_text());p=b['protocol']
    assert sha(F/'study.py')==p['study_sha256']
    assert sha(F/'vendor_fa_finite_p1_two_sweeps.cu')==p['extension_sha256']
    assert b['vendor_sources_before']==b['vendor_sources_after']
    assert b['status']==('finite_extension_compile_failed' if version==1 else 'finite_extension_compiled_not_executed')
    builds.append({'version':version,'status':b['status'],'raw_sha256':sha(F/'results.json'),
        'protocol_sha256':sha(F/'protocol.json'),'extension_sha256':p['extension_sha256'],
        'compile_seconds':b['wall_seconds'],'library':b.get('library'),
        'diagnosis':('Included full native parameter header requiring ATen generator headers; replaced only parameter carrier with an explicit adapter to the unchanged templated conversion.' if version==1 else
            'Generic conversion misapplied to32x32; first local operator gate failed. Follow native32x32 specialization/traits in v3; no relaxed tolerance.' if version==2 else
            'Specialized conversion still received plain logical dQ storage, whereas native accumulation uses an encoded register-copy layout. Second local gate failed; v4 pairs native accumulation layout with native conversion.' if version==3 else None)})
(A/'fa_two_sweeps_build_summary_20260907.json').write_text(json.dumps({'builds':builds,'native_vendor_sources_unchanged':True,'GPU_operator_calls':0,'attributions':0},indent=2))
s=(A/'fa_shared_mean_operator_20260907.py').read_text()
begin=s.index('runpy.run_path(');end=s.index("os.environ['MACA_PATH']",begin)
s=s[:begin]+"b=json.loads(Path(p['build_result']).read_text())\nassert sha(Path(p['build_result']))==p['build_result_sha256']\nassert b['status']=='finite_extension_compiled_not_executed'\n"+s[end:]
s=s.replace('from vendor_fa_finite_shared_mean_runtime import VendorFAFiniteP1SharedMean',
    'from vendor_fa_finite_two_sweeps_runtime import VendorFAFiniteP1TwoSweeps')
s=s.replace('from vendor_fa_finite_gqa_runtime import VendorFAFiniteP1CompactGQA',
    'from vendor_fa_finite_shared_mean_reuse_runtime import VendorFAFiniteP1SharedMeanReuse')
s=s.replace('old=VendorFAFiniteP1CompactGQA(', 'old=VendorFAFiniteP1SharedMeanReuse(')
s=s.replace("new=VendorFAFiniteP1SharedMean(A/'libdeltatrace_fa_finite_shared_mean.so',b['library']['sha256'])",
    "new=VendorFAFiniteP1TwoSweeps(p['library'],b['library']['sha256'])")
s=s.replace("'shared_mean'","'two_sweeps'")
start=s.index('        checks={}');end=s.index('        del outputs',start)
s=s[:start]+'''        checks={}
        for key in outputs['old']:
            old_array=outputs['old'][key].cpu().numpy()
            new_array=outputs['two_sweeps'][key].cpu().numpy()
            assert np.isfinite(old_array).all() and np.isfinite(new_array).all()
            a=old_array.astype(np.float64);c=new_array.astype(np.float64);error=c-a
            rms=float(np.sqrt(np.mean(a*a)));error_rms=float(np.sqrt(np.mean(error*error)))
            maximum=float(np.max(np.abs(error)))
            checks[key]={'exact':bool(np.array_equal(old_array,new_array)),
                'max_abs_difference':maximum,'old_RMS':rms,'error_RMS':error_rms,
                'relative_L2':error_rms/max(rms,1e-300),'max_abs_over_old_RMS':maximum/max(rms,1e-300),
                'sign_flips':int(np.count_nonzero((a*c)<0))}
            if key=='dq':
                for method,array in [('old',old_array),('two_sweeps',new_array)]:
                    file=A/f'{method}_dq_repeat{repeat}.npy';np.save(file,array,allow_pickle=False)
                    checks[key][method+'_array']={'file':file.name,'sha256':sha(file)}
        r['comparisons'].append(checks);save()
        assert all(checks[k]['exact'] for k in ['tau','center','dk','dv'])
        assert checks['dq']['relative_L2']<=0.001 and checks['dq']['max_abs_over_old_RMS']<=0.01
''' +s[end:]
s=s.replace('shared_mean_operator_exact_no_end_to_end_claim','two_sweeps_operator_numerical_screen_passed')
s=s.replace("['study.py','protocol.json','results.json','build_results.json','compile.log']+list(p['sources'])",
    "['study.py','protocol.json','results.json']+list(p['sources'])+[f.name for f in A.glob('*_dq_repeat*.npy')]")
ast.parse(s);(A/'fa_two_sweeps_operator_20260907.py').write_text(s)
p=json.loads((A/'fa_shared_mean_operator_protocol_20260907.json').read_text())
p.update(purpose='Six local same-real-input calls comparing shared-mean FA to two-sweep finite FA. Preserve tau/center/dK/dV; quantify dQ rounding from native-FA-style FP32 atomic summation and unchanged native convert_dQ. No new model/data/quality runs.',
    study_sha256=sha(A/'fa_two_sweeps_operator_20260907.py'),extension_sha256=sha(A/'vendor_fa_finite_p1_two_sweeps.cu'),
    old_library='${ARTIFACT_ROOT}/codex_fa_shared_mean_reuse_operator_20260907_v1/libdeltatrace_fa_finite_shared_mean_reuse.so',
    old_library_sha256='d5e0cac33f206fcbd0a3c31364fea06cc37b39165dea7c571dfd40519119665b',
    library='${ARTIFACT_ROOT}/codex_fa_two_sweeps_build_20260907_v4/libdeltatrace_fa_finite_two_sweeps.so',
    build_result='${ARTIFACT_ROOT}/codex_fa_two_sweeps_build_20260907_v4/results.json',build_result_sha256=builds[-1]['raw_sha256'],
    numerical_gate=b['protocol']['local_review_if_compiled'],
    next_if_compiled='If numerical gates pass and measured local median new/old <=1.05, allow one bounded original B1/B4 integration; otherwise stop before model runs. No threshold relaxation or parameter sweep.',
    precision_scope='FP16 matrix operands and centered coefficients, FP32 MMA/atomic accumulation, unmodified nativeFA FP32 scale/FP16 conversion. Explicitly changes dQ reduction order; this engineering gate is not a quality/sign guarantee.')
names=['vendor_fa_finite_shared_mean_reuse_runtime.py','vendor_fa_finite_two_sweeps_runtime.py']
p['sources']={n:sha(A/n) for n in names}
(A/'fa_two_sweeps_operator_protocol_20260907.json').write_text(json.dumps(p,indent=2))
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_fa_two_sweeps_operator_20260907_v3',
    'study.py='+str(A/'fa_two_sweeps_operator_20260907.py'),'protocol.json='+str(A/'fa_two_sweeps_operator_protocol_20260907.json'),
    *[n+'='+str(A/n) for n in names],'--request',str(A/'launch_fa_two_sweeps_operator_20260907.json')],check=True)
print(json.dumps(builds))
