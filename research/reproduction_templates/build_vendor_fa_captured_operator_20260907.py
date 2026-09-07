"""Freeze a standalone operator test on the failed capture's unchanged tensors.

The original exact-vector guard remains failed. No recapture or model rerun.
"""
import hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
F=A/'snapshot${ARTIFACT_ROOT}/codex_vendor_fa_finite_actual_20260907_v1'
d=json.loads((F/'results.json').read_text());old=d['protocol']
assert d['status']=='failed' and not d['operator_attempts'] and len(d['captured_layers'])==1
driver='''import hashlib,json,os,sys,time,traceback,zipfile,gc,shutil
from pathlib import Path
os.environ['MACA_PATH']='/opt/maca'
HERE=Path(__file__).resolve().parent
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(HERE/'inductor_cache')
os.environ['TRITON_CACHE_DIR']=str(HERE/'triton_cache')
p=json.loads((HERE/'protocol.json').read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(HERE/'study.py')==p['study_sha256']
for name,digest in p['sources'].items():assert sha(HERE/name)==digest
source=Path(p['capture_results']);assert sha(source)==p['capture_results_sha256']
capture=json.loads(source.read_text())
assert capture['status']=='failed' and not capture['operator_attempts']
record=capture['captured_layers'][0];assert record==p['actual_operands']
for item in record['operands'].values():
    origin=source.parent/item['file'];assert sha(origin)==item['sha256']
    shutil.copyfile(origin,HERE/item['file'])
import numpy as np,torch
import flash_attn.flash_attn_interface as fa_native
from vendor_fa_finite_runtime import VendorFAFiniteP1
extension=VendorFAFiniteP1(p['library'],p['library_sha256'])
report={'status':'running','protocol':p,'operator_attempts':[],
        'native_root_forwards':0,'native_vjps':0,'manual_passes':0,
        'capture_exact_parent_guard':'failed_in_predecessor_not_relaxed',
        'device':torch.cuda.get_device_name(),'torch':torch.__version__}
def native_sources():
    return {name:sha(Path(path)) for name,path in p['native_source_paths'].items()}
report['native_sources_before']=native_sources()
assert report['native_sources_before']==p['native_source_sha256']
def save():(HERE/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
def array(name,value):
    path=HERE/(name+'.npy');np.save(path,value.detach().cpu().numpy(),allow_pickle=False)
    return {'file':path.name,'sha256':sha(path),'shape':list(value.shape),'dtype':str(value.dtype)}
'''
original=(A/'vendor_fa_finite_actual_probe_20260907.py').read_text()
driver+=original[original.index('\ndef dense('):original.index('\nsave()\ntry:\n')]
driver+='''
save();started=time.time()
try:
    ops={name:torch.from_numpy(np.load(HERE/item['file'],allow_pickle=False)).cuda() for name,item in record['operands'].items() if not name.startswith('expected_') and name not in ['out0','out1']}
    for repeat in range(4):
        order=['dense_P1','finite_FA','standard_FA'];offset=repeat%3;order=order[offset:]+order[:offset]
        for kind in order:
            attempt=operator_call(kind,ops,record,repeat)
            print('OPERATOR_DONE',kind,repeat,attempt['seconds'],attempt['incremental_peak_bytes'],flush=True)
    report['profiles']={}
    for kind in ['finite_FA','standard_FA']:
        with torch.profiler.profile(activities=list(torch.profiler.supported_activities())) as prof:
            operator_call(kind,ops,record,4,profile=True)
        trace=kind+'_trace.json';prof.export_chrome_trace(str(HERE/trace))
        report['profiles'][kind]={'trace':trace,'sha256':sha(HERE/trace),
            'GPU_kernels':[e.name for e in prof.events() if str(e.device_type)=='DeviceType.CUDA']}
        del prof;save()
    report['native_sources_after']=native_sources();assert report['native_sources_before']==report['native_sources_after']
    assert sha(Path(p['library']))==p['library_sha256']
    assert len(report['operator_attempts'])==14
    report['status']='complete'
except Exception:
    report['status']='failed';report['error']=traceback.format_exc();raise
finally:
    report['elapsed_seconds']=time.time()-started;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','results.json']+list(p['sources'])+[x.name for x in HERE.glob('*_trace.json')]:z.write(HERE/name,name)
    with zipfile.ZipFile(HERE/'actual_tensor_arrays.zip','w',zipfile.ZIP_DEFLATED) as z:
        for x in HERE.glob('*.npy'):z.write(x,x.name)
'''
study=A/'vendor_fa_captured_operator_20260907.py';study.write_text(driver,encoding='utf-8')
native=d['native_sources_before']
paths={Path(k).name:k for k in native}
hashes={Path(k).name:v for k,v in native.items()}
p={
 'purpose':'Local numerical, memory, and speed test of the vendor FA finite extension on actual original NI0 layer35 tensors; not a benchmark, end-to-end pass, or recapture approval.',
 'capture_results':'${ARTIFACT_ROOT}/codex_vendor_fa_finite_actual_20260907_v1/results.json',
 'capture_results_sha256':sha(F/'results.json'),'actual_operands':d['captured_layers'][0],
 'capture_failure_preserved':'Exact parent-vector check failed before any finite operator ran. Local operator comparison uses that captured run as its own fixed reference; cause unresolved.',
 'study_sha256':sha(study),'sources':{name:sha(A/name) for name in ['vendor_fa_finite_runtime.py','compiled_finite_rules.py','signed_secant_rules.py']},
 'library':old['library'],'library_sha256':old['library_sha256'],
 'build_result':old['build_result'],'build_result_sha256':old['build_result_sha256'],
 'native_source_paths':paths,'native_source_sha256':hashes,
 'operator_budget':old['operator_budget'],'predeclared_numerical_review':old['predeclared_numerical_review'],
 'cost_scope':old['cost_scope'],'model_budget':{'forward':0,'backward':0,'attribution':0},
}
protocol=A/'vendor_fa_captured_operator_protocol_20260907.json';protocol.write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_vendor_fa_captured_operator_20260907_v2',
    'study.py='+str(study),'protocol.json='+str(protocol)]+[name+'='+str(A/name) for name in p['sources']]+[
    '--request',str(A/'vendor_fa_captured_operator_launch_20260907.json')],check=True)
