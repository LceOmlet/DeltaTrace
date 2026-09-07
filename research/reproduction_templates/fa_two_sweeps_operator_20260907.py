"""Bounded FA GQA integration probe on previously captured real operands."""
import gc,hashlib,json,os,runpy,time,traceback,zipfile
from pathlib import Path
A=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
p=json.loads((A/'protocol.json').read_text())
assert sha(A/'study.py')==p['study_sha256']
for name,digest in p['sources'].items(): assert sha(A/name)==digest
b=json.loads(Path(p['build_result']).read_text())
assert sha(Path(p['build_result']))==p['build_result_sha256']
assert b['status']=='finite_extension_compiled_not_executed'
os.environ['MACA_PATH']='/opt/maca'
import numpy as np, torch
from vendor_fa_finite_two_sweeps_runtime import VendorFAFiniteP1TwoSweeps
from vendor_fa_finite_shared_mean_reuse_runtime import VendorFAFiniteP1SharedMeanReuse
source=Path(p['capture_results'])
assert sha(source)==p['capture_results_sha256']
captured=json.loads(source.read_text())
record=captured['captured_layers'][0]
assert record==p['actual_operands']
ops={}
for name in ['q0','k0','q1','k1','v0','u','lse0','lse1']:
    item=record['operands'][name]; f=source.parent/item['file']
    assert sha(f)==item['sha256']
    ops[name]=torch.from_numpy(np.load(f,allow_pickle=False)).cuda()
compact=dict(ops)
for name in ['k0','k1','v0']:
    compact[name]=ops[name][:,::record['groups']].contiguous()
    assert torch.equal(compact[name].repeat_interleave(record['groups'],dim=1),ops[name])
old=VendorFAFiniteP1SharedMeanReuse(p['old_library'],p['old_library_sha256'])
new=VendorFAFiniteP1TwoSweeps(p['library'],b['library']['sha256'])
r={'status':'running','protocol':p,'build_library':b['library'],
   'attributions':0,'native_model_forwards':0,'VJPs':0,'quality_queries':0,
   'calls':[],'comparisons':[],
   'scope':'One previously captured original NI0 layer35, B1,Hq32,Hkv8,N601. Operator integration only; no new benchmark or whole-model speed/quality claim. Historical capture exact-vector gate failure is not changed.'}
def save():
    t=A/'results.partial';t.write_text(json.dumps(r,indent=2));t.replace(A/'results.json')
save()
try:
    for repeat in range(3):
        outputs={}
        for name in (['old','two_sweeps'] if repeat!=1 else ['two_sweeps','old']):
            assert len(r['calls'])<6
            gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
            base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            activity={};tick=time.perf_counter()
            outputs[name]=(old if name=='old' else new)(compact,record['scale'],activity)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-tick
            r['calls'].append({'name':name,'repeat':repeat,'warm':repeat==0,'seconds':elapsed,
                'incremental_peak_bytes':torch.cuda.max_memory_allocated()-base,
                'activity':activity,
                'outputs':{k:{'sha256':hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest(),
                              'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in outputs[name].items()}})
            save()
        checks={}
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
        del outputs
    r['status']='two_sweeps_operator_numerical_screen_passed'
except Exception:
    r['status']='failed';r['error']=traceback.format_exc();raise
finally:
    save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','results.json']+list(p['sources'])+[f.name for f in A.glob('*_dq_repeat*.npy')]:
            z.write(A/name,name)
    print(json.dumps({'status':r['status'],'operator_calls':len(r['calls']),'attributions':0}),flush=True)
