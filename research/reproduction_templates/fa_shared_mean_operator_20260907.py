"""Bounded FA GQA integration probe on previously captured real operands."""
import gc,hashlib,json,os,runpy,time,traceback,zipfile
from pathlib import Path
A=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
p=json.loads((A/'protocol.json').read_text())
assert sha(A/'study.py')==p['study_sha256']
for name,digest in p['sources'].items(): assert sha(A/name)==digest
runpy.run_path(str(A/'build.py'),run_name='__main__')
b=json.loads((A/'results.json').read_text())
(A/'build_results.json').write_text(json.dumps(b,indent=2))
assert b['status']=='finite_extension_compiled_not_executed'
assert b['vendor_sources_before']==b['vendor_sources_after']
os.environ['MACA_PATH']='/opt/maca'
import numpy as np, torch
from vendor_fa_finite_shared_mean_runtime import VendorFAFiniteP1SharedMean
from vendor_fa_finite_gqa_runtime import VendorFAFiniteP1CompactGQA
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
old=VendorFAFiniteP1CompactGQA(p['old_library'],p['old_library_sha256'])
new=VendorFAFiniteP1SharedMean(A/'libdeltatrace_fa_finite_shared_mean.so',b['library']['sha256'])
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
        for name in (['old','shared_mean'] if repeat!=1 else ['shared_mean','old']):
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
            a,c=outputs['old'][key],outputs['shared_mean'][key]
            assert torch.isfinite(a).all() and torch.isfinite(c).all()
            checks[key]={'exact':bool(torch.equal(a,c)),
                         'max_abs_difference':float((a.float()-c.float()).abs().max())}
        r['comparisons'].append(checks);save()
        assert all(x['exact'] for x in checks.values()),'Shared-tile midpoint must preserve the existing coefficient and output quantization.'
        del outputs
    r['status']='shared_mean_operator_exact_no_end_to_end_claim'
except Exception:
    r['status']='failed';r['error']=traceback.format_exc();raise
finally:
    save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','results.json','build_results.json','compile.log']+list(p['sources']):
            z.write(A/name,name)
    print(json.dumps({'status':r['status'],'operator_calls':len(r['calls']),'attributions':0}),flush=True)
