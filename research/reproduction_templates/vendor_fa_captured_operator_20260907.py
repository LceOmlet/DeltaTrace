import hashlib,json,os,sys,time,traceback,zipfile,gc,shutil
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

def dense(ops,scale):
    from compiled_finite_rules import logarithmic_mean_with_checks
    q0,k0,q1,k1,v0,u=[ops[k].float() for k in ['q0','k0','q1','k1','v0','u']]
    n=q0.shape[-2];mask=torch.arange(n,device=q0.device)[None,:]>torch.arange(n,device=q0.device)[:,None]
    z0=(q0@k0.transpose(-1,-2))*scale;z1=(q1@k1.transpose(-1,-2))*scale
    z0=z0.masked_fill(mask,float('-inf'));z1=z1.masked_fill(mask,float('-inf'))
    mean,checks=logarithmic_mean_with_checks(z0,z1)
    t=u@v0.transpose(-1,-2);tau=mean.sum(-1);center=(mean*t).sum(-1)/tau
    ds=mean*(t-center[...,None]);p1=(z1-ops['lse1'][...,None]).exp()
    return {'dq':ds@((k0+k1)*.5)*scale,'dk':ds.transpose(-1,-2)@((q0+q1)*.5)*scale,
        'dv':p1.transpose(-1,-2)@u,'tau':tau,'center':center}

def standard_FA(ops,scale,groups):
    # Actual native GQA configuration, same original endpoint/U. Cost reference only.
    with torch.enable_grad():
        q=ops['q1'].transpose(1,2).detach().contiguous().requires_grad_(True)
        k=ops['k1'][:,::groups].transpose(1,2).detach().contiguous().requires_grad_(True)
        v=ops['v1'][:,::groups].transpose(1,2).detach().contiguous().requires_grad_(True)
        out=fa_native.flash_attn_func(q,k,v,dropout_p=0.,softmax_scale=scale,causal=True)
        grads=torch.autograd.grad(out,(q,k,v),ops['u'].transpose(1,2).half().contiguous())
    return {name:value.transpose(1,2) for name,value in zip(['dq','dk','dv'],grads)}

def operator_call(kind,ops,record,repeat,profile=False):
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    attempt={'kind':kind,'repeat':repeat,'warmup':repeat==0,'profile':profile,'complete':False,'activity':{}}
    report['operator_attempts'].append(attempt)
    try:
        with torch.no_grad():
            if kind=='finite_FA':values=extension(ops,record['scale'],attempt['activity'])
            elif kind=='dense_P1':values=dense(ops,record['scale'])
            else:values=standard_FA(ops,record['scale'],record['groups'])
        torch.cuda.synchronize();attempt.update(complete=True,seconds=time.perf_counter()-tick,
            resident_bytes=resident,peak_bytes=torch.cuda.max_memory_allocated(),
            incremental_peak_bytes=torch.cuda.max_memory_allocated()-resident)
        assert all(torch.isfinite(v).all() for v in values.values())
        if kind!='standard_FA' and repeat==0:
            attempt['output_arrays']={name:array(kind+'_'+name,value) for name,value in values.items()}
        attempt['output_stats']={name:{'shape':list(v.shape),'dtype':str(v.dtype),
            'sha256_values':hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()} for name,v in values.items()}
        return attempt
    except Exception:
        attempt['error']=traceback.format_exc();attempt['seconds']=time.perf_counter()-tick;raise
    finally:save()

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
