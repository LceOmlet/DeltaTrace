"""Compile explicitly named finite P1 extension against pinned genuine FA headers."""
import hashlib,json,os,subprocess,time,traceback,zipfile
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
p=json.loads((A/'protocol.json').read_text());assert sha(A/'study.py')==p['study_sha256']
assert sha(A/'vendor_fa_finite_p1_shared_mean.cu')==p['extension_sha256']
parent=Path(p['vendor_source_parent']);receipt=json.loads((parent/'results.json').read_text())
assert sha(parent/'results.json')==p['vendor_parent_sha256'] and receipt['status']=='compiled_unmodified_vendor_source'
manifest=json.loads((parent/'source_manifest.json').read_text());root=parent/'vendor_source'
def source_receipt():
    found={name:sha(root/name) for name in manifest['files']}
    assert all(found[name]==row['sha256'] for name,row in manifest['files'].items())
    return found
r={'status':'running','protocol':p,'native_model_forwards':0,'native_model_backwards':0,
   'attribution_calls':0,'GPU_kernel_launches':0,'vendor_sources_before':source_receipt()}
def save():
    t=A/'results.partial';t.write_text(json.dumps(r,indent=2));t.replace(A/'results.json')
save();start=time.time()
try:
    sdk=Path('/opt/maca');env=os.environ.copy();env.update(MACA_PATH=str(sdk),CUDA_PATH=str(sdk/'tools/cu-bridge'),MACA_CLANG_PATH=str(sdk/'mxgpu_llvm/bin'))
    env['PATH']=str(sdk/'tools/cu-bridge/bin')+':'+str(sdk/'mxgpu_llvm/bin')+':'+env.get('PATH','')
    env['LD_LIBRARY_PATH']=str(sdk/'lib')+':'+str(sdk/'mxgpu_llvm/lib')+':'+env.get('LD_LIBRARY_PATH','')
    compiler=sdk/'tools/cu-bridge/bin/cucc'
    version=subprocess.run([str(compiler),'--version'],capture_output=True,text=True,env=env,timeout=30)
    r['compiler']={'returncode':version.returncode,'stdout':version.stdout,'stderr':version.stderr}
    command=[str(compiler),'-std=c++17','-O3','-fPIC','-shared','-D__FAST_HALF_CVT__','-D__MERGE_LDS_B64']
    for include in [root/'csrc',root/'csrc/flash_attn',root/'csrc/flash_attn/src',sdk/'tools/cu-bridge/include',sdk/'include',sdk/'include/mcblas']:
        command+=['-I',str(include)]
    command+=[str(A/'vendor_fa_finite_p1_shared_mean.cu'),'-o',str(A/'libdeltatrace_fa_finite_shared_mean.so')]
    r['command']=command;save()
    with (A/'compile.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=A,timeout=p['max_compile_seconds'])
    r['compile_returncode']=result.returncode
    r['status']='finite_extension_compiled_not_executed' if result.returncode==0 else 'finite_extension_compile_failed'
    if result.returncode==0:
        lib=A/'libdeltatrace_fa_finite_shared_mean.so';r['library']={'bytes':lib.stat().st_size,'sha256':sha(lib)}
        for name,command in [('symbols',['nm','-D','--defined-only',str(lib)]),('dependencies',['ldd',str(lib)])]:
            result=subprocess.run(command,capture_output=True,text=True,env=env,timeout=30)
            r[name]={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
        assert 'deltatrace_fa_finite_p1_shared_mean' in r['symbols']['stdout']
except Exception:
    r['status']='build_probe_error';r['error']=traceback.format_exc()
finally:
    r['wall_seconds']=time.time()-start;r['vendor_sources_after']=source_receipt();save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','vendor_fa_finite_p1_shared_mean.cu','results.json','compile.log']:
            if (A/name).exists():z.write(A/name,name)
    print(json.dumps({'status':r['status'],'seconds':r['wall_seconds']}),flush=True)
