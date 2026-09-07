"""Compile the unchanged vendor backward head128 source. Never import the model."""
import hashlib,json,os,subprocess,time,traceback,zipfile
from pathlib import Path
A=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((A/'protocol.json').read_text());m=json.loads((A/'source_manifest.json').read_text())
assert sha(A/'study.py')==p['study_sha256']
assert sha(A/'source.zip')==p['archive_sha256'] and sha(A/'source_manifest.json')==p['source_manifest_sha256']
r={'status':'running','protocol':p,'model_forwards':0,'model_backwards':0,'attribution_calls':0,'GPU_kernel_launches':0,'source_files_verified':0}
def save():
    t=A/'results.partial';t.write_text(json.dumps(r,indent=2));t.replace(A/'results.json')
save();start=time.time()
try:
    root=A/'vendor_source';root.mkdir()
    with zipfile.ZipFile(A/'source.zip') as z:
        assert set(z.namelist())==set(m['files'])
        for name,meta in m['files'].items():
            assert not name.startswith('/') and '..' not in Path(name).parts
            raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==meta['sha256'] and len(raw)==meta['bytes']
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            r['source_files_verified']+=1
    sdk=Path(os.environ.get('MACA_PATH','/opt/maca'))
    env=os.environ.copy();env.update(MACA_PATH=str(sdk),CUDA_PATH=str(sdk/'tools/cu-bridge'),MACA_CLANG_PATH=str(sdk/'mxgpu_llvm/bin'))
    env['PATH']=str(sdk/'tools/cu-bridge/bin')+':'+str(sdk/'mxgpu_llvm/bin')+':'+env.get('PATH','')
    env['LD_LIBRARY_PATH']=str(sdk/'lib')+':'+str(sdk/'mxgpu_llvm/lib')+':'+env.get('LD_LIBRARY_PATH','')
    compiler=sdk/'tools/cu-bridge/bin/cucc'
    version=subprocess.run([str(compiler),'--version'],capture_output=True,text=True,env=env,timeout=30)
    r['compiler']={'path':str(compiler),'returncode':version.returncode,'stdout':version.stdout,'stderr':version.stderr}
    r['sdk_version']=(sdk/'Version.txt').read_text()
    for rel in ['include/mctlass/version.h','include/cute/config.hpp']:
        f=sdk/rel
        if f.exists():r.setdefault('sdk_header_sha256',{})[rel]=sha(f)
    source=root/'csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu'
    # Same byte-for-byte .cu -> .cpp staging used by vendor CMakeLists.txt.
    staged=A/'flash_bwd_hdim128_fp16_sm80.cpp';staged.write_bytes(source.read_bytes());assert sha(staged)==sha(source)
    command=[str(compiler),'-std=c++17','-O3','-fPIC','-DEXPORT_LIB','-D__FAST_HALF_CVT__','-D__MERGE_LDS_B64']
    for include in [root/'csrc',root/'csrc/flash_attn',root/'csrc/flash_attn/src',sdk/'tools/cu-bridge/include',sdk/'include',sdk/'include/mcblas']:
        command+=['-I',str(include)]
    command+=['-c',str(staged),'-o',str(A/'vendor_bwd128.o')]
    r['compile_command']=command;save()
    with (A/'compile.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=A,timeout=p['max_compile_wall_seconds'])
    r['compile_returncode']=result.returncode
    r['status']='compiled_unmodified_vendor_source' if result.returncode==0 else 'unmodified_vendor_compile_failed'
    if result.returncode==0:
        assert (A/'vendor_bwd128.o').stat().st_size>0
        r['object']={'sha256':sha(A/'vendor_bwd128.o'),'bytes':(A/'vendor_bwd128.o').stat().st_size}
except Exception as exc:
    r['status']='probe_error';r['error']=repr(exc);r['traceback']=traceback.format_exc()
finally:
    r['wall_seconds']=time.time()-start;save()
    r['source_files_after']={name:sha(A/'vendor_source'/name) for name in m['files'] if (A/'vendor_source'/name).exists()};save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','source_manifest.json','results.json','compile.log','flash_bwd_hdim128_fp16_sm80.cpp']:
            if (A/name).exists():z.write(A/name,name)
    print(json.dumps({'status':r['status'],'wall_seconds':r['wall_seconds'],'source_files_verified':r['source_files_verified']}),flush=True)
