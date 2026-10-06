"""Compile only the isolated owner candidate with the recorded cucc flags.

No package install, cache clearing, model, GPU launch or production overwrite.
The original prepared build owns the compiler/header setup; source and output
paths point only to this explicitly unaccepted candidate.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('entry', AUDIT/'stage_environment_entry.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
OUT = entry.ROOT+'/candidates/appworld-row-cuts-finite-20261007-v1'


def main():
    sources = {name: hashlib.sha256((HERE/'candidate'/name).read_bytes()).hexdigest()
               for name in ('vendor_fa_finite_p1_bf16_d256.cu',
                            'vendor_fa_finite_bf16_d256.py')}
    # Preserve an already-built attempt and reuse only its exact source bytes.
    init = ("import pathlib,hashlib; p=pathlib.Path(%r); p.mkdir(exist_ok=True); "
            "expected=%r; assert all(not (p/n).exists() or hashlib.sha256((p/n).read_bytes()).hexdigest()==s "
            "for n,s in expected.items()),'Preserve different existing candidate bytes'; print(p)") % (OUT,sources)
    subprocess.run(entry.SSH+['/opt/conda/bin/python','-'],input=init.encode(),check=True)
    for name in sources:
        subprocess.run(entry.SCP+[str(HERE/'candidate'/name),entry.SSH[-1]+':'+OUT+'/'+name],check=True)
    code = r'''
import pathlib,os,subprocess,time,json,hashlib,psutil,signal
out=pathlib.Path(@OUT@)
sources=@SOURCES@
for name,sha in sources.items():
    assert hashlib.sha256((out/name).read_bytes()).hexdigest()==sha
result=out/'build.json'
if result.exists():
    old=json.loads(result.read_text())
    assert old['source_sha256']==sources and old['returncode']==0
    assert hashlib.sha256((out/'libfinite_row_query_starts.so').read_bytes()).hexdigest()==old['library_sha256']
    print(json.dumps(old));raise SystemExit(0)
sdk=pathlib.Path('/opt/maca')
vendor=pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/repo/third_party/metax_fa_2_5_3')
prepared=pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/downloads/prepare_remote.py')
env=dict(os.environ,MACA_PATH=str(sdk),CUDA_PATH=str(sdk/'tools/cu-bridge'),MACA_CLANG_PATH=str(sdk/'mxgpu_llvm/bin'))
env['PATH']=':'.join([str(sdk/'tools/cu-bridge/bin'),str(sdk/'mxgpu_llvm/bin'),'/opt/conda/bin','/usr/bin','/bin',env.get('PATH','')])
env['LD_LIBRARY_PATH']=':'.join([str(sdk/'lib'),str(sdk/'mxgpu_llvm/lib'),env.get('LD_LIBRARY_PATH','')])
compiler=str(sdk/'tools/cu-bridge/bin/cucc')
version=subprocess.run([compiler,'--version'],env=env,capture_output=True,text=True,check=True)
cmd=[compiler,'-std=c++17','-O3','-fPIC','-shared','-D__FAST_HALF_CVT__','-D__MERGE_LDS_B64']
for include in [vendor/'csrc',vendor/'csrc/flash_attn',vendor/'csrc/flash_attn/src',sdk/'tools/cu-bridge/include',sdk/'include',sdk/'include/mcblas']:
    assert include.is_dir();cmd+=['-I',str(include)]
cmd+=[str(out/'vendor_fa_finite_p1_bf16_d256.cu'),'-o',str(out/'libfinite_row_query_starts.so')]
start=time.time()
before=psutil.virtual_memory().available
peak_pss=0;peak_processes=[];timed_out=False
with (out/'compile.log').open('wb') as log:
    p=subprocess.Popen(cmd,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    birth=psutil.Process(p.pid).create_time()
    while p.poll() is None:
        processes=[]
        try:
            parent=psutil.Process(p.pid)
            for child in [parent,*parent.children(recursive=True)]:
                try:processes.append(dict(pid=child.pid,pss_bytes=child.memory_full_info().pss))
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        except psutil.NoSuchProcess:pass
        total=sum(x['pss_bytes'] for x in processes)
        if total>peak_pss:peak_pss=total;peak_processes=processes
        if time.time()-start>90:
            timed_out=True
            assert psutil.Process(p.pid).create_time()==birth and os.getpgid(p.pid)==p.pid
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
            break
        time.sleep(.2)
report=dict(status='unaccepted_isolated_compilation_only',started_unix=start,finished_unix=time.time(),returncode=p.returncode,
    command=cmd,source_sha256=sources,compiler_version=version.stdout+version.stderr,
    original_build_source=dict(path=str(prepared),sha256=hashlib.sha256(prepared.read_bytes()).hexdigest(),lines=[93,94,95,96,97,98,99,101,102,103,104,105]),
    headers=dict(path=str(vendor/'UPSTREAM.json'),sha256=hashlib.sha256((vendor/'UPSTREAM.json').read_bytes()).hexdigest()),
    compile_resources=dict(sampled_peak_tree_PSS_bytes=peak_pss,peak_processes=peak_processes,sampling_seconds=.2,timed_out=timed_out),available_host_before=before,available_host_after=psutil.virtual_memory().available,
    GPU_kernel_launches=0,model_loads=0,checkpoint_loads=0,production_changes=0)
if p.returncode==0:
    report['library_sha256']=hashlib.sha256((out/'libfinite_row_query_starts.so').read_bytes()).hexdigest()
report['compile_log_sha256']=hashlib.sha256((out/'compile.log').read_bytes()).hexdigest()
result.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
'''.replace('@SOURCES@',repr(sources)).replace('@OUT@',repr(OUT))
    compile(code, '<isolated owner compile script>', 'exec')
    script='source '+entry.ENTRY+'/metax-entry.env.sh\n/opt/conda/bin/python - <<\'PY\'\n'+code+'\nPY\n'
    run=subprocess.run(entry.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=110)
    stderr_path=HERE/'compile-owner.stderr.txt'
    if stderr_path.exists():
        previous=stderr_path.read_bytes()
        previous_path=HERE/('compile-owner.stderr-'+hashlib.sha256(previous).hexdigest()[:12]+'.txt')
        if not previous_path.exists():previous_path.write_bytes(previous)
    stderr_path.write_bytes(run.stderr)
    run.check_returncode()
    report=json.loads(run.stdout)
    (HERE/'compiled-owner.json').write_bytes(run.stdout)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
