"""One official-Inductor fusion screen motivated by the actual145-kernel profile."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
prior=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_finite_gpu_20260908_v1'
source=(A/'qwen35_finite_gpu_20260908.py').read_text(encoding='utf-8')
source=source.replace("os.environ['TRITON_CACHE_DIR']=p['compiler_cache']", "os.environ['TRITON_CACHE_DIR']=p['compiler_cache']\nos.environ['TORCHINDUCTOR_CACHE_DIR']=str(HERE/'inductor_cache')")
source=source.replace("'calls':[],'artifacts':[]}","'compiled_pullback_attempts':0,'calls':[],'artifacts':[]}")
line='    chunk=verify_native_sources(p[\'native_stage_source_sha256\'])'
assert source.count(line)==1
source=source.replace(line,line+"\n    assert sha((Path(p['prior_GPU_directory'])/'results.json').read_bytes())==p['prior_GPU_sha256']\n    from finite_fla_gpu import make_compiled_finite_pullback\n    compiled_pullback=make_compiled_finite_pullback()")
source=source.replace("'native':'native_backward_attempts'}[kind]", "'native':'native_backward_attempts','compiled':'compiled_pullback_attempts'}[kind]")
source=source.replace("ceiling={'cached':1,'finite':5,'native':5}[kind]", "ceiling={'cached':0,'finite':5,'native':4,'compiled':5}[kind]")
source=source.replace("                else:out=native_backward()", "                elif kind=='compiled':out=compiled_pullback(endpoints,do,scale)\n                else:out=native_backward()")
start=source.index("    for kind in ['cached','finite','native']:")
stop=source.index("except Exception:\n    sys.setprofile(None)",start)
source=source[:start]+'''    for kind in ['compiled','finite','native']:
        out=run(kind,'cold');persist(kind+'_coefficients',out);del out
    for iteration,order in enumerate([['compiled','finite','native'],['native','finite','compiled'],['finite','compiled','native']]):
        for kind in order:
            out=run(kind,'formal_'+str(iteration))
            assert all(torch.isfinite(v).all() for v in out.values());del out
    for kind in ['compiled','finite']:
        out=run(kind,'diagnostic',profile=True);del out
    r['sources_after']=source_receipt();assert r['sources_before']==r['sources_after']
    assert r['compiled_pullback_attempts']==r['finite_pullback_attempts']==5 and r['native_backward_attempts']==4
    from torch._dynamo.utils import counters
    r['compiler_counters']={str(k):dict(v) for k,v in counters.items()}
    r['compiler_generated_sources']=[]
    for path in (HERE/'inductor_cache').rglob('*.py'):
        raw=path.read_bytes();r['compiler_generated_sources'].append({'file':str(path.relative_to(HERE)),'sha256':sha(raw),'bytes':len(raw)})
    r['status']='official_compiler_fused_mixed_finite_screen_complete'
''' +source[stop:]
source=source.replace("'attempts':[r['cached_adjoint_mixed_attempts'],r['finite_pullback_attempts'],r['native_backward_attempts']]",
                      "'attempts':[r['compiled_pullback_attempts'],r['finite_pullback_attempts'],r['native_backward_attempts']]")
source=source.replace("+[x.name for x in HERE.glob('*.npz')]:z.write(HERE/name,name)",
    "+[x.name for x in HERE.glob('*.npz')]:z.write(HERE/name,name)\n        for path in (HERE/'inductor_cache').rglob('*.py'):z.write(path,str(path.relative_to(HERE)))")
ast.parse(source);study=A/'qwen35_finite_gpu_compiled_20260908.py';study.write_text(source,encoding='utf-8')
p=json.loads((A/'qwen35_finite_gpu_protocol_20260908.json').read_bytes())
for key in ['maximum_cached_adjoint_mixed_calls','maximum_complete_local_finite_calls','maximum_native_backward_helper_calls',
            'maximum_native_dv_and_state_adjoint_calls','maximum_mixed_coefficient_calls']:p.pop(key)
p.update(prior_GPU_directory='${ARTIFACT_ROOT}/codex_qwen35_finite_gpu_20260908_v1',prior_GPU_sha256=sha((prior/'results.json').read_bytes()),
    maximum_compiled_finite_calls=5,maximum_eager_finite_calls=5,maximum_native_backward_helper_calls=4,
    maximum_native_dv_and_state_adjoint_calls=28,maximum_mixed_coefficient_calls=10,
    purpose='Fuse measured layout/conversion/elementwise overhead with installed torch.compile/Inductor. Same finite algebra and native FLA functions, no custom backend/meta patch or substitute model.',
    timing='One cold per route (compiled first), three rotated matched compiled/eager/native triples, one separately profiled compiled/eager pair. No profiler in formal timings. All native stages and packing included.',
    compiler='fullgraph=True, dynamic=False, default backend, max_autotune=False, triton.cudagraphs=False. Generated sources and compiler counters retained.',
    stop='One compiled variant only. Preserve any compile failure without fallback/retry; do not upgrade or patch vendor runtime. No model or quality runs.')
files={'study.py':study.read_bytes(),'finite_fla_gpu.py':(R/'research/runtime/finite_fla_gpu.py').read_bytes()}
p['files_sha256']={k:sha(v) for k,v in files.items()}
files['protocol.json']=json.dumps(p,indent=2).encode();(A/'qwen35_finite_gpu_compiled_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_finite_gpu_compiled_20260908_v1");d.mkdir(exist_ok=False);'
        'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
        'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
        '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_finite_gpu_compiled_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'source_sha256':p['files_sha256']}))
