"""One three-GEMM reuse candidate against the archived compiled implementation."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
prior=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_finite_gpu_compiled_20260908_v1'
source=(A/'qwen35_finite_gpu_compiled_20260908.py').read_text(encoding='utf-8')
source=source.replace('compiled_pullback=make_compiled_finite_pullback()',
    'compiled_pullback=make_compiled_finite_pullback(reuse_scalar_products=True)\n    from finite_fla_gpu_baseline import make_compiled_finite_pullback as baseline_factory\n    baseline_pullback=baseline_factory()')
source=source.replace("elif kind=='finite':out=finite_fla_pullback(endpoints,do,scale)","elif kind=='finite':out=baseline_pullback(endpoints,do,scale)")
source=source.replace("'official_compiler_fused_mixed_finite_screen_complete'","'three_GEMM_reuse_screen_complete'")
source=source.replace("['study.py','protocol.json','finite_fla_gpu.py','results.json']", "['study.py','protocol.json','finite_fla_gpu.py','finite_fla_gpu_baseline.py','results.json']")
ast.parse(source);study=A/'qwen35_finite_reuse_20260908.py';study.write_text(source,encoding='utf-8')
p=json.loads((A/'qwen35_finite_gpu_compiled_protocol_20260908.json').read_bytes())
p.update(prior_GPU_directory='${ARTIFACT_ROOT}/codex_qwen35_finite_gpu_compiled_20260908_v1',prior_GPU_sha256=sha((prior/'results.json').read_bytes()),
    route_labels={'compiled':'compiled_reuse_candidate','finite':'archived_compiled_baseline','native':'native_normalized_backward'},
    maximum_compiled_finite_calls=5,maximum_archived_compiled_baseline_calls=5,maximum_native_backward_helper_calls=4,
    purpose='Exactly one inner-product reuse candidate removes three GEMMs in scalar decay factors. Compare against the byte-preserved prior compiled source, same process/input/cotangent.',
    timing='One cold per route; three rotated candidate/baseline/native triples; one separate candidate/baseline profile. All packing and native adjoints included; no profiler in formal calls.',
    precision='Same BF16 bmm/FP32 result. Inner-product reassociation changes W rounding; compare full coefficients and per-head finite residuals. No new numerical pass threshold.',
    stop='One candidate only; no further local contraction variants or precision scan. Retain if actual cost and numeric evidence support it; continue nonlinear/model integration regardless.')
p.pop('maximum_eager_finite_calls')
files={'study.py':study.read_bytes(),'finite_fla_gpu.py':(R/'research/runtime/finite_fla_gpu.py').read_bytes(),
       'finite_fla_gpu_baseline.py':(prior/'finite_fla_gpu.py').read_bytes()}
p['files_sha256']={k:sha(v) for k,v in files.items()}
files['protocol.json']=json.dumps(p,indent=2).encode();(A/'qwen35_finite_reuse_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_finite_reuse_20260908_v1");d.mkdir(exist_ok=False);'
        'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
        'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
        '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_finite_reuse_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'source_sha256':p['files_sha256']}))
