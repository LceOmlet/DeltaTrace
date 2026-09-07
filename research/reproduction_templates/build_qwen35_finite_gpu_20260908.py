import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
parent=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_paired_fla_capture_20260908_v1';r=json.loads((parent/'results.json').read_bytes());base=r['protocol']
p={k:base[k] for k in ['isolated_site','wheels','source_tree_sha256','native_stage_source_sha256','compiler_cache']}
p.update(parent_directory='${ARTIFACT_ROOT}/codex_qwen35_paired_fla_capture_20260908_v1',parent_sha256=sha((parent/'results.json').read_bytes()),
    input_npz_sha256={x['file']:x['sha256'] for x in r['artifacts']},
    explicit_native_changes={'fla/utils.py':base['fla_mapped_utils_sha256'],
       'fla/ops/common/chunk_delta_h.py':base['fla_scheduled_chunk_sha256'],
       'fla/ops/gated_delta_rule/wy_fast.py':base['wy_backported_sha256']},
    maximum_model_loads=0,maximum_model_forwards=0,maximum_quality_queries=0,
    maximum_cached_adjoint_mixed_calls=1,maximum_complete_local_finite_calls=5,maximum_native_backward_helper_calls=5,
    maximum_native_dv_and_state_adjoint_calls=20,maximum_mixed_coefficient_calls=6,
    purpose='Actual GPU mixed finite implementation and matched local cost control on saved NI0/MH1 normalized operands; no ordinary-gradient parameter sweep.',
    timing='One cold per route, three alternated finite/native pairs, one separately profiled pair. Includes contiguous copies, packing and two native stages in finite API. No CUDA graphs, cached coefficients or profiler in formal timings.',
    cost_limit='Local pullback from already captured native intermediates. Endpoint forward/capture costs and whole-model performance are not measured here; native backward only needs endpoint1, finite needs both endpoints.',
    precision='Existing BF16 bmm with FP32 output plus FP32 elementwise/scan; preserve actual model FA/FLA. No new precision gate or bitwise requirement.',
    stop='One bounded implementation screen; preserve any compiler/runtime failure, no silent fallback or automatic retry. If overhead is large, profile the same call and optimize shared operators before model/quality expansion.')
files={'study.py':(A/'qwen35_finite_gpu_20260908.py').read_bytes(),'finite_fla_gpu.py':(R/'research/runtime/finite_fla_gpu.py').read_bytes()}
for raw in files.values():ast.parse(raw)
p['files_sha256']={k:sha(v) for k,v in files.items()}
files['protocol.json']=json.dumps(p,indent=2).encode();(A/'qwen35_finite_gpu_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_finite_gpu_20260908_v1");d.mkdir(exist_ok=False);'
        'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
        'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
        '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_finite_gpu_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'source_sha256':p['files_sha256']}))
