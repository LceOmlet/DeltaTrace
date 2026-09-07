import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
base=json.loads((A/'qwen35_fla_backward_protocol_20260908_v2.json').read_bytes())
keys=['checkpoint','verified_shard_sizes','weight_identity_receipt_sha256','checkpoint_config_tokenizer_sha256',
      'native_model_sha256','isolated_site','fla_mapped_utils_sha256','fla_scheduled_chunk_sha256',
      'wy_backported_sha256','compiler_cache','wheels','author_root','author_source_sha256','cache_sha256','native_stage_source_sha256']
p={k:base[k] for k in keys}
p.update(spans_directory='${ARTIFACT_ROOT}/codex_qwen35_official_spans_20260908_v1',
    spans_sha256=sha((A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_official_spans_20260908_v1/results.json').read_bytes()),
    source_tree_sha256='4aa4d44a2c31821298c4b4f563210a6a4ba8d3c41d6d2c9dd7e3565e386702e6',
    prefix_length=129,maximum_model_loads=1,maximum_root_forward_attempts=1,
    maximum_native_adjoint_stage_attempts=2,maximum_complete_backward_calls=0,maximum_quality_queries=0,
    workload='Real B4 [NI0 EOS, NI0 original, MH1 EOS, MH1 original], right padded. Capture first FLA layer129-token prefix and3 chunk boundary states.',
    native_reuse='Only chunk_bwd_dv_local and chunk_gated_delta_rule_bwd_dhu, on both real original prefixes in one B2. No full ordinary backward or replay forward.',
    stop='One capture and two native stage calls; preserve failures, no automatic retries, precision changes or quality sweep.',
    interpretation='Engineering data for finite coefficients. Arbitrary fixed local cotangent; no whole-model attribution, metric or speed claim.')
files={'study.py':(A/'qwen35_paired_fla_capture_20260908.py').read_bytes(),
       'official_fixed_text_inputs.py':(R/'research/runtime/official_fixed_text_inputs.py').read_bytes()}
for raw in files.values():ast.parse(raw)
p['files_sha256']={k:sha(v) for k,v in files.items()}
files['protocol.json']=json.dumps(p,indent=2).encode();(A/'qwen35_paired_fla_capture_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_paired_fla_capture_20260908_v1");d.mkdir(exist_ok=False);'
        'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
        'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
        '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_paired_fla_capture_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'study_sha256':p['files_sha256']['study.py'],'protocol_sha256':sha(files['protocol.json']),'limits':[1,1,2,0]}))
