"""Freeze a single real-operand BF16/D256 FA screen, using existing root states."""
import ast,base64,hashlib,json,shlex,zipfile,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
build=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_finite_fa_build_20260908_v2'
with zipfile.ZipFile(build/'review_bundle.zip') as z:
    assert z.testzip() is None
    for name in z.namelist():
        assert Path(name).name==name;(build/name).write_bytes(z.read(name))
b=json.loads((build/'results.json').read_bytes());assert b['status']=='finite_extension_compiled_not_executed'
assert b['vendor_sources_before']==b['vendor_sources_after']
old=json.loads((A/'qwen35_gdn_dtype_matched_protocol_20260908.json').read_bytes())
keys=['checkpoint','verified_shard_sizes','weight_identity_receipt_sha256','checkpoint_config_tokenizer_sha256',
      'native_model_sha256','isolated_site','fla_mapped_utils_sha256','fla_scheduled_chunk_sha256',
      'wy_backported_sha256','compiler_cache','wheels','source_tree_sha256','parent_directory','parent_sha256']
p={k:old[k] for k in keys}
parent=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_gdn_integration_20260908_v2'
pr=json.loads((parent/'results.json').read_bytes());assert sha((parent/'results.json').read_bytes())==p['parent_sha256']
check=next(a for a in pr['artifacts'] if a['file']=='native_root_checkpoints.pt')
p.update(root_checkpoints_sha256=check['sha256'],build_directory='${ARTIFACT_ROOT}/codex_qwen35_finite_fa_build_20260908_v2',
    build_sha256=sha((build/'results.json').read_bytes()),library_sha256=b['library']['sha256'],
    purpose='Preserve current content_P1 strength while extending the same traceable finite FA framework to the real Qwen3.5 BF16/D256/GQA4 and605/368-token right-padded workload.',
    source_boundaries='Original model class/default FA forward untouched. Public FA varlen auxiliary output/LSE and native autograd backward only. No probability-buffer decoding, private FA binding, vendor header edit or shadow model.',
    budget={'full_model_loads':0,'full_model_forwards':0,'meta_model_constructions':1,'decoder3_weight_loads':1,
            'decoder3_forward_attempts':1,'auxiliary_public_FA_forward_attempts':1,'public_FA_backward_attempts':1,
            'finite_calls':3,'finite_kernel_launches':9,'generation_calls':0,'quality_queries':0},
    finite_calls=['actual cold','actual warm','equal-endpoint limit'],
    finite_rule='content_P1 PV, symmetric Q/K interaction, unchanged logarithmic-mean softmax rule. BF16 matrix inputs/weights/emitted multipliers; FP32 accumulations/nonlinear reductions. No global N*N buffers.',
    capture='Passive original decoder3 attention capture from saved real root input. Official meta-init/materialization policy and original mask/rotary helpers. Compare decoder output to saved root decoder4 input.',
    local_upstream='Actual input-endpoint native attention output, padding zero. Not an answer-logprob target or a quality attribution.',
    validation='Retain all coefficients; compare actual effects by head and sample, equal-endpoint limit to actual public FA native backward, padded outputs and public auxiliary output to unchanged default model FA.',
    cost_scope='One cold/one warm finite-only call; auxiliary LSE and genuine native backward separately reported. This is not whole-model attribution cost or a matched comparison to FT.',
    stop='Stop after these three finite calls. No tile sweep, precision scan, new root/quality pass or automatic rerun on numerical disagreement. Incomplete decoder nonlinear/MLP/final objective propagation stays explicitly pending.')
files={'study.py':(A/'qwen35_finite_fa_actual_20260908.py').read_bytes(),
       'vendor_fa_finite_bf16_d256.py':(R/'research/runtime/vendor_fa_finite_bf16_d256.py').read_bytes(),
       'native_attention_capture.py':(R/'research/runtime/native_attention_capture.py').read_bytes(),
       'vendor_fa_finite_p1_bf16_d256.cu':(R/'research/prototypes/vendor_fa_finite_p1_bf16_d256.cu').read_bytes()}
assert sha(files['vendor_fa_finite_p1_bf16_d256.cu'])==b['protocol']['extension_sha256']
for name,raw in files.items():
    if name.endswith('.py'):ast.parse(raw)
p['files_sha256']={k:sha(v) for k,v in files.items()};files['protocol.json']=json.dumps(p,indent=2).encode()
(A/'qwen35_finite_fa_actual_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_finite_fa_actual_20260908_v1");d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_finite_fa_actual_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}),encoding='utf-8')
print(json.dumps({'build_library':b['library'],'protocol_sha256':sha(files['protocol.json']),'files_sha256':p['files_sha256']}))
