"""Freeze one bounded full-decoder screen for each original mixer family."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
old=json.loads((A/'qwen35_finite_fa_saved_protocol_20260908.json').read_bytes())
keys=['checkpoint','verified_shard_sizes','checkpoint_config_tokenizer_sha256','native_model_sha256','isolated_site',
      'fla_mapped_utils_sha256','fla_scheduled_chunk_sha256','wy_backported_sha256','compiler_cache','wheels',
      'source_tree_sha256','parent_directory','parent_sha256','root_checkpoints_sha256','build_directory','library_sha256']
p={k:old[k] for k in keys}
gdn=json.loads((A/'qwen35_gdn_dtype_matched_protocol_20260908.json').read_bytes())
p.update(native_stage_source_sha256=gdn['native_stage_source_sha256'],
    installed_FA_interface_sha256=sha((A/'snapshot${SITE_PACKAGES}/flash_attn/flash_attn_interface.py').read_bytes()),
    purpose='Complete standard attention and both decoder families while retaining the strong finite rules and actual model backends. Reuse saved original root layer0/3 inputs; no new full root.',
    budget={'full_model_loads':0,'full_model_forwards':0,'meta_model_constructions':1,'original_decoder_loads':2,
        'original_decoder_forward_attempts':2,'original_decoder_backward_attempts':2,'public_FA_auxiliary_forward_attempts':1,
        'complete_finite_decoder_calls':6,'FA_finite_calls':3,'FA_finite_kernel_launches':9,'FLA_finite_calls':3,
        'native_auxiliary_linear_conv_forwards':3,'native_auxiliary_linear_conv_backwards':3,
        'generation_calls':0,'quality_queries':0},
    layers=[3,0],workload='Original official NI0/MH1 paired EOS/input trajectories, B4 native and B2 finite,605/368 right padding.',
    upstream='Actual input-endpoint decoder output, valid positions only. Local engineering pairing; not the answer objective.',
    method='Unchanged content_P1 FA core and content1 GDN recurrence; output sigmoid gate uses content1. Established symmetric MLP product/RMS finite rules. Native Q/K 1+weight normalization,partial64 RoPE,and head-interleaved Q/gate layout.',
    compiler='Four composite finite-only graphs via official torch.compile/fullgraph/static; native BF16 bmm retained. max_autotune=False,cudagraphs=False; default tuning may still occur and cold costs are reported.',
    observations='Passive module hooks and existing native mixer observers. Record native autograd-node callbacks instead of a main-thread backward-count assertion. Save raw captures before diagnostics.',
    validation='One cold actual/diagnostic, one normal warm actual, one equal-endpoint finite call per decoder. Compare original replay to saved root, captured product/residual boundaries, all finite input coefficients to native gradient at equal endpoints, per-boundary finite effects and padding.',
    cost_scope='Complete single decoder finite-only cost; excludes root/replay and auxiliary-LSE preparation. Compiler/diagnostic cold call and normal warm call separated. No whole-model/FT cost claim.',
    stop='Six complete finite calls total. Stop on compile/runtime or numerical concerns; do not repeat a successful core screen, run precision/tile sweeps, new root, generation or quality jobs. Preserve failures and saved operands for diagnosis.',
    next='After both decoder families are checked, connect final normalization/logprob target seed and reverse all32 original layers. Profile the actual complete path before selecting one justified FA or other major-cost optimization.')
files={'study.py':(A/'qwen35_decoder_integration_20260908.py').read_bytes()}
for name in ['qwen35_decoder_finite.py','native_attention_capture.py','qwen35_gdn_finite.py','finite_fla_gpu.py','vendor_fa_finite_bf16_d256.py']:
    files[name]=(R/'research/runtime'/name).read_bytes()
for name in ['signed_secant_rules.py','compiled_swiglu_secant.py']:files[name]=(R/'core'/name).read_bytes()
for name,raw in files.items():ast.parse(raw)
p['files_sha256']={k:sha(v) for k,v in files.items()};files['protocol.json']=json.dumps(p,indent=2).encode()
(A/'qwen35_decoder_integration_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1");d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_decoder_integration_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}),encoding='utf-8')
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'files_sha256':p['files_sha256']}))
