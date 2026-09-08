"""Freeze one saved real B2 native FT content-adjoint screen, no quality sweep."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
old=json.loads((A/'qwen35_whole_finite_protocol_20260908.json').read_bytes())
keys=['checkpoint','verified_shard_sizes','checkpoint_config_tokenizer_sha256','native_model_sha256',
    'isolated_site','compiler_cache','source_tree_sha256','parent_directory','parent_sha256',
    'fla_mapped_utils_sha256','fla_scheduled_chunk_sha256','wy_backported_sha256','wheels',
    'native_stage_source_sha256','installed_FA_interface_sha256','decoder_parent_directory','saved_decoders','saved_LSE',
    'author_root','author_source_sha256','cache_sha256','spans_directory','spans_sha256']
p={k:old[k] for k in keys}
previous=json.loads((A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/results.json').read_bytes())
names=['3.self_attn.o_proj.weight','3.input_layernorm.weight','0.linear_attn.out_proj.weight','0.linear_attn.norm.weight','0.linear_attn.conv1d.weight']
p['decoder_parameter_receipts']={('model.language_model.layers.'+n):previous['weight_tensor_receipts']['model.language_model.layers.'+n] for n in names}
ft=json.loads((A/'qwen35_FT_variant_audit_20260908.json').read_bytes())
core=R/'research/third_party/flashtrace_qwen35_e81b3be/flashtrace/core.py'
assert sha(core.read_bytes())==next(x['sha256'] for x in ft['receipts'] if x['file']=='flashtrace/core.py')
p.update(purpose='Validate explicitly corrected FT content decomposition against saved original layer3 FA and layer0 GDN outputs, real author605/368 B2. Necessary comparator preparation, not DeltaTrace or complete FT.',
    baseline={'upstream_commit':ft['commit'],'identity':'Corrected native-content FT prototype, not unchanged author implementation.',
        'unchanged':'Pinned author proximity and normalization with native BF16 source/head numerators; no new relevance formula or quality score tuning.',
        'corrections':'Consume actual value projection, include full-attention sigmoid gate and GDN frozen RMS/SiLU output gain, assign GDN value convolution to original preconv positions.',
        'rounding':'Gains and components in FP32, native gradients BF16, projected operands BF16/native output rounding. GDN linear-conv frozen sigmoid decomposition differs from fused-SiLU rounding; measure it.'},
    target='Author FT sink representation sum at absolute closed NI0[566,603]/MH1[346,366],38/21 tokens. Same answer scope as DeltaTrace; different internal target (representation versus logprob).',
    budget={'meta_model_constructions':1,'parameter_loads':5,'full_model_loads':0,'root_forwards':0,'decoder_replays':0,
        'saved_decoder_reuses':2,'public_FA_auxiliary_forwards':1,'public_FA_value_backwards':1,
        'public_FLA_auxiliary_forwards':1,'public_FLA_value_backwards':1,
        'public_linear_conv_forwards':1,'public_linear_conv_backwards':1,
        'FT_projection_and_proximity_calls':2,'extra_fixed_CPU_review_projections':2,
        'whole_FT_attributions':0,'generation_calls':0,'quality_queries':0},
    batching='Original endpoint rows1/3 of saved paired B4, true B2. Native FA public varlen pack for605/368; GDN native right-padded B2 with zero sink seeds after true length.',
    auxiliary_semantics='Freeze actual q/k/g/beta and output gains. Public native value backward yields linear content contraction, not full model derivative. FA expands compact KV to all query heads only in explicit auxiliary to preserve head/O information; no model replacement. Public conv activation=None backward distributes frozen SiLU content to preconv positions.',
    storage='No globalT-square or identity value probes; source components B*T*H*D. Projected hidden contributions streamed in32-source chunks. Diagnostics include FP64 sums and saved private full components.',
    checks='Native autograd node execution, source and parameter identities, finite/padding, original vs auxiliary core, missing/output gains, per-head and projected reconstruction. Independent CPU FP64 math reference of dV for original operands, native conv transpose and fixed head/row projection. CPU reference is diagnostic only, never used as runtime model or attribution backend.',
    cost_scope='Single engineering call per operator includes first native compile/default autotuning, additional forwards, discarded native q/k/g/beta/conv parameter gradients, all source checks and saved capture IO. Projection cost includes diagnostics. Final ZIP/download reported separately. No cold/warm speed ratio or full FT speed claim.',
    stop='One pass, no automatic rerun. On execution/source/nonfinite/padding failure preserve results and diagnose saved evidence. No precision scan, root replay, whole FT, generation or quality sweep in this budget.',
    next='If native content boundaries are usable, close this local screen and integrate official0-3-hop aggregation across32 layers, then freeze small original-metric same-checkpoint comparison. Do not call boundary reconstruction a quality win.')
p['revision']='v2 protocol-only correction: v1 omitted four source-guard fields and stopped before torch import, with zero parameter loads or GPU calls. Preserve v1 unchanged; same runtime/study and operator budget.'
files={'study.py':(A/'qwen35_FT_native_content_20260908.py').read_bytes(),
    'qwen35_ft_native_content.py':(R/'research/runtime/qwen35_ft_native_content.py').read_bytes(),
    'finite_fla_gpu.py':(R/'research/runtime/finite_fla_gpu.py').read_bytes(),'official_ft_core.py':core.read_bytes()}
for raw in files.values():ast.parse(raw)
p['files_sha256']={k:sha(v) for k,v in files.items()};files['protocol.json']=json.dumps(p,indent=2).encode()
(A/'qwen35_FT_native_content_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_FT_native_content_20260908_v2");d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_FT_native_content_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}),encoding='utf-8')
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'source_sha256':p['files_sha256'],'budget':p['budget']}))
