"""Freeze a single59-answer-token/B2 integration pass, not a quality sweep."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
old=json.loads((A/'qwen35_decoder_integration_protocol_20260908.json').read_bytes())
p={k:v for k,v in old.items() if k not in ['purpose','budget','layers','workload','upstream','method','compiler','observations','validation','cost_scope','stop','next','files_sha256']}
gdn=json.loads((A/'qwen35_gdn_dtype_matched_protocol_20260908.json').read_bytes())
for key in ['author_root','author_source_sha256','cache_sha256','spans_directory','spans_sha256']:p[key]=gdn[key]
previous=json.loads((A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/results.json').read_bytes())
byname={x['file']:x for x in previous['artifacts']}
p.update(decoder_parent_directory='${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1',
    saved_decoders={str(i):byname[f'layer{i}_native_capture.pt'] for i in [0,3]},saved_LSE=byname['layer3_public_LSE.pt'],
    boundary_compiler_cache='${ARTIFACT_ROOT}/codex_qwen35_decoder_integration_20260908_v1/inductor_cache',
    purpose='Connect explicit author-answer logprob seed, final RMS and all32 original decoder finite pullbacks. Reuse root inputs and saved layer0/3 operands; one bounded integration pass.',
    target='Sum FP32 logsoftmax values for author remapped closed answer spans NI0[209,246] and MH1[119,139],38/21 tokens. Complete fixed trajectory remains context, full248320 vocabulary. Baseline replaces only original eligible user tokens by tokenizer EOS.',
    target_scope_change='Old8B P1 seeds used the whole fixed response. This9B integration explicitly uses the author answer sink; do not merge results with old P1 metrics or call target scopes equal. Finite rules unchanged. Official RISE/MAS evaluation remains a separate original function.',
    budget={'meta_model_constructions':1,'full_model_loads':0,'root_forwards':0,'decoder_weight_loads':32,
        'original_decoder_replays':30,'original_saved_decoder_reuses':2,'original_head_weight_loads':1,'original_norm_weight_loads':1,
        'original_head_norm_forward':1,'original_head_norm_backward':1,'finite_answer_seed_calls':2,'finite_final_norm_calls':2,
        'complete_finite_decoder_calls':32,'finite_FA_calls':8,'finite_FA_kernel_launches':24,'finite_FLA_calls':24,
        'new_public_auxiliary_FA_forwards':7,'saved_public_LSE_reuses':1,'native_auxiliary_linear_conv_forwards':24,
        'native_auxiliary_linear_conv_backwards':24,'whole_model_attributions':1,'generation_calls':0,'quality_queries':0},
    batching='Real605/368 right-padded paired B4 original modules and B2 finite propagation.59 selected answer rows packed together for the original independent Linear head; no padded target labels.',
    implementation='Same validated complete decoder rules. Native original class/weights/BF16 load, actual default FA and actual FLA. GPU passive captures consumed/released per layer. No shadow forward, globalT-square arrays, operator replacements, extra precision/tile sweeps or model generation.',
    seed_validation='One actual seed and one equal endpoint head+norm finite limit versus one native head/norm backward. Compare packed head logprobs to saved original full-root logprobs, recording GEMM row-shape/rounding effects. No bitwise gate.',
    ledger='Per-layer replay output versus original next checkpoint, upstream-paired replay discontinuity, local finite allocation and final signed token sum. Preserve all unassigned residuals without score correction. Save each layer input coefficient for recovery.',
    cost_scope='Engineering run with partial weight loads, passive captures, source checks, saved operand IO, compiler cold/default tuning, head native-gradient diagnostic, per-layer FP64 diagnostics. Report each stage. Not warm whole-model/FT performance. Original prior root cost is not free.',
    stop='One pass, no automatic retry on failure. Stop on source/shape/nonfinite/padding error; diagnose saved boundary without rerunning prior completed layers. No new full root, precision scan, generation or quality sweep.',
    next='Analyze saved full-chain residual and measured components. If integration is usable, freeze one actual-path profiler/cost screen and a small original-metric comparison with same-model FT; do not declare FA exhausted from local times.')
files={'study.py':(A/'qwen35_whole_finite_20260908.py').read_bytes()}
for name in ['qwen35_answer_finite.py','qwen35_decoder_finite.py','native_attention_capture.py','qwen35_gdn_finite.py','finite_fla_gpu.py','vendor_fa_finite_bf16_d256.py','official_fixed_text_inputs.py']:
    files[name]=(R/'research/runtime'/name).read_bytes()
for name in ['signed_secant_rules.py','compiled_swiglu_secant.py','compiled_finite_rules.py','compiled_logprob_seed.py']:files[name]=(R/'core'/name).read_bytes()
for raw in files.values():ast.parse(raw)
p['files_sha256']={k:sha(v) for k,v in files.items()};files['protocol.json']=json.dumps(p,indent=2).encode()
(A/'qwen35_whole_finite_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_whole_finite_20260908_v1");d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_whole_finite_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}),encoding='utf-8')
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'source_sha256':p['files_sha256'],'budget':p['budget']}))
