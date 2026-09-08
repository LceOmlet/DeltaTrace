"""Freeze one corrected FT0-3 pass on two existing author trajectories."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
old=json.loads((A/'qwen35_FT_native_content_protocol_20260908.json').read_bytes())
keys=['checkpoint','verified_shard_sizes','checkpoint_config_tokenizer_sha256','native_model_sha256','isolated_site',
    'compiler_cache','source_tree_sha256','parent_directory','parent_sha256','fla_mapped_utils_sha256','fla_scheduled_chunk_sha256',
    'wy_backported_sha256','wheels','native_stage_source_sha256','installed_FA_interface_sha256','decoder_parent_directory',
    'saved_decoders','spans_directory','spans_sha256','baseline']
p={k:old[k] for k in keys};whole=json.loads((A/'qwen35_whole_finite_protocol_20260908.json').read_bytes())
p['root_checkpoints_sha256']=whole['root_checkpoints_sha256']
p.update(finite32_directory='${ARTIFACT_ROOT}/codex_qwen35_whole_finite_20260908_v1',
    finite32_sha256='9ee732b66cabcf59eb1d312fb222ab21407cd381c0ebb52904d4bc3710dbc79b',
    purpose='Complete explicitly corrected same-checkpoint FT32-layer token aggregation and unchanged author0-3-hop recurrence on existing NI0/MH1 fixed trajectories. No new model, dataset or candidate selection.',
    previous_boundary='Native content boundary v2 completed and CPU audited, source c7a457e44d50fe0d2f5257d3c481f229683eefdf54944b83875d6e88f7ff4444 unchanged. No repeat local precision screen.',
    budget={'meta_model_constructions':1,'full_model_loads':0,'root_forwards':0,'decoder_loads':32,'decoder_replays':30,
        'decoder_capture_reuses':2,'cached_parameter_reuses':96,'new_saved_minimal_layer_inputs':32,
        'public_FA_auxiliary_forwards':32,'public_FA_value_backwards':32,'public_FLA_auxiliary_forwards':96,
        'public_FLA_value_backwards':96,'public_linear_conv_forwards':96,'public_linear_conv_backwards':96,
        'content_calls':128,'aggregate_calls':128,'whole_FT_attributions':4,'generation_calls':0,'quality_queries':0,
        'CPU_author_hop_controller_replays':2},
    target='Author sink representations: NI0 absolute[566,603],MH1[346,366]. Thinking[357,565]/[227,345]. Official default observation mask and cumulative SUM, four fixed hops0-3, renorm_threshold0.',
    same_model='Same original Qwen3.5-9B checkpoint/BF16/default FA and actual FLA as saved DeltaTrace32. Reuse same root layer inputs and paired B4 original replays; only original rows enter true B2 FT attribution.',
    hop_rules='Match pinned e81b3be compute_multi_hop_ifr: native BF16 thinking weights and their normalization, full true-length token total denominator, multiplicative thinking ratios and observation SUM. Final eligible projection separate. CPU reference executes the unmodified author controller over saved aggregates, not a replacement model or attention.',
    cache='One original decoder replay per missing capture at hop0. Preserve minimal actual original B2 operands and consumed original parameter tensors on CPU and disk with SHA; reuse for later hops, explicitly timed host-to-device transfers. Full native captures discarded per layer. No globalT-square or identity-valued probes.',
    implementation='Unchanged validated native-content helper and author proximity; same BF16 vendor GEMM with32-token streaming. New code only orchestrates32 layers and author hop recurrence. Passive native observers, no model/operator/backward replacement.',
    audit='Source tree, parameter bytes versus previous32-layer load, actual original forward and native backward node counts. Each layer/hop finite/padding and weighted reconstruction ledger. Persist all per-layer token scores, hop weights/ratios/totals and accumulated observations. Original controller reference verifies normalization/ratio semantics without recomputing attributions.',
    cost_scope='Engineering integration includes original decoder weight load/replay/captures, native auxiliary forwards and discarded gradients, cold/default native tuning, CPU cache and archive plus transfers, per-head/O reconstruction diagnostics. Four hop outputs share one operand acquisition and previous-hop computations. Not warm production/fair FT versus DeltaTrace cost.',
    stop='One four-hop pass. Preserve failure and completed hop vectors; do not automatically rerun completed layers/hops, change precision or add quality samples. No generation/deletion evaluation in this execution budget.',
    next='If complete, inspect immutable FT0-3 versus existing DeltaTrace scores on the same eligible original tokens, freeze one small original-metric evaluation budget. Same-model quality remains unproven until original needle/RISE/MAS measured; keep corrected and unchanged FT identities separate.')
files={'study.py':(A/'qwen35_FT32_20260908.py').read_bytes(),
    'official_ft_core.py':(R/'research/third_party/flashtrace_qwen35_e81b3be/flashtrace/core.py').read_bytes()}
for f in ['qwen35_ft_hops.py','qwen35_ft_native_content.py','finite_fla_gpu.py','native_attention_capture.py','qwen35_gdn_finite.py','qwen35_decoder_finite.py']:
    files[f]=(R/'research/runtime'/f).read_bytes()
for f in ['signed_secant_rules.py','compiled_swiglu_secant.py']:files[f]=(R/'core'/f).read_bytes()
assert sha(files['qwen35_ft_native_content.py'])=='c7a457e44d50fe0d2f5257d3c481f229683eefdf54944b83875d6e88f7ff4444'
for source in files.values():ast.parse(source)
p['files_sha256']={k:sha(v) for k,v in files.items()};files['protocol.json']=json.dumps(p,indent=2).encode()
(A/'qwen35_FT32_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_FT32_20260908_v1");d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_FT32_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}),encoding='utf-8')
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'source_sha256':p['files_sha256'],'budget':p['budget']}))
