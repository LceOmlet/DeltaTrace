"""Freeze selective FT recovery after a demonstrated missing causal range guard."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
p=json.loads((A/'qwen35_FT32_protocol_20260908.json').read_bytes())
d=A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_FT32_20260908_v1';raw=(d/'results.json').read_bytes();old=json.loads(raw)
assert old['status']=='failed' and old['content_calls']==34
p.update(resume_parent='${ARTIFACT_ROOT}/codex_qwen35_FT32_20260908_v1',resume_parent_sha256=sha(raw),
    purpose='Resume from32 saved actual native layer inputs. Enforce original author j<sink_end+1 before proximity normalization. Recompute only FT0 layer31 with observed forbidden source score, reuse31 unchanged FT0 layers, then FT1-3. No model replay or checkpoint parameter reread.',
    previous_boundary='Prior local value-gradient boundary remains validated. FT32 first pass found zero-content future/padding relevance from rounded proximity without author source range. Original failure preserved; not a new precision rule or score correction.',
    budget={'meta_model_constructions':1,'full_model_loads':0,'root_forwards':0,'decoder_replays':0,'checkpoint_tensor_loads':0,
        'actual_layer_cache_loads':32,'cached_parameter_loads':97,'FT0_layer31_recomputations':1,'FT0_saved_layer_score_reuses':31,
        'public_FA_auxiliary_forwards':25,'public_FA_value_backwards':25,'public_FLA_auxiliary_forwards':72,'public_FLA_value_backwards':72,
        'public_linear_conv_forwards':72,'public_linear_conv_backwards':72,'content_calls':97,'aggregate_calls':97,
        'extra_unmasked_aggregation_diagnostics':2,'completed_hop_outputs':4,'CPU_author_hop_controller_replays':2,'generation_calls':0,'quality_queries':0},
    correction='Require explicit per-example source_limits:604/367 for answer,566/346 for thinking. Mask outside before head/token sums and denominator, as original FT only iterates J_max=sink_end+1. No change to allowed-source proximity formula. Native value helpers unchanged.',
    selective_recovery='Immutable FT0 scores show exactly one forbidden entry, layer31 NI position604,1.991398448808468e-06; remaining31 layer scores have zero forbidden mass and are reused byte-identically. Regenerate actual layer31 contribution and normalization; never rescale stored scores to force a result. Recompute downstream hop weights from corrected full FT0.',
    additional_paid_work='Original failed pass had34 content calls. This resume adds97;131 total versus intended128, with FT0 layer31 and FT1 layer0/failed1 repeated and explicitly charged. Zero decoder/root replay repeated. Two unmasked/bounded source-range observations share native components and GEMM outputs.',
    cache='Hash-check32 previously saved native minimal input/parameter PT files; parameters checked against original checkpoint load receipts. CPU cache reused for all hops, each GPU transfer timed. No globalT-square matrices.',
    stop='One selective resume. Save corrected FT0 and each completed later hop. Stop on source/nonfinite/forbidden contribution or score error and preserve evidence. No automatic full replay, precision sweep or quality evaluation.',
    cost_scope='Count both failed89.12-second job and current resume, original30 decoder replay costs, all auxiliary/backward/discarded gradients, IO, cache transfers, initial tuning and diagnostics; no full-method speed claim.')
files={'study.py':(A/'qwen35_FT32_resume_20260908.py').read_bytes(),
    'official_ft_core.py':(R/'research/third_party/flashtrace_qwen35_e81b3be/flashtrace/core.py').read_bytes()}
for name in ['qwen35_ft_hops.py','qwen35_ft_native_content.py','finite_fla_gpu.py']:files[name]=(R/'research/runtime'/name).read_bytes()
for source in files.values():ast.parse(source)
p['files_sha256']={k:sha(v) for k,v in files.items()};files['protocol.json']=json.dumps(p,indent=2).encode()
(A/'qwen35_FT32_resume_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_qwen35_FT32_resume_20260908_v1");d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_qwen35_FT32_resume_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}),encoding='utf-8')
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'source_sha256':p['files_sha256'],'budget':p['budget']}))
