"""Freeze unchanged finite rules on original FT's actual raw NI0 input."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
old=json.loads((A/'qwen35_target_regression_protocol_20260908.json').read_bytes())
ft_path=A/'snapshot${ARTIFACT_ROOT}/codex_official_ft_exp2_default_20260908_v1/results.json'
ft=json.loads(ft_path.read_bytes());assert ft['status']=='unchanged_official_FT_NI0_trace_completed'
contract_path=A/'snapshot/tmp/qwen35_official_input_contract_20260908.json'
contract=json.loads(contract_path.read_bytes());assert contract['current_keep_equals_historical'] and contract['current_gold_equals_historical']
files={'study.py':(R/'research/reproduction_templates/dt_official_input_NI0_20260908.py').read_bytes()}
for name,digest in old['unchanged_runtime_sha256'].items():
 if name=='official_fixed_text_inputs.py':continue
 path=R/('core' if name in ['signed_secant_rules.py','compiled_swiglu_secant.py','compiled_finite_rules.py','compiled_logprob_seed.py'] else 'research/runtime')/name
 raw=path.read_bytes();assert sha(raw)==digest,name;files[name]=raw
files['native_dense_attention_capture.py']=(R/'research/runtime/native_dense_attention_capture.py').read_bytes()
for raw in files.values():ast.parse(raw)
fp=ft['protocol'];p={k:old[k] for k in ['checkpoint','verified_shard_sizes','checkpoint_config_tokenizer_sha256','native_model_sha256',
                                     'isolated_site','compiler_cache','boundary_compiler_cache','installed_FA_interface_sha256','native_stage_source_sha256']}
p.update(official_root=fp['official_root'],official_package_blob_sha1=fp['package_blob_sha1'],
 dependency_overlays=fp['dependency_overlays'],runtime_source_sha256=fp['runtime_source_sha256'],
 input_contract_path='/tmp/qwen35_official_input_contract_20260908.json',input_contract_sha256=sha(contract_path.read_bytes()),
 official_FT_results_path='${ARTIFACT_ROOT}/codex_official_ft_exp2_default_20260908_v1/results.json',official_FT_results_sha256=sha(ft_path.read_bytes()),
 cache_path=fp['cache_path'],cache_sha256=fp['cache_sha256'],
 finite_FA_library=old['build_directory']+'/libdeltatrace_fa_finite_bf16_d256.so',finite_FA_library_sha256=old['library_sha256'],
 scope='Existing signed content_P1 finite propagation on exact official raw NI0 input, total588. No FT modification or reuse of605-token cached activations.',
 target='Sum of actual FP32 logsoftmax over all248 fixed response tokens including tokenizerEOS,full248320vocabulary. Preserve EOS baseline only at310eligible user tokens. No answer-only/weighted/selected target variation.',
 unchanged_finite_runtime_sha256={k:sha(v) for k,v in files.items() if k not in ['study.py','native_dense_attention_capture.py']},
 added_observation='Passive dense-FA observer extending unchanged existing capture. Actual Transformers uses flash_attn_func for unpadded equal-length batches; capture its real operands and request LSE from same public native function. No new attention or forward/backward implementation.',
 backend_contract='FT stays untouched B1eager+FLA. One native eager B1 clean diagnostic under same model/input; DT endpoints run actual defaultFA+FLA B2. Report batch/backend logprob differences; no claim of identical backend. Future original deletion scoring must be common.',
 batching='One attribution sample; EOS/original paired as actualB2. No padding or manufactured second benchmark sample. Existing finite callbacks already accept variable sample batch.',
 budget={'complete_model_loads':1,'diagnostic_eager_B1_root_forwards':1,'attribution_FA_B2_root_forwards':1,
  'original_decoder_replays_B2':32,'finite_decoders':32,'finite_FA_calls':8,'finite_FLA_calls':24,
  'public_dense_FA_auxiliary_forwards':8,'native_linear_conv_auxiliary_forwards':24,'native_linear_conv_auxiliary_backwards':24,
  'original_recovery_CPU_calls':1,'new_FT_runs':0,'generation_calls':0,'deletion_queries':0,'new_samples':0,'wall_time_seconds':900},
 stop='One frozen NI0 run. Stop and preserve the first input/source/runtime/nonfinite error; no automatic retry, FT edits, target/layer/precision tuning or extra samples. Finite residuals and replay differences retained, no tiny-rounding repair gate.',
 cost='Full resident model load, common-eager diagnostic separately timed, completeFAroot+CPUcheckpoints+32native replays+finite rules+auxFA/conv+diagnosticIO all recorded. Cold compilation and diagnostic overhead included. No fair warm speed claim from one run.',
 files_sha256={k:sha(v) for k,v in files.items()})
files['protocol.json']=json.dumps(p,indent=2).encode();(A/'dt_official_input_NI0_protocol_20260908.json').write_bytes(files['protocol.json'])
packed=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode();python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path("${ARTIFACT_ROOT}/codex_dt_official_NI0_20260908_v1");d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_dt_official_input_NI0_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],
                  'unchanged_runtime_files':len(p['unchanged_finite_runtime_sha256']),'budget':p['budget']}))
