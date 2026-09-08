"""Freeze native full/selected logit rows with unchanged finite mathematics."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
parent=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MAS_three_cases_20260908_v2';old=json.loads((parent/'protocol.json').read_bytes())
keys=['checkpoint','verified_shard_sizes','checkpoint_config_tokenizer_sha256','native_model_sha256','isolated_site',
 'compiler_cache','boundary_compiler_cache','installed_FA_interface_sha256','native_stage_source_sha256',
 'official_root','official_package_blob_sha1','official_source_blob_sha1','dependency_overlays','runtime_source_sha256',
 'input_contract_path','input_contract_sha256','cache_path','cache_sha256','finite_FA_library','finite_FA_library_sha256',
 'unchanged_finite_runtime_sha256','expected_weight_stats']
p={k:old[k] for k in keys}
names=list(old['unchanged_finite_runtime_sha256'])+['native_dense_attention_capture.py','native_target_logit_rows.py','qwen35_dense_finite_runner.py','fixed_input_metric_view.py']
files={name:(parent/name).read_bytes() for name in names}
files['study.py']=(R/'research/reproduction_templates/dt_native_logit_rows_NI0_20260908.py').read_bytes()
for raw in files.values():ast.parse(raw)
for name,want in p['unchanged_finite_runtime_sha256'].items():assert sha(files[name])==want
p.update(
 scope='One fixed originalNI0, four complete currentDT runs full/selected/selected/full actual native head rows. NoFT attribution, no new method or benchmark sample.',
 input_sha256=old['control_input_sha256'],run_order=[['A0_full',False],['B0_selected',True],['B1_selected',True],['A1_full',False]],
 method_change='Only model public logits_to_keep argument. Original realFA/FLA model, exact targets/prompt/eligibleEOS, full vocabulary and every finite rule remain unchanged. All four calls start fresh actual endpoint roots and32actual decoder replays. No activation reuse or substitute head.',
 input_adapter=old['input_adapter'],
 target='Whole248fixed response tokens includingEOS. Causal predictor rows339..586 from actual588-token input. Same PackedAnswerTargets and seed for full/selected outputs.',
 precision='Native defaultBF16/FA. Do not demand bitwise scores or gate on arbitrary historical-vector epsilon. Compare selected-versus-full drift to same-method repeats and all four original quality curves; report errors and do not silently promote substantial quality changes.',
 timing='A0includes first full call;B0includes first selected-head shape;B1/A1 are a single warm selected/full pair. Same persistent model/compiled objects and controller. Record complete attribution, component times, peak allocated/reserved and housekeeping/archive costs. One warm pair cannot establish robust population speed. Model load, native eagerB1 diagnostic and84evaluation forwards counted separately.',
 budget={'model_loads':1,'native_eager_B1_diagnostics':1,'complete_DT_attributions':4,'actual_endpoint_roots':4,
         'native_decoder_replays':128,'finite_decoder_calls':128,'public_FA_auxiliary_calls':32,
         'original_metric_curves':4,'steps_per_curve':20,'native_B1_scoring_forwards':84,
         'FT_traces':0,'generation_calls':0,'new_samples':0,'wall_time_seconds':600},
 acceptance='Validate realhead input rows/output size and complete target packing; evaluate needle,originalRISE/MAS,logprob and signed-vector differences for every run. Save full vectors before decisions. No coefficient/source change, gold-based target selection, score clipping or metric recalibration. Report shape-default numeric differences, any regression, and observed cost without declaring unmeasured generality.',
 batching='One actual sample, two EOS/input endpoint rows. SmallCPU indexing sentinel checks union/gather for repeated positions across two samples, but is NOT an actual multi-example model or throughput acceptance.',
 stop='Only this four-call paired check. Preserve first failure and partial results; no blind retry, extra rounds, FT repair or sample expansion. Inspect completed cost/quality before deciding adoption or the next bottleneck.',
 files_sha256={name:sha(raw) for name,raw in files.items()})
files['protocol.json']=json.dumps(p,indent=2).encode()
dst=A/'dt_native_logit_rows_NI0_protocol_20260908.json';assert not dst.exists();dst.write_bytes(files['protocol.json'])
blob=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python';directory='${ARTIFACT_ROOT}/codex_dt_native_logit_rows_NI0_20260908_v1'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path('+repr(directory)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_dt_native_logit_rows_NI0_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'budget':p['budget']}))
