"""Freeze a three-example original-data MAS check before any quality result."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
base=A/'snapshot${ARTIFACT_ROOT}/codex_dt_official_NI0_20260908_v1'
p=json.loads((base/'protocol.json').read_bytes())
mp=json.loads((A/'snapshot${ARTIFACT_ROOT}/codex_dt_mlp_original_metrics_NI0_20260908_v1/protocol.json').read_bytes())
sp=json.loads((A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_official_spans_20260908_v1/protocol.json').read_bytes())
ft=json.loads((A/'snapshot${ARTIFACT_ROOT}/codex_official_ft_exp2_default_20260908_v1/protocol.json').read_bytes())
files={name:(base/name).read_bytes() for name in p['files_sha256'] if name!='study.py'}
for name in ['qwen35_dense_finite_runner.py','native_target_logit_rows.py','fixed_input_metric_view.py']:
    files[name]=(R/'research/runtime'/name).read_bytes()
files['official_span_mapping.py']=(A/'snapshot${ARTIFACT_ROOT}/codex_qwen35_official_spans_20260908_v1/official_span_mapping.py').read_bytes()
files['study.py']=(R/'research/reproduction_templates/dt_MAS_three_cases_20260908.py').read_bytes()
for name,raw in files.items():ast.parse(raw)
for name,want in p['unchanged_finite_runtime_sha256'].items():assert sha(files[name])==want
p.update(
 scope='Fixed currentDT versus complete unchangedFT on NI1,NI2,MH1. Small descriptive check of MAS heterogeneity, not an independent confirmation or sample-size/power study.',
 selection=[['niah_mq_q2',1],['niah_mq_q2',2],['morehopqa',1]],
 author_data_root=sp['author_root'],old_checkpoint=sp['old_checkpoint'],span_source_sha256=sp['span_source_sha256'],
 spans_path=ft['spans_path'],spans_sha256=ft['spans_sha256'],
 cache_paths={name:sp['author_root']+'/exp/exp2/data/'+name+'.jsonl' for name in sp['cache_sha256']},cache_hashes=sp['cache_sha256'],
 official_source_blob_sha1=mp['official_source_blob_sha1'],expected_weight_stats=mp['expected_weight_stats'],
 control_vectors_path='${ARTIFACT_ROOT}/codex_dt_official_NI0_20260908_v1/signed_result.npz',
 control_vectors_sha256=sha((base/'signed_result.npz').read_bytes()),control_input_sha256=mp['input_sha256'],
 control_vector_relative_L2_ceiling=0.02,control_needle_absolute_drift_ceiling=0.025,
 max_total_tokens=2048,method_order=['FT0','FT1','FT2','FT3','DT'],
 target='DT whole fixed response including tokenizerEOS/full vocabulary; unchanged FT native answer/reasoning span policy. Shared original scoring uses the whole fixed response. Preserve raw prompt and eligibleEOS convention from NI0.',
 implementation='Shared DT controller imports all finite mathematics unchanged and executes fresh actual roots plus32actual replays. No logits_to_keep optimization in this quality experiment. NI0 control checks broad default-precision drift before the three new cases. No native or FT source edits. Author token/span/filter helpers only are reused, never an extracted FT controller.',
 input_adapter=mp['input_adapter'],
 budget={'model_loads':1,'DT_attributions':4,'DT_native_roots':4,'DT_native_decoder_replays':128,'DT_finite_decoder_calls':128,
         'auxiliary_public_FA_calls':32,'unchanged_full_FT_traces':3,'original_metric_curves':15,'original_steps_per_curve':20,
         'native_B1_scoring_forwards':315,'new_generated_trajectories':0,'new_benchmark_datasets':0,'wall_time_seconds':900},
 batching='Sample B1; DT has two real endpoint rows. This pilot does not claim multi-example minibatching. Persistent model and compiler objects shared across four actual DT executions; native public backend setter switches to untouched FT eager and back to common FA scoring.',
 analysis='Keep each original curve,allFT0-3,needle where defined,raw response versus density,paired DT-minus-eachFT metrics. Report NI andMH separately; do not pool heterogeneous tasks into a victory statistic. The NI0 rerun is a controller check,not another independent sample.',
 stop='Exactly the three declared additional historically-used author cases. Preserve first error/partial data and stop; no automatic retry or substitute sample, no method tuning, no full dataset, no expansion to force significance. If all complete, inspect per-case evidence before further resources.',
 files_sha256={name:sha(raw) for name,raw in files.items()})
files['protocol.json']=json.dumps(p,indent=2).encode()
protocol=A/'dt_MAS_three_cases_protocol_20260908.json';assert not protocol.exists();protocol.write_bytes(files['protocol.json'])
blob=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python';directory='${ARTIFACT_ROOT}/codex_dt_MAS_three_cases_20260908_v1'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path('+repr(directory)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_dt_MAS_three_cases_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'budget':p['budget']}))
