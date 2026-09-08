"""Freeze original NI0 metric curves on already-saved scores; no FT modification."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
base=A/'snapshot${ARTIFACT_ROOT}/codex_dt_official_NI0_20260908_v1';p=json.loads((base/'protocol.json').read_bytes());r=json.loads((base/'results.json').read_bytes())
paired=A/'snapshot${ARTIFACT_ROOT}/codex_dt_mlp_content1_NI0_20260908_v2';pr=json.loads((paired/'results.json').read_bytes());assert pr['status']=='paired_DT_MLP_content1_NI0_completed'
files={'study.py':(R/'research/reproduction_templates/dt_mlp_original_metrics_NI0_20260908.py').read_bytes(),
 'fixed_input_metric_view.py':(R/'research/runtime/fixed_input_metric_view.py').read_bytes()}
for raw in files.values():ast.parse(raw)
tree=json.loads((A/'official_FT_e81_source_tree_20260908.json').read_bytes());entries={x['path']:x['sha'] for x in tree['tree'] if x['type']=='blob'}
official={**p['official_package_blob_sha1'],**{n:entries[n] for n in ['llm_attr_eval.py','shared_utils.py']}}
for name,want in official.items():
 path=(A/'snapshot${PRIVATE_MOUNT_PATH}'/name)
 if not path.exists():path=R/'research/third_party/flashtrace_qwen35_e81b3be'/name
 raw=path.read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==want,name
p.update(scope='Original seven20-step NI0 EOS deletion curves: originalFT0-3 saved vectors, frozenDT, same-run symmetricDT and soleMLPcontent1 candidate. One sample, no new attribution or FT calculation.',
 FT_vectors_path='${ARTIFACT_ROOT}/codex_official_ft_exp2_default_20260908_v1/official_outputs.npz',
 DT_frozen_vectors_path='${ARTIFACT_ROOT}/codex_dt_official_NI0_20260908_v1/signed_result.npz',
 DT_paired_vectors_path='${ARTIFACT_ROOT}/codex_dt_mlp_content1_NI0_20260908_v2/signed_vectors.npz',
 input_sha256=r['input']['clean_sha256'],expected_weight_stats=r['weight_stats_after'],
 official_source_blob_sha1=official,
 method_order=['FT0','FT1','FT2','FT3','DT_frozen','DT_sym_control','DT_MLP_content1'],
 input_adapter='Explicit FixedInputMetricView preserves the exact raw588 attribution input. Original evaluator legacy Context/chat formatting would change the input; record its length/hash without scoring that alternate input. Only formatting is adapted, not FT/metric/scoring mathematics. Do not call this stock run_exp or paper-table reproduction.',
 metric_contract='Direct original flashtrace.improved.faithfulness_test_skip_tokens(k=20), original llm_attr_eval.LLMAttributionEvaluator.compute_logprob_response_given_prompt, all eligible310, originalEOS, actualQwen3.5 BF16/FA/FLA, commonB1 scoring for all seven curves. No score clipping or rescaling. Original model output/logsoftmax precision retained.',
 observation='Passive metric return-frame capture records raw scores/density/normalized response/alignment/corrected response. Original model hooks verify every actual input, eligibleEOS substitutions and147 completed forwards. No scorer/controller replacement or fake planning values.',
 budget={'model_loads':1,'original_metric_curves':7,'original_steps_per_curve':20,'actual_native_B1_scoring_forwards':147,'attribution_calls':0,'FT_runs':0,'generation_calls':0,'new_samples':0,'wall_time_seconds':600},
 stop='One fixed NI0 comparison. Preserve first error and partial raw curves, no automatic retry or formula/score tuning. No Morehop/full-dataset expansion before reading this decision-relevant result.',
 cost='Evaluation-only model load, seven serial original-method curves and all147 nativeB1 forwards; no fake minibatch claim. Cold start, passive input capture and original full-vocabulary logprob costs included. Not an attribution performance benchmark.',
 files_sha256={k:sha(v) for k,v in files.items()})
paths={p['FT_vectors_path']:A/'snapshot${ARTIFACT_ROOT}/codex_official_ft_exp2_default_20260908_v1/official_outputs.npz',
 p['DT_frozen_vectors_path']:base/'signed_result.npz',p['DT_paired_vectors_path']:paired/'signed_vectors.npz',
 p['input_contract_path']:A/'snapshot/tmp/qwen35_official_input_contract_20260908.json'}
p['data_sha256']={remote:sha(path.read_bytes()) for remote,path in paths.items()};p['data_sha256'][p['cache_path']]=p['cache_sha256']
files['protocol.json']=json.dumps(p,indent=2).encode();f=A/'dt_mlp_original_metrics_NI0_protocol_20260908.json';assert not f.exists();f.write_bytes(files['protocol.json'])
blob=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python';directory='${ARTIFACT_ROOT}/codex_dt_mlp_original_metrics_NI0_20260908_v1'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path('+repr(directory)+');d.mkdir(exist_ok=False);'
 'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
 'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
 '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
(A/'launch_dt_mlp_original_metrics_NI0_20260908.json').write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'budget':p['budget']}))
