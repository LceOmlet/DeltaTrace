"""Freeze exactly two original curves for already saved production MH vectors; no launch."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_normgate_production_20260908_v1';p=json.loads((D/'protocol.json').read_bytes())
raw=(D/'results.json').read_bytes();production=json.loads(raw);assert production['status']=='eight_call_production_integration_cost_complete'
vector_sha=sha((D/'vectors.npz').read_bytes());assert vector_sha=='8daba25f22bc1a48ae47d3700d94610897ce68aa0a76b36630cf3ebfa418e0a9'
files={name:(D/name).read_bytes() for name in p['files_sha256'] if name!='study.py'}
for name,value in files.items():assert sha(value)==p['files_sha256'][name],name
files['study.py']=(R/'research/reproduction_templates/dt_normgate_production_MH_metrics_20260908.py').read_bytes()
for name,value in files.items():ast.parse(value,filename=name)
ref=p['paired_references']['morehopqa_1']
paired_folder=A/'snapshot${ARTIFACT_ROOT}/codex_dt_normgate_repair_MH1_20260908_v1'
assert sha((paired_folder/'results.json').read_bytes())==ref['results_sha256']
parent_p=json.loads((paired_folder/'protocol.json').read_bytes())
for name in ['native_model_sha256','installed_FA_interface_sha256','runtime_source_sha256','native_stage_source_sha256',
             'checkpoint_config_tokenizer_sha256','expected_weight_stats','official_source_blob_sha1','checkpoint','cache_paths','cache_hashes']:
    assert parent_p[name]==p[name],name
for number,method in [(6,'current'),(7,'candidate')]:
    run=production['runs'][number];assert run['case']=='morehopqa_1' and run['method']==method and run['status']=='complete'
    assert not run['integration']['all_bin_input_receipts_equal']
for name in ['paired_references','production_norm_gate_rules','call_schedule','modified_production_runtime']:
    p.pop(name,None)
p.update(scope='Original42score MH1 followup for the saved normal-production current/candidate vectors whose actual deletion bins differ from paired-study bins. No repeated attribution or altered initialization.',
    cases=['morehopqa_1'],case_indices=[['morehopqa',1]],saved_production_run_numbers={'current':6,'candidate':7},
    production_results_path='${ARTIFACT_ROOT}/codex_dt_normgate_production_20260908_v1/results.json',production_results_sha256=sha(raw),
    production_vectors_path='${ARTIFACT_ROOT}/codex_dt_normgate_production_20260908_v1/vectors.npz',production_vectors_sha256=vector_sha,
    paired_reference=ref,
    method_change='No method execution or modification. Consume exact saved float64 full/float32 evaluated production vectors, reuse original MH1 prompt,target,keep,gold. Every non-study source is byte-identical to production.',
    budget={'model_loads':1,'native_eager_B1_diagnostics':1,'complete_DT_attributions':0,'finite_FA_calls':0,'finite_FLA_calls':0,
        'original_metric_curves':2,'native_B1_scoring_forwards':42,'FT_traces':0,'generation_calls':0,'new_samples':0,'wall_time_seconds':300},
    acceptance='All42actual B1 scorer input receipts must equal corresponding frozen production vector deletion bins. Actual unchanged original metric/scorer callables; passive original-function profiler and input/output hooks only. Preserve native BF16/FA/FLA,source/weight identities. Report scores0/20 versus prior paired scores and own full RISE/MAS/needle,including saved same-input FT comparisons with0newFT.',
    stop='Exactly42original native scorer forwards maximum,plus one fixed NI0eager initialization diagnostic. Preserve first failure andpartial data;no DT rerun,extra curves,framework modifications,or initialization changes toforce bitmatching.',
    costs='One actual model load,original NI0eager diagnostic,default native FA switch,then current/candidate original20step curves. All costs recorded; no finite attribution backend is invoked. Historical FT copied with hash provenance,not rerun.',
    files_sha256={name:sha(value) for name,value in files.items()})
files['protocol.json']=json.dumps(p,indent=2).encode();protocol=A/'dt_normgate_production_MH_metrics_protocol_20260908.json'
launch=A/'launch_dt_normgate_production_MH_metrics_20260908.json';assert not protocol.exists() and not launch.exists()
protocol.write_bytes(files['protocol.json'])
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(value).decode() for name,value in files.items()}).encode())).decode()
py='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python';directory='${ARTIFACT_ROOT}/codex_dt_normgate_production_MH_metrics_20260908_v1'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path('+repr(directory)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(py)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
launch.write_text(json.dumps({'cmd':py+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],
    'launch_payload':str(launch),'remote_directory':directory,'budget':p['budget']}))
