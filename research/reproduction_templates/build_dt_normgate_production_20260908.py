"""Freeze eight production calls and a single-directory launch payload; never launch."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path

A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_normgate_repair_NI1_20260908_v1'
p=json.loads((D/'protocol.json').read_bytes())
exclude={'study.py','layer0_normgate_paired_repair_20260908.py','finite_fla_coefficient_capture_20260908.py','fla_beta_content_mismatch_audit_20260908.py'}
files={n:(D/n).read_bytes() for n in p['files_sha256'] if n not in exclude}
for name,raw in files.items():assert sha(raw)==p['files_sha256'][name],name
runner=(R/'research/runtime/qwen35_dense_finite_runner.py').read_bytes()
modified={'qwen35_dense_finite_runner.py':{'paired_sha256':sha(files['qwen35_dense_finite_runner.py']),'production_sha256':sha(runner)}}
files['qwen35_dense_finite_runner.py']=runner
files['study.py']=(R/'research/reproduction_templates/dt_normgate_production_20260908.py').read_bytes()
for name,raw in files.items():ast.parse(raw,filename=name)
refs={}
for key,suffix,status in [('niah_mq_q2_1','NI1','NI1_paired_normgate_repair_original_metrics_complete'),
                          ('morehopqa_1','MH1','MH1_paired_normgate_repair_original_metrics_complete')]:
    folder=A/('snapshot${ARTIFACT_ROOT}/codex_dt_normgate_repair_'+suffix+'_20260908_v1')
    raw=(folder/'results.json').read_bytes();result=json.loads(raw);assert result['status']==status
    parent_protocol=json.loads((folder/'protocol.json').read_bytes())
    for name in ['native_model_sha256','installed_FA_interface_sha256','runtime_source_sha256',
                 'native_stage_source_sha256','checkpoint_config_tokenizer_sha256','expected_weight_stats',
                 'official_source_blob_sha1','checkpoint','cache_paths','cache_hashes']:
        assert parent_protocol[name]==p[name],(key,name)
    for name in files:
        if name not in ('study.py','qwen35_dense_finite_runner.py'):
            assert parent_protocol['files_sha256'][name]==sha(files[name]),(key,name)
    remote='/tmp/'+folder.name
    refs[key]={'results_path':remote+'/results.json','results_sha256':sha(raw),
        'vectors_path':remote+'/vectors.npz','vectors_sha256':sha((folder/'vectors.npz').read_bytes())}
for key in ['capture_steps','parent_results_path','parent_results_sha256','parent_vectors_path','parent_vectors_sha256',
            'quality_parent_path','quality_parent_sha256','diagnostic_change','decomposition','observer_vector_relative_L2_ceiling',
            'decision_source_NI1_results_sha256','decision_source_NI1_metrics']:
    p.pop(key,None)
p.update(scope='Eight production integrations/cost observations of the frozen layer0 symmetric norm-gate candidate on existing NI1/MH1. No extra paired candidate propagation, no coefficient diagnostic backend, no new model scores.',
    cases=['niah_mq_q2_1','morehopqa_1'],case_indices=[['niah_mq_q2',1],['morehopqa',1]],paired_references=refs,
    production_norm_gate_rules={'current':{},'candidate':{'0':'symmetric'}},
    call_schedule=[['niah_mq_q2_1','current','initial'],['niah_mq_q2_1','candidate','initial'],
        ['niah_mq_q2_1','current','measured'],['niah_mq_q2_1','candidate','measured'],
        ['niah_mq_q2_1','candidate','measured'],['niah_mq_q2_1','current','measured'],
        ['morehopqa_1','current','integration'],['morehopqa_1','candidate','integration']],
    method_change='Only layer0 attribution norm_gate_rule is symmetric. The normal runner forwards an explicit copied configuration during its one existing GDN pass. Default omitted configuration follows old operations. All other runtime snapshots match both paired studies.',
    modified_production_runtime=modified,
    budget={'model_loads':1,'native_eager_B1_diagnostics':1,'complete_DT_attributions':8,
        'native_root_forwards':8,'native_decoder_replays':256,'finite_decoder_calls':256,
        'finite_FLA_calls':192,'finite_FA_calls':64,'public_FA_auxiliary_calls':64,
        'additional_candidate_repropagations':0,'diagnostic_coefficient_capture_calls':0,
        'native_B1_scoring_forwards':0,'original_metric_curves_with_new_scores':0,
        'maximum_original_metric_cached_replays':8,'maximum_cached_original_score_reads':168,
        'FT_traces':0,'generation_calls':0,'new_samples':0,'wall_time_seconds':600},
    acceptance='Record exact complete/evaluated vector differences, original torch ordering,20 deletion bins and all21full input hashes against the paired originals. Coefficient/order drift is reported,not a bitwise admission requirement. For each of8production vectors,identical bin inputs permits recomputing own NEW density/metrics with the unchanged original function reading hash-verified saved score totals. Different order within unchanged bins is harmless. Report own metrics/density minus paired values. Changed bin inputs require a separately bounded original-metric followup,not automatic curves. Source/weight identities and actual root input hashes must hold.',
    stop='Exactly eight scheduled DT calls unless first execution/source/count failure. Preserve partial failure; no retry, parameter sweep,extra examples,framework edits,or added scorer forwards. Integration mismatch is reported as pending quality followup,not silently accepted or permission to run more.',
    costs='NI initial C/S recorded separately; measured C,S,S,C medians use two samples each. Report full capture/checkpoint/replay attribution latency and peak allocated memory, allocated before/after GC, cold/cache context; no pure-kernel or statistical speed claim. Native model order remains load eager,one original NI0 eager diagnostic,switch original FA. No empty_cache within sequence and no observer/L-r0 captures. Source/hash/CPU export work outside attribute timing is separately visible.',
    files_sha256={k:sha(v) for k,v in files.items()})
files['protocol.json']=json.dumps(p,indent=2).encode()
protocol_path=A/'dt_normgate_production_protocol_20260908.json'
launch_path=A/'launch_dt_normgate_production_20260908.json'
assert not protocol_path.exists() and not launch_path.exists()
protocol_path.write_bytes(files['protocol.json'])
blob=base64.b64encode(zlib.compress(json.dumps({k:base64.b64encode(v).decode() for k,v in files.items()}).encode())).decode()
py='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python';directory='${ARTIFACT_ROOT}/codex_dt_normgate_production_20260908_v1'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path('+repr(directory)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(py)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
launch_path.write_text(json.dumps({'cmd':py+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],
    'runner_sha256':sha(runner),'launch_payload':str(launch_path),'remote_directory':directory,'budget':p['budget']}))
