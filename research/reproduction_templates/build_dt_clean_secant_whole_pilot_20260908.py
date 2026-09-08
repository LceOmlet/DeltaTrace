"""Freeze uniform all8FA paired pilot; no remote launch."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
donor=A/'snapshot${ARTIFACT_ROOT}/codex_dt_normgate_production_20260908_v1'
dp=json.loads((donor/'protocol.json').read_bytes())
local=A/'snapshot${ARTIFACT_ROOT}/codex_dt_fa19_clean_secant_20260908_v1'
lr=json.loads((local/'results.json').read_bytes());lb=json.loads((local/'build_results.json').read_bytes())
assert lr['status']=='one_clean_secant_saved_layer_candidate_observed' and lr['dv_bitwise_equal_to_original']
assert lb['status']=='finite_extension_compiled_not_executed'
assert sha((local/'results.json').read_bytes())==json.loads((local/'terminal_receipt.json').read_bytes())['files']['results.json']['sha256']
old_study=(R/'research/reproduction_templates/dt_normgate_production_20260908.py').read_text(encoding='utf-8')
tree=ast.parse(old_study)
selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in ('CountFinite','check_counts','deletion_audit')]
assert {node.name for node in selected}=={'CountFinite','check_counts','deletion_audit'}
utility='"""Unchanged audit/count helpers extracted from the preserved production study."""\nimport torch,hashlib\nsha=lambda b:hashlib.sha256(b).hexdigest()\n\n'+'\n\n'.join(ast.get_source_segment(old_study,node) for node in selected)+'\n'
utility_path=R/'research/reproduction_templates/normal_finite_study_utils_20260908.py'
if utility_path.exists():assert utility_path.read_text(encoding='utf-8')==utility
else:utility_path.write_text(utility,encoding='utf-8',newline='\n')
files={'study.py':(R/'research/reproduction_templates/dt_clean_secant_whole_pilot_20260908.py').read_bytes(),
       utility_path.name:utility.encode()}
for name in dp['files_sha256']:
    if name=='study.py':continue
    matches=[root/name for root in (R/'research/runtime',R/'core') if (root/name).is_file()]
    assert len(matches)==1,(name,matches)
    files[name]=matches[0].read_bytes()
for name in ('vendor_fa_clean_secant_20260908.py','vendor_fa_clean_secant_runner_20260908.py'):
    files[name]=(R/'research/runtime'/name).read_bytes()
assert sha(files['vendor_fa_clean_secant_20260908.py'])==lb['protocol']['files_sha256']['vendor_fa_clean_secant_20260908.py']
for name,data in files.items():ast.parse(data,filename=name)
remote='${ARTIFACT_ROOT}/codex_dt_clean_secant_whole_pilot_20260908_v1'
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
keys=('checkpoint','checkpoint_config_tokenizer_sha256','native_model_sha256','isolated_site','compiler_cache',
      'boundary_compiler_cache','installed_FA_interface_sha256','native_stage_source_sha256','official_root',
      'official_source_blob_sha1','dependency_overlays','runtime_source_sha256','expected_weight_stats','cache_paths',
      'cache_hashes','finite_FA_library','finite_FA_library_sha256','paired_references')
p={key:dp[key] for key in keys}
p.update({'scope':'Uniform clean-secant softmax at all8FA layers with existing layer0 symmetric GDN control. Explicit finite-backend injection into unchanged normal runner; no layer-fitting/callcount dispatch or native/framework modification.',
    'case_indices':[['morehopqa',1],['niah_mq_q2',1]],
    'candidate_FA_layers':[3,7,11,15,19,23,27,31],
    'call_schedule':[['morehopqa_1','control'],['morehopqa_1','candidate'],['niah_mq_q2_1','candidate'],['niah_mq_q2_1','control']],
    'rule_scope':{'control':'Original LM finite FA; norm_gate_rules={0:symmetric}.',
                  'candidate':'Same norm_gate_rules, unchanged FLA/QKmidpoint/PVP1 and normal runner; AllFACleanSecantBackend injected once for every full_attention layer.'},
    'candidate_library':'/tmp/'+local.name+'/'+lb['protocol']['diagnostic_library_name'],
    'candidate_library_sha256':lb['library']['sha256'],
    'local_candidate_results_path':'/tmp/'+local.name+'/results.json',
    'local_candidate_results_sha256':sha((local/'results.json').read_bytes()),
    'protected_sources':[{'path':'/tmp/'+local.name+'/'+name,'sha256':sha((local/name).read_bytes())} for name in ('results.json','build_results.json','protocol.json')]
        +[{'path':dp['finite_FA_library'],'sha256':dp['finite_FA_library_sha256']},
          {'path':'/tmp/'+local.name+'/'+lb['protocol']['diagnostic_library_name'],'sha256':lb['library']['sha256']}],
    'utility_provenance':{'source':'research/reproduction_templates/dt_normgate_production_20260908.py','sha256':sha(old_study.encode()),
        'functions':['CountFinite','check_counts','deletion_audit'],'scope':'Exact extracted function/class source; audit deletion groups only. Original author function performs metrics.'},
    'budget':{'model_loads':1,'native_eager_NI0_B1_initializations':1,'complete_DT_attributions':4,
        'DT_B2_roots':4,'native_decoder_replays':128,'finite_decoder_calls':128,'public_FA_LSE_calls':32,
        'finite_FA_calls':32,'finite_FLA_calls':96,'original_metric_curves':4,'native_B1_scorer_forwards':84,
        'FT_calls':0,'generation_calls':0,'new_samples':0,'repeat_timing_runs':0,'wall_time_seconds':900},
    'evaluation':'Same original author prompt/target caches and k20 faithfulness_test_skip_tokens, same actual LLMAttributionEvaluator for both methods. FixedInputMetricView only preserves original frozen raw input, explicitly not stock paper-table reproduction. Every21input hash/deletion mask must match own fresh signed vector audit. No cached score simulation.',
    'signed_outputs':'Retain fullFP64 vectors, evaluatedFP32 prompt vectors, own actual21point original curves, needle, complete signed/normalization/endpoint data. No negative clipping, mass tuning, changed target, changed sorting or amended FT.',
    'fixed_control_masks':'CPU contraction of both new signed evaluated vectors on newly scored control21 masks using actual same-execution control scores;0extra scorers. Record clean/allEOS equality across method curves. Do not substitute historical scores or call this a new MAS curve.',
    'cost':'One run per method percase,opposite order acrosscases. Preserve fullnormal runner timings/peak memory,compiler setup andall candidate row-state checks/copies insideattribute timing. Candidate15row states and extra reductions stillpresent. This is pilotcost,not clean kernel-only timing or proof of stable speed.',
    'risks':'All8 layers alter upstream multipliers,may worsen lower layers or introduce small-variance/cancellation effects. One-layer improvement doesnot guarantee whole-model needle,RISE,MAS. Fifteen-row diagnostic overhead,CPU summaries and synchronization can slowcandidate. No posthoc layerselection.',
    'decision':'Completion requires joint original-metric and actual-cost evidence for both cases. All8quality regression closespromotion of thiscandidate;no automatic layer scan,redo or more samples. Rootreviews result before any further study.',
    'stop':'First nonfinite/source/layout/input/ABI/count/metric invariant failure or900seconds;preserve partialevidence. No retry,backend fallback,precision change,FT call,extra scoring,extra candidate or timingrepeat.',
    'files_sha256':{name:sha(data) for name,data in files.items()}})
files['protocol.json']=json.dumps(p,indent=2).encode()
pp=A/'dt_clean_secant_whole_pilot_protocol_20260908.json';lp=A/'launch_dt_clean_secant_whole_pilot_20260908.json'
assert not pp.exists() and not lp.exists(),'Do not overwrite frozen protocol or payload.'
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(data).decode() for name,data in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
        'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
        'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
        '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],
    'payload':str(lp),'protocol':str(pp),'remote_directory':remote,'budget':p['budget']}))
