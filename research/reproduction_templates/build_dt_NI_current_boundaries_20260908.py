"""Freeze one current NI DT and four frozen original scoring captures; no launch."""
import ast,base64,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_clean_secant_whole_pilot_20260908_v1'
raw=(D/'results.json').read_bytes();vectors=(D/'vectors.npz').read_bytes()
assert sha(raw)=='192ae5916a6ecd561b256e3efa43847d58d2d0a083148550894c7a0a9c3cc923'
assert sha(vectors)=='1380a35216f9c187ad5a6df2c876db1de2a0a7f37eeae611ac737a48dcad93b2'
prior=json.loads(raw);dp=json.loads((D/'protocol.json').read_bytes())
assert prior['status']=='all8FA_clean_secant_4DT84score_pilot_complete'
curve=prior['cases']['niah_mq_q2_1']['curves']['control']
assert len(curve['scores'])==21 and len(curve['input_receipts'])==21
assert prior['runs'][3]['case']=='niah_mq_q2_1' and prior['runs'][3]['method']=='control'
assert prior['runs'][3]['details']['norm_gate_rules']=={'0':'symmetric'}
files={}
omit={'study.py','vendor_fa_clean_secant_20260908.py','vendor_fa_clean_secant_runner_20260908.py','normal_finite_study_utils_20260908.py'}
for name,want in dp['files_sha256'].items():
    if name in omit:continue
    value=(D/name).read_bytes();assert sha(value)==want
    candidates=[base/name for base in (R/'research/runtime',R/'core') if (base/name).exists()]
    assert len(candidates)==1 and sha(candidates[0].read_bytes())==want,('current shared source drift',name)
    files[name]=value
files['study.py']=(R/'research/reproduction_templates/dt_NI_current_boundaries_20260908.py').read_bytes()
for name,value in files.items():ast.parse(value,filename=name)
keys=('checkpoint','checkpoint_config_tokenizer_sha256','native_model_sha256','isolated_site','compiler_cache',
      'boundary_compiler_cache','installed_FA_interface_sha256','native_stage_source_sha256','official_root',
      'official_source_blob_sha1','dependency_overlays','runtime_source_sha256','expected_weight_stats','cache_paths',
      'cache_hashes','finite_FA_library','finite_FA_library_sha256')
p={key:dp[key] for key in keys}
p.update(scope='Boundary-only measurement of one normal current NI1 DT with layer0 symmetric gate. No candidate,interior diagnostics,full metric curve,new samples or native/FT modifications.',
    case='niah_mq_q2_1',capture_steps=[0,1,10,20],boundaries=[str(i) for i in range(33)]+['norm'],
    norm_gate_rules={'0':'symmetric'},reference_method='niah_mq_q2_1_control',
    reference_results_path='/tmp/'+D.name+'/results.json',reference_results_sha256=sha(raw),
    reference_vectors_path='/tmp/'+D.name+'/vectors.npz',reference_vectors_sha256=sha(vectors),
    source_provenance='Only the study changes. Shared runtime/core byte-pinned to successful latest wholepilot and current repository. Study directly simplifies completed MH19/6 observer: remove all decoder interior profilers and retain passive34module boundary hooks.',
    parent_study_sha256=sha((R/'research/reproduction_templates/dt_decoder19_6_conditional_20260908.py').read_bytes()),
    budget={'model_loads':1,'native_eager_NI0_B1_initializations':1,'complete_DT_attributions':1,'DT_B2_roots':1,
        'native_decoder_replays':32,'finite_decoder_calls':32,'public_FA_LSE_calls':8,'finite_FA_calls':8,'finite_FLA_calls':24,
        'native_B1_scorer_forwards':4,'extra_operator_calls':0,'interior_decoder_diagnostics':0,
        'FT_calls':0,'generation_calls':0,'new_samples':0,'full_metric_curves':0,'candidate_calls':0,'wall_time_seconds':600},
    ledger='For current frozen boundary multiplier m_l and actual B1clean-minus-deleted hidden delta_l, C_l=<m_l,delta_l>. Decoder i error=C_i-C_i+1;final norm=C32-Cnorm;head/seed/scoring=Cnorm-(native clean score-native deleted score);input mapping=sum(current evaluated signed scores on frozen deleted mask)-C0. All terms prediction-minus-actual and telescope to evaluated prediction-minus-native effect.',
    interpretation='Head/seed/scoring also contains DT FP32 seed versus original BF16 scorer and B1/B2 endpoint differences. Boundary localization alone does not prove an internal operator defect. Token groups label actual hidden-coordinate contraction positions,not independent source causality; signed cancellation across boundaries must be retained.',
    drift='Compare fresh full/evaluated signed vectors and root/seed effect with frozen current-control reference;compare4native scores and exact inputs with corresponding original21curve points. B1clean/B2input and B1allEOS/B2EOS perboundary drift are reported. Historical34coefficient drift is unavailable because latestpilot did not capture those coefficients;do not replace with old coefficients or enforce bitidentity/2percent admission.',
    artifacts='Private CPU pt:34actual m tensors,34actual B2 endpoint hidden tensors,and4sets of34actual B1 hidden tensors. Linear O(34*T*hidden),no weights/fullvocabulary logits/T-squared attention matrix. Small vectors.npz keeps signed token contractions/adjacent differences and current input attribution;review ZIP excludes private tensors. Record tensor/file bytes and SHA256.',
    acceptance='Exact source/weights/tokenized input/frozen scorer-mask hashes. One normal DT8FA24FLA32replays8publicLSE. Four actual scorer forwards with34captured boundaries each. Nonfinite or ledger/group closure failure stops;finite historical drift is report-only. No newMAS/RISE or speed comparison from4points.',
    stop='One load/one eagerNI0/one currentDT/four fixed scores only. First failure or600seconds preserves partialevidence;unreturned native DT work is unknown rather thanzero. No retry,alternate initialization,extra operator,candidate,precision change,FT call or automatic further study.',
    files_sha256={name:sha(value) for name,value in files.items()})
files['protocol.json']=json.dumps(p,indent=2).encode()
protocol=A/'dt_NI_current_boundaries_protocol_20260908.json';launch=A/'launch_dt_NI_current_boundaries_20260908.json'
assert not protocol.exists() and not launch.exists(),'Do not overwrite frozen measurement.'
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(value).decode() for name,value in files.items()}).encode())).decode()
py='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python';directory='${ARTIFACT_ROOT}/codex_dt_NI_current_boundaries_20260908_v1'
loader=('import base64,pathlib,subprocess,json,zlib;d=pathlib.Path('+repr(directory)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(py)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
protocol.write_bytes(files['protocol.json']);launch.write_text(json.dumps({'cmd':py+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],'payload':str(launch),'remote_directory':directory,'budget':p['budget']}))
