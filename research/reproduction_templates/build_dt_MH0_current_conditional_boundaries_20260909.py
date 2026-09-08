"""Freeze one fixed MH0 DT plus four original boundary scores; never launch."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace';sha=lambda b:hashlib.sha256(b).hexdigest()
donor=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH0_pristine_FT0_comparison_20260909_v1'
dp=json.loads((donor/'protocol.json').read_bytes());dr=json.loads((donor/'results.json').read_bytes())
assert dr['status']=='MH0_current_DT_pristine_FT0_1DT1Both42score_complete' and dr['protocol']==dp
receipt=json.loads((donor/'terminal_receipt.json').read_bytes());assert receipt['pid_alive'] is False
for name,want in receipt['files'].items():assert sha((donor/name).read_bytes())==want,name
assert dr['sources_before']==dr['sources_after'] and dr['weight_stats_before']==dr['weight_stats_after']
assert dr['DT_entered']==dr['DT_returned']==1 and dr['scorer_entered']==dr['scorer_returned']==42
files={'study.py':(R/'research/reproduction_templates/dt_MH0_current_conditional_boundaries_20260909.py').read_bytes()}
for name,want in dp['files_sha256'].items():
    if name=='study.py':continue
    raw=(donor/name).read_bytes();assert sha(raw)==want
    paths=[root/name for root in (R/'core',R/'research/runtime',R/'research/reproduction_templates') if (root/name).is_file()]
    assert len(paths)==1 and paths[0].read_bytes()==raw,('Current method changed',name)
    files[name]=raw
key='morehopqa_0';case=dr['cases'][key];freeze=dr['input_freeze_before_model_load'][key]
assert case['input']['total_length']==853 and case['input']['input_sha256']=='6e1df21d426c5eb28217976f63b7c7932e502bde4fb064f77867df8205ebd618'
assert not case['gold'];ids=np.asarray(freeze['input_ids'],dtype=np.int64)
assert sha(ids.tobytes())==case['input']['input_sha256']
assert np.array_equal(ids[case['input']['prompt_length']:],np.asarray(freeze['target_ids']))
v=np.load(donor/'vectors.npz',allow_pickle=False);w=v[key+'_DT_evaluated'];full=v[key+'_DT_full']
assert w.dtype==np.float32 and np.isfinite(w).all() and w.shape==(case['input']['prompt_length'],)
assert full.shape==(853,) and np.array_equal(full[:len(w)].astype(np.float32),w)
curve=case['curves']['DT'];order=curve['sorted_keep'];assert sorted(order)==case['input']['keep']
assert np.all(np.diff(w[order].astype(np.float64))<=0)
assert curve['input_receipts']==dr['runs'][0]['deletion_audit']['input_receipts']
assert curve['input_receipts'][0]['deleted_positions']==[] and curve['input_receipts'][-1]['deleted_positions']==case['input']['keep']
assert len(curve['input_receipts'])==len(curve['scores'])==21
n,extra=divmod(len(order),20);offset=0;deleted=set();eos=int(ids[-1]);x=ids.copy()
receipts=[{'input_sha256':sha(x.tobytes()),'deleted_positions':[]}]
for step in range(20):
    group=order[offset:offset+n+(step<extra)];offset+=len(group);deleted.update(group);x[group]=eos
    assert np.where(x!=ids)[0].tolist()==sorted(deleted)
    receipts.append({'input_sha256':sha(x.tobytes()),'deleted_positions':sorted(deleted)})
assert receipts==curve['input_receipts'] and receipts[-1]['input_sha256']==freeze['baseline_sha256']
remote='${ARTIFACT_ROOT}/codex_dt_MH0_current_conditional_boundaries_20260909_v1'
python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
p=copy.deepcopy(dp)
for k in ('quality_schedule','prior_stability','FT_public_call','FT0_definition','successful_FT_provenance','evaluation','signed_outputs','cost','evaluation_scope','decision','method_target_semantics','stop'):
    p.pop(k,None)
p.update({
    'scope':'One actual fixed-current MH0 DT and four unchanged original evaluator B1 calls. Passive coefficient and native boundary captures locate conditional error jumps over every decoder; no new candidate, FT, generation, quality sweep or launch by this builder.',
    'call_schedule':[[key,'DT']], 'capture_steps':[0,3,10,20],
    'source_standalone':{'results_path':'/tmp/'+donor.name+'/results.json','results_sha256':sha((donor/'results.json').read_bytes()),
        'vectors_path':'/tmp/'+donor.name+'/vectors.npz','vectors_sha256':sha((donor/'vectors.npz').read_bytes()),
        'protocol_path':'/tmp/'+donor.name+'/protocol.json','protocol_sha256':sha((donor/'protocol.json').read_bytes()),
        'study_sha256':dp['files_sha256']['study.py'],'vector_key':key+'_DT',
        'scope':'Use this successful standalone current-DT own original deletion masks, not stability-control masks or newer multi-sample-run sorting. New diagnostic scores and vector are measured and any drift is reported; no bitwise equivalence requirement or silent substitution.'},
    'frozen_capture_receipts':{str(step):receipts[step] for step in [0,3,10,20]},
    'frozen_source_scores':{str(step):curve['scores'][step] for step in [0,3,10,20]},
    'source_mask_CPU_freeze':{'all_21_receipts_reconstructed':True,'source_vector_descending_on_source_order':True,
        'source_evaluated_vector_exact_FP32_projection_of_full':True,'allEOS_complete_keep_closure':True},
    'rule_scope':'Fixed already measured current DT: norm_gate_rules={0:symmetric}; finite_fla_by_layer={0:Layer0FLAEndpointAverage(original_backend)}; all other layers and native/core sources unchanged.',
    'backend_schedule':'One original BF16 model load; existing NI0 eager B1 initialization; official setter to FA; one current DT B2 attribution with passive boundary copies; four original default BF16/FA B1 evaluator calls with unmodified model arguments and no reused state.',
    'source_reuse':'Every non-study finite runtime/core/utility/span helper byte-identical to the source standalone comparison. Pin original evaluator, all16official package files, native sources, input cache, checkpoint configs and weight stat identities before/after.',
    'budget':{'model_loads':1,'native_eager_NI0_B1_initializations':1,'complete_DT_attributions':1,
        'DT_B2_roots':1,'native_decoder_replays_DT':32,'finite_decoder_calls':32,'public_FA_LSE_calls':8,'finite_FA_calls':8,
        'finite_GDN_callback_sites':24,'finite_FLA_backend_calls':25,'native_FLA_adjoint_stage_calls':50,
        'candidate_layer0_average_wrapper_calls':1,'finite_GDN_conv_preactivation_calls':24,'finite_GDN_conv_autograd_calls':24,
        'boundary_coefficient_captures':34,'native_B1_scorer_forwards':4,'boundary_captures_per_B1_scorer':34,
        'complete_official_FT_Both_calls':0,'generation_calls':0,'new_candidates':0,'original_metric_curves':0,
        'wall_time_seconds':900},
    'diagnostic_sign':'actual minus predicted. Decoder i jump = m_(i+1) dot (hC_(i+1)-hA_(i+1)) - m_i dot (hC_i-hA_i). Negative means DT overpredicts local drop; telescope includes native score/FP32 same-logit reduction, head seed, final norm and input-map terms.',
    'drift_policy':'Save this run full/evaluated signed vectors, full relativeL2/maxabs/equality to source, B2 root endpoint drift, B1 actual source-mask score drift, allEOS B1/B2 boundary drift and exact conditional-error drift partition. Nonzero drift is neither automatically a failure nor silently dismissed; large layer jumps are the diagnostic target.',
    'outputs':'Public binary NPZ: current full/evaluated signed vector plus per-token boundary contractions and32decoder jumps for steps3/10/20. Private remote .pt: actual34coefficients, actual paired native boundary tensors and four actual B1 boundary captures, no weights; exclude .pt from review ZIP and public Git. Record private bytes/SHA/save time for later no-repeat reuse.',
    'cost_scope':'All entered/returned work, initialization, observer transfers/CPU64 contractions, same-logit FP32 diagnostic and private save retained. Diagnostic instrumentation cost is not a production latency estimate. No additional model/FA/FLA/conv computation beyond declared attribution/scoring.',
    'decision':'Locate actual MH0 remaining conditional error without assuming NI1/MH1 layer causes. No candidate implementation, precision repair, model counterfactual claim, new original MAS curve or goal-completion decision from this diagnostic.',
    'stop':'First source/input/mask/finite/count/telescoping invariant failure or900seconds; save failure and entered/returned/unknown nonreturned stage counts. No extra model run, precision polishing or mask substitution.',
    'protected_sources':[{'path':dp['finite_FA_library'],'sha256':dp['finite_FA_library_sha256']}]+[
        {'path':'/tmp/'+donor.name+'/'+name,'sha256':sha((donor/name).read_bytes())} for name in ['results.json','vectors.npz','protocol.json']],
    'files_sha256':{name:sha(raw) for name,raw in files.items()}})
for name,raw in files.items():ast.parse(raw,filename=name)
files['protocol.json']=json.dumps(p,indent=2).encode()
pp=A/'dt_MH0_current_conditional_boundaries_protocol_20260909.json';lp=A/'launch_dt_MH0_current_conditional_boundaries_20260909.json'
assert not pp.exists() and not lp.exists(),'Do not overwrite frozen artifacts.'
blob=base64.b64encode(zlib.compress(json.dumps({name:base64.b64encode(raw).decode() for name,raw in files.items()}).encode())).decode()
loader=('import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);'
    'files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];'
    'f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);'
    '(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))')
pp.write_bytes(files['protocol.json']);lp.write_text(json.dumps({'cmd':python+' -c '+shlex.quote(loader),'timeout':10}))
print(json.dumps({'protocol_sha256':sha(files['protocol.json']),'study_sha256':p['files_sha256']['study.py'],
    'payload':str(lp),'protocol':str(pp),'remote_directory':remote,'budget':p['budget']}))
