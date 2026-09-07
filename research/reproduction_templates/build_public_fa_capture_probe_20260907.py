"""Freeze public FA metadata equivalence/cost on three actual original cases."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
old=(A/'pv_interaction_development16_20260907.py').read_text()
tree=ast.parse(old)
def fn(name):return ast.get_source_segment(old,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
driver=old[:old.index('\nclass Recorder(')]
driver+='''
import zipfile
import flash_attn.flash_attn_interface as fa_native
import transformers.integrations.flash_attention as fa_adapter
import transformers.utils.generic as generic
from qwen_signed_secant_native_paired_pv_rules import capture_checkpoint_pair
from qwen_signed_secant_native_paired_pv_rules import propagate_paired_secant as private_capture
from qwen_signed_secant_paired_public_fa import propagate_paired_secant as public_capture
for name,digest in p['sources'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
parent_path=Path(p['required_parent']);assert hashlib.sha256(parent_path.read_bytes()).hexdigest()==p['required_parent_sha256']
parent=json.loads(parent_path.read_bytes());assert parent['status']=='complete'
assert report['checkpoint_before']==parent['checkpoint_before']==parent['checkpoint_after']
torch.backends.cuda.matmul.allow_tf32=False
'''
driver+=fn('native_sources')+'\n'+"original_methods={name:type(m).forward for name,m in model.named_modules()}\n"+fn('native_method_audit')+'\n'
driver+='''
assert native_sources()==parent['native_sources_before']==parent['native_sources_after']
report.update(scope=p['purpose'],native_sources_before=native_sources(),native_root_forwards=0,native_vjps=0,
 evaluation_forwards=0,ft_attribution_forwards=0,ordinary_reference_forwards=0,native_attribution_forwards=0,
 manual_passes=0,extra_layer_replay_calls=0,extra_native_fa_attention_calls=0,
 native_attribution_endpoint_trajectories=0,extra_layer_replay_endpoint_trajectories=0)
'''
driver+=fn('save')+'\n'
run=fn('strong_run').replace("pv_rule='symmetric'","pv_rule='content_P1',capture_mode='private'")
run=run.replace('    torch.cuda.empty_cache();preprop_peak=0;finished=False',"    propagate_paired_secant=public_capture if capture_mode=='public' else private_capture\n    torch.cuda.empty_cache();preprop_peak=0;finished=False")
run=run.replace("result['extra_native_fa_attention_calls']==0","result['extra_native_fa_attention_calls']==(36 if capture_mode=='public' else 0)")
run=run.replace("report['manual_passes']+=1","report['manual_passes']+=1\n    report['extra_native_fa_attention_calls']+=result['extra_native_fa_attention_calls']\n    result['capture_mode']=capture_mode")
run=run.replace("report['extra_native_fa_attention_calls']+=result['extra_native_fa_attention_calls']", "assert result['extra_native_fa_attention_calls']==activity['auxiliary_completed']")
run=run.replace("    torch.cuda.empty_cache();preprop_peak=0;finished=False", "    activity={'auxiliary_attempts':0,'auxiliary_completed':0,'metadata':[]}\n    torch.cuda.empty_cache();preprop_peak=0;finished=False")
run=run.replace('result=propagate_paired_secant(model,reference,clean,pv_rule=pv_rule)', "result=propagate_paired_secant(model,reference,clean,pv_rule=pv_rule,**({'activity':activity} if capture_mode=='public' else {}))")
run=run.replace("        full['peak_allocated_bytes']=max(preprop_peak,full['peak_allocated_bytes'])", "        full['public_FA_activity']=activity\n        report['extra_native_fa_attention_calls']+=activity['auxiliary_completed']\n        full['peak_allocated_bytes']=max(preprop_peak,full['peak_allocated_bytes'])",1)
run=run.replace("{'pv_rule':pv_rule,'completed':finished", "{'pv_rule':pv_rule,'capture_mode':capture_mode,'completed':finished")
driver+=run+'\n'
driver+='''
save()
try:
    model.set_attn_implementation('flash_attention_2');native_method_audit(True)
    for case_number,(dataset,index) in enumerate(p['selection']):
        path=ROOT/f'exp/exp2/data/{dataset}.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==p['official_cache_sha256'][dataset]
        ex=runner.ds_utils.load_cached(path)[index]
        expected=next(r for r in parent['records'] if (r['dataset'],r['idx'])==(dataset,index))
        row={'dataset':dataset,'idx':index,'N':len(expected['input_ids']),
             'input_ids_sha256':expected['input_ids_sha256'],'runs':[],'profiles':{}}
        report['records'].append(row)
        for repeat in range(4):
            order=['private','public'] if (case_number+repeat)%2==0 else ['public','private']
            for mode in order:
                native_method_audit(True);report['active']=[dataset,index,repeat,mode];save()
                result=strong_run(ex,expected,capture_mode=mode)
                row['runs'].append({'repeat':repeat,'warmup':repeat==0,'mode':mode,'result':result})
                save();print('PUBLIC_CAPTURE_DONE',dataset,index,repeat,mode,result['seconds'],flush=True)
        if case_number==0:
            for mode in ['private','public']:
                with torch.profiler.profile(activities=list(torch.profiler.supported_activities())) as prof:
                    diagnostic=strong_run(ex,expected,capture_mode=mode)
                trace=mode+'_FA_trace.json';prof.export_chrome_trace(str(HERE/trace))
                names=[e.name for e in prof.events() if str(e.device_type)=='DeviceType.CUDA']
                kernels=sum('flash_fwd_kernel' in name for name in names)
                row['profiles'][mode]={'trace':trace,'sha256':hashlib.sha256((HERE/trace).read_bytes()).hexdigest(),
                    'actual_flash_forward_kernels':kernels,'result':diagnostic}
                assert kernels==(108 if mode=='public' else 72)
                del prof,diagnostic;save()
        print('PUBLIC_CAPTURE_CASE_DONE',dataset,index,flush=True)
    report['checkpoint_after']=checkpoint_receipt();assert report['checkpoint_before']==report['checkpoint_after']
    report['native_sources_after']=native_sources();assert report['native_sources_before']==report['native_sources_after']
    for key,value in p['budget'].items():assert report[key]==value,(key,report[key],value)
    report['status']='complete';report.pop('active',None);native_method_audit(True)
except Exception:
    report['status']='failed';report['error']=traceback.format_exc();raise
finally:
    report['elapsed_seconds']=time.time()-entry_started;save()
    names=['study.py','protocol.json','results.json']+list(p['sources'])+[x.name for x in HERE.glob('*_FA_trace.json')]
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(HERE/name,name)
'''
ast.parse(driver)
study=A/'public_fa_capture_probe_20260907.py';study.write_text(driver,encoding='utf-8')
prior=json.loads((A/'pv_interaction_development16_protocol_20260907.json').read_text())
p={k:prior[k] for k in ['checkpoint','checkpoint_receipt','checkpoint_receipt_sha256','native_model_source_sha256',
 'official_normalized_sources','official_cache_sha256','official_evaluation_backend','maximum_queue_seconds']}
sources=list(prior['sources'])+['qwen_public_fa_layer_replay.py','qwen_signed_secant_paired_public_fa.py']
p.update(purpose='Three original NI0/NI2/MH0 cases, frozen P1 rule. Replace private LSE-slot capture with passive public-call observation plus one unmodified public FA metadata call per layer. No native model/FA implementation changes. Explicit finite arithmetic remains dense; no multi-example attribution or finite FA-kernel completion claim.',
 study_sha256=sha(study),sources={name:sha(A/name) for name in sources},
 required_parent='${ARTIFACT_ROOT}/codex_pv_interaction_development16_20260907_v1/results.json',
 required_parent_sha256='679842f0ccfc32769a416868f6c08f70ddffff84298b80b02f02ca9da1457476',
 selection=[['niah_mq_q2',0],['niah_mq_q2',2],['morehopqa',0]],
 wait_for_pid=106651,wait_for_script='codex_public_fa_capture_probe_20260907_v1',
 repeats='Each case/mode1warm+3measured, rotated; NI0 one extra profile per mode. Profile calls charged but excluded from latency medians.',
 budget={'native_root_forwards':26,'native_vjps':0,'evaluation_forwards':0,'ft_attribution_forwards':0,'ordinary_reference_forwards':0,
  'native_attribution_forwards':26,'manual_passes':26,'extra_layer_replay_calls':936,'extra_native_fa_attention_calls':468,
  'native_attribution_endpoint_trajectories':52,'extra_layer_replay_endpoint_trajectories':1872},
 predeclared_decision={'exact_frozen_P1_vectors_required_for_reusing_quality':True,
  'max_per_case_median_latency_ratio_to_same_job_private':1.15,
  'max_per_case_peak_extra_bytes_to_same_job_private':76021760,
  'threshold_scope':'Researcher-chosen conservative implementation adoption guard, not a user-mandated threshold or new scientific quality gate. Failed cost guard means retain as explicit unsupported default candidate; do not hide extra calls.'},
 attribution_batch_scope='Physical B2 is one example at two actual endpoints; not two-example batching.',
 metadata_scope='Documented public LSE only; testing S_dmask must be None or empty Tensor and is never decoded. Actual descriptor retained. Auxiliary output never used by model. Python public-call observer is an explicit capability requirement, not all-future-version compatibility.',
 previous_attempt={'directory':'${ARTIFACT_ROOT}/codex_public_fa_capture_probe_20260907_v1','status':'failed_return_type_guard','raw_sha256':sha(A/'snapshot${ARTIFACT_ROOT}/codex_public_fa_capture_probe_20260907_v1/results.json'),
  'change':'Accept no testing matrix represented by None as well as empty Tensor; retain actual descriptor. Fix exception-path auxiliary-call accounting. No arithmetic, cost or numerical adoption threshold changes.',
  'actual_budget_reconstructed_from_calls_and_failure_site':{'root_forwards':2,'root_endpoint_trajectories':4,'extra_layer_replays':37,'extra_endpoint_layer_replays':74,'auxiliary_FA_calls':1,'completed_manual_passes':1,'failed_manual_attempts':1,'evaluation_forwards':0,'backwards':0},
  'raw_counter_caveat':'v1 extra_native_fa_attention_calls incorrectly remained0 because updated only on full-attribution success. One completed auxiliary call is proven by the subsequent failed return guard. Raw preserved; use explicit corrected historical budget, not overwritten raw.'})
protocol=A/'public_fa_capture_probe_protocol_20260907.json';protocol.write_text(json.dumps(p,indent=2),encoding='utf-8')
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_public_fa_capture_probe_20260907_v2',
 f'study.py={study}',f'protocol.json={protocol}',*[f'{name}={A/name}' for name in sources],
 '--request',str(A/'public_fa_capture_probe_launch_20260907.json')],check=True)
print('Frozen public capture probe:26root/0backward,936layer replays,468extra public FA calls.')
