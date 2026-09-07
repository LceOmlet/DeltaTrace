"""Freeze original three-case end-to-end P1 and original-curve comparison."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
old=(A/'public_fa_capture_probe_20260907.py').read_text()
driver=old[:old.index('\nsave()\ntry:\n')]
driver=driver.replace('from qwen_signed_secant_native_paired_pv_rules import capture_checkpoint_pair','from qwen_signed_secant_paired_public_fa import capture_checkpoint_pair')
driver=driver.replace("capture_mode='private'","capture_mode='finite'")
driver=driver.replace("propagate_paired_secant=public_capture if capture_mode=='public' else private_capture",
 "assert capture_mode in ['dense','finite']\n    propagate_paired_secant=finite_capture if capture_mode=='finite' else public_capture\n    finite_activity=[]")
driver=driver.replace("**({'activity':activity} if capture_mode=='public' else {})", "activity=activity,**({'finite_activity':finite_activity} if capture_mode=='finite' else {})")
driver=driver.replace("full['public_FA_activity']=activity", "full['public_FA_activity']=activity\n        full['finite_FA_activity']=finite_activity\n        report['finite_FA_calls_attempted']+=sum(x.get('calls_attempted',0) for x in finite_activity)\n        report['finite_FA_calls_enqueued']+=sum(x.get('calls_enqueued',0) for x in finite_activity)")
driver=driver.replace("(36 if capture_mode=='public' else 0)","36")
extra='''
from functools import partial
from qwen_signed_secant_paired_vendor_fa import propagate_paired_secant as finite_entry
from vendor_fa_finite_runtime import VendorFAFiniteP1
from native_backward_memory_reference_target_head import ordinary_input_backward
tick=time.perf_counter()
build_path=Path(p['build_result']);assert hashlib.sha256(build_path.read_bytes()).hexdigest()==p['build_result_sha256']
extension=VendorFAFiniteP1(p['library'],p['library_sha256'])
finite_capture=partial(finite_entry,finite_attention=extension)
report.update(finite_library_initialization_seconds=time.perf_counter()-tick,
              finite_FA_calls_attempted=0,finite_FA_calls_enqueued=0)
'''
driver+=extra
pv=(A/'pv_interaction_development16_20260907.py').read_text()
driver+=pv[pv.index('\nclass Recorder('):pv.index('\nfrom qwen_signed_secant_native_paired_pv_rules import')]
driver+='''
save()
try:
    model.set_attn_implementation('flash_attention_2');native_method_audit(True)
    for case_number,(dataset,index) in enumerate(p['selection']):
        path=ROOT/f'exp/exp2/data/{dataset}.jsonl'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==p['official_cache_sha256'][dataset]
        ex=runner.ds_utils.load_cached(path)[index]
        expected=next(r for r in parent['records'] if (r['dataset'],r['idx'])==(dataset,index))
        row={key:expected[key] for key in ['dataset','idx','input_ids','input_ids_sha256','prompt_len',
            'user_positions','keep_local_indices','eligible_positions','gold_full_local','gold_eligible_local']}
        row.update(N=len(expected['input_ids']),runs=[],profiles={},scores={},metrics={},evaluation_masks={},evaluation_costs={})
        report['records'].append(row)
        for repeat in range(4):
            order=['dense','finite'] if (case_number+repeat)%2==0 else ['finite','dense']
            for mode in order:
                native_method_audit(True);report['active']=[dataset,index,repeat,mode];save()
                result=strong_run(ex,expected,capture_mode=mode)
                row['runs'].append({'repeat':repeat,'warmup':repeat==0,'mode':mode,'result':result})
                save();print('FINITE_WHOLE_DONE',dataset,index,repeat,mode,result['seconds'],result['peak_allocated_bytes'],flush=True)
        if case_number==0:
            for mode in ['dense','finite']:
                with torch.profiler.profile(activities=list(torch.profiler.supported_activities()),record_shapes=True,profile_memory=True) as prof:
                    diagnostic=strong_run(ex,expected,capture_mode=mode)
                trace=mode+'_FA_trace.json';prof.export_chrome_trace(str(HERE/trace))
                names=[e.name for e in prof.events() if str(e.device_type)=='DeviceType.CUDA']
                native_kernels=sum('flash_fwd_kernel' in name for name in names)
                finite_kernels=sum('deltatrace_fa_finite_p1_kernel' in name for name in names)
                row['profiles'][mode]={'trace':trace,'sha256':hashlib.sha256((HERE/trace).read_bytes()).hexdigest(),
                    'actual_flash_forward_kernels':native_kernels,'finite_kernels':finite_kernels,'result':diagnostic,
                    'scope':'Extra complete attribution profile with shapes/memory; all calls charged, excluded from normal medians.'}
                assert native_kernels==108 and finite_kernels==(108 if mode=='finite' else 0)
                del prof,diagnostic;save()
        ids=torch.tensor([row['input_ids']],dtype=torch.long,device='cuda:0');mask=torch.ones_like(ids)
        native_method_audit(True);report['active']=[dataset,index,'ordinary_native_backward'];save()
        reference=ordinary_input_backward(model,ids,mask,row['prompt_len'])
        row['ordinary_backward']=reference
        report['native_root_forwards']+=1;report['native_vjps']+=1;report['ordinary_reference_forwards']+=1
        for mode in ['dense','finite']:
            selected=next(r['result'] for r in row['runs'] if r['mode']==mode and r['repeat']==1)
            signed=torch.tensor(selected['signed_full_sequence'],dtype=torch.float64)
            row['scores'][mode]=signed[row['user_positions']].clamp_min(0).float().tolist()
            del signed
        del ids,mask;save()
    report['all_native_vectors_frozen_before_any_quality']=True;save()
    model.set_attn_implementation(p['official_evaluation_backend']);native_method_audit(True)
    for row in report['records']:
        dataset,index=row['dataset'],row['idx'];ex=runner.ds_utils.load_cached(ROOT/f'exp/exp2/data/{dataset}.jsonl')[index]
        expected=next(r for r in parent['records'] if (r['dataset'],r['idx'])==(dataset,index))
        ids=torch.tensor([row['input_ids']],dtype=torch.long,device='cuda:0');plen=row['prompt_len']
        positions=row['user_positions'];keep=row['keep_local_indices']
        evaluator.original_prompt=ids[:,:plen];evaluator.original_response=ids[:,plen:];evaluator.positions=positions;evaluator.keep=set(keep)
        gold=runner.ds_utils.ruler_gold_prompt_token_indices(ex,tokenizer);assert gold==row['gold_full_local']
        raw=tokenizer(' '+ex.prompt,add_special_tokens=False).input_ids
        assert all(raw[j]==int(ids[0,positions[j]]) for j in gold)
        for mode in ['dense','finite']:
            report['active']=[dataset,index,'original_quality',mode];save()
            score=torch.tensor(row['scores'][mode],dtype=torch.float32);evaluator.curve,evaluator.deleted=[],[]
            with measured() as cost,torch.no_grad():
                values=both.faithfulness_test_skip_tokens(evaluator,score[None],ex.prompt,ex.target,
                    keep_prompt_token_indices=keep,user_prompt_indices=positions)
            report['native_root_forwards']+=cost['native_forwards'];report['evaluation_forwards']+=cost['native_forwards']
            verify_masks(score[None],evaluator.curve,evaluator.deleted,ids,positions,keep);assert cost['native_forwards']==21
            assert [evaluator.curve[0],evaluator.curve[-1]]==expected['common_eager_evaluation_endpoints16']
            recovery=None if not gold else both.evaluate_attr_recovery_skip_tokens(score[None],keep_prompt_token_indices=keep,gold_prompt_token_indices=gold)
            row['metrics'][mode]={'rise':values[0],'mas':values[1],'mas_unnormalized':values[2],'recovery':recovery,'raw_curve':list(evaluator.curve)}
            row['evaluation_masks'][mode]=list(evaluator.deleted);row['evaluation_costs'][mode]=cost
            if recovery is not None:
                k=max(1,math.ceil(.1*len(keep)));top=torch.topk(score[torch.tensor(keep,dtype=torch.long)].clamp_min(0),k,largest=True).indices.tolist()
                chosen=[keep[j] for j in top];assert len(set(chosen)&set(row['gold_eligible_local']))/len(row['gold_eligible_local'])==recovery
                row.setdefault('recovery_topk_local',{})[mode]=chosen
            native_method_audit(True);save();print('ORIGINAL_QUALITY_DONE',dataset,index,mode,flush=True)
        row['historical_FT_control']={'source_sha256':p['required_parent_sha256'],
            'metrics':{name:values for name,values in expected['metrics'].items() if name.startswith('flashtrace_')},
            'scope':'Pinned previously verified original 0-3 hop controls, same checkpoint/inputs/evaluator; not freshly recomputed in this integration pilot.'}
        row['complete']=True;del ids;save()
    model.set_attn_implementation('flash_attention_2');native_method_audit(True)
    report['checkpoint_after']=checkpoint_receipt();assert report['checkpoint_before']==report['checkpoint_after']
    report['native_sources_after']=native_sources();assert report['native_sources_before']==report['native_sources_after']
    assert hashlib.sha256(Path(p['library']).read_bytes()).hexdigest()==p['library_sha256']
    for key,value in p['budget'].items():assert report[key]==value,(key,report[key],value)
    report['status']='complete';report.pop('active',None)
except Exception:
    report['status']='failed';report['error']=traceback.format_exc();raise
finally:
    report['elapsed_seconds']=time.time()-entry_started;save()
    names=['study.py','protocol.json','results.json']+list(p['sources'])+[x.name for x in HERE.glob('*_FA_trace.json')]
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(HERE/name,name)
'''
ast.parse(driver);study=A/'vendor_fa_end_to_end_probe_20260907.py';study.write_text(driver,encoding='utf-8')
p=json.loads((A/'public_fa_capture_probe_protocol_20260907.json').read_text())
for key in ['previous_attempt','predeclared_decision','metadata_scope']:p.pop(key,None)
sources=list(p['sources'])+['vendor_fa_finite_runtime.py','qwen_signed_secant_vendor_fa.py','qwen_signed_secant_paired_vendor_fa.py']
b=json.loads((A/'vendor_fa_finite_build_summary_20260907.json').read_text())
p.update(purpose='Integrate traceable vendor FA finite content-P1 operator into original whole-model finite attribution, eliminating global NxN production attention intermediates. Default actual model FA unchanged. Three original cases, full same-job dense control, original 21-point curves, actual native input-backward memory references; not independent or long-sequence confirmation.',
 study_sha256=sha(study),sources={name:sha(A/name) for name in sources},
 wait_for_pid=123015,wait_for_script='codex_vendor_fa_captured_operator_20260907_v2',
 build_result='${ARTIFACT_ROOT}/codex_vendor_fa_finite_build_20260907_v1/results.json',build_result_sha256=b['raw_sha256'],
 library='${ARTIFACT_ROOT}/codex_vendor_fa_finite_build_20260907_v1/libdeltatrace_fa_finite.so',library_sha256=b['library']['sha256'],
 budget={'native_root_forwards':155,'native_vjps':3,'evaluation_forwards':126,'ft_attribution_forwards':0,
 'ordinary_reference_forwards':3,'native_attribution_forwards':26,'manual_passes':26,'extra_layer_replay_calls':936,
 'extra_native_fa_attention_calls':936,'native_attribution_endpoint_trajectories':52,
 'extra_layer_replay_endpoint_trajectories':1872,'finite_FA_calls_attempted':468,'finite_FA_calls_enqueued':468},
 repeats='Each original case/dense or finite mode:1warm+3measured in rotated order. NI0 extra complete shape/memory profile per mode. Three ordinary native input backward references. All26attributions frozen before six original21-point deletion curves.',
 predeclared_review={'precision':'User accepts default-FA style mixed precision absent implementation errors; bitwise equality and FP32-level error are not adoption gates.',
 'numerical':'Report whole vector relative error, sign changes and their contribution mass, ranking/recovery, global unassigned and layer validity. No after-the-fact threshold tuning.',
 'quality':'Fresh original RISE/MAS/needle for dense and finite. Original FT0-3 pinned parent controls retained as historical comparison. Three-case pilot does not establish new quality superiority or inherit full16quality automatically.',
 'cost':'Include preparation, root capture, all replays, auxiliary public FA, finite wrappers/kernels and runtime checks. Library initialization and cold warmup reported separately. No silent fallback or autotuned method substitution.',
 'memory':'Remove production NxN probability/logmean/score/dense-audit materialization; verify actual profile tensor shapes and finite kernels. Compare whole attribution peak against same-job dense and actual ordinary native backward.'})
protocol=A/'vendor_fa_end_to_end_protocol_20260907.json';protocol.write_text(json.dumps(p,indent=2),encoding='utf-8')
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_vendor_fa_end_to_end_20260907_v1',
 f'study.py={study}',f'protocol.json={protocol}',*[f'{name}={A/name}' for name in sources],
 '--request',str(A/'vendor_fa_end_to_end_launch_20260907.json')],check=True)
request=json.loads((A/'vendor_fa_end_to_end_launch_20260907.json').read_text());assert len(request['cmd'])<128000
print('Frozen end-to-end protocol; payload',len(request['cmd']),'characters.')
