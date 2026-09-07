"""Freeze original long-context/long-rollout cost and finite-vector checks."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=(A/'vendor_fa_end_to_end_probe_20260907.py').read_text()
driver=base[:base.index('\nclass Recorder(')]
# Compact serialization changes only experiment bookkeeping outside API timers.
driver=driver.replace('json.dumps(report,ensure_ascii=False,indent=2)','json.dumps(report,ensure_ascii=False,separators=(",",":"))')
driver=driver.replace("result['endpoint_scores32']={'before':reference['score32_sum64'],'after':clean['score32_sum64']}",
 "result['endpoint_scores32']={'before':reference['score32_sum64'],'after':clean['score32_sum64']}\n            result['capture_peak_before_propagation']=preprop_peak\n            result['propagation_peak_before_full_max']=result['peak_allocated_bytes']")
driver+=r'''
capacity_file=Path(p['capacity_file']);assert hashlib.sha256(capacity_file.read_bytes()).hexdigest()==p['capacity_sha256']
capacity=json.loads(capacity_file.read_text());assert capacity['status']=='complete' and capacity['native_forwards']==capacity['native_vjps']==0
report.update(gpu_capacity_bytes=torch.cuda.get_device_properties(0).total_memory,quality_evaluated=False)
save()
try:
    model.set_attn_implementation('flash_attention_2');native_method_audit(True)
    for case_number,(dataset,index) in enumerate(p['selection']):
        path=ROOT/f'exp/exp2/data/{dataset}.jsonl';assert hashlib.sha256(path.read_bytes()).hexdigest()==p['official_cache_sha256'][dataset]
        ex=runner.ds_utils.load_cached(path)[index]
        engine=both.LLMIFRAttributionBoth(model,tokenizer,show_progress=False)
        ids,mask,plen,glen=engine._ensure_generation(ex.prompt,ex.target)
        positions=list(engine.user_prompt_indices);keep=both.keep_token_indices(engine.user_prompt_tokens)
        expected={'input_ids':ids[0].tolist(),'prompt_len':plen,'user_positions':positions,'keep_local_indices':keep,'eligible_positions':[positions[j] for j in keep]}
        digest=hashlib.sha256(json.dumps(expected['input_ids']).encode()).hexdigest()
        cap=next(x for x in capacity['datasets'][dataset]['records'] if x['idx']==index)
        assert digest==cap['input_ids_sha256'] and len(expected['input_ids'])==cap['full_tokens'] and plen==cap['formatted_prompt_tokens']
        row=dict(expected,dataset=dataset,idx=index,input_ids_sha256=digest,N=ids.shape[1],runs=[],profiles={})
        report['records'].append(row);del engine,ids,mask
        for repeat in range(4):
            order=['dense','finite'] if (case_number+repeat)%2==0 else ['finite','dense']
            for mode in order:
                native_method_audit(True);report['active']=[dataset,index,repeat,mode];save()
                value=strong_run(ex,expected,capture_mode=mode)
                row['runs'].append({'repeat':repeat,'warmup':repeat==0,'mode':mode,'result':value})
                save();print('ORIGINAL_LONG_DONE',dataset,index,repeat,mode,value['seconds'],value['peak_allocated_bytes'],flush=True)
        ids=torch.tensor([expected['input_ids']],device='cuda:0',dtype=torch.long);mask=torch.ones_like(ids)
        native_method_audit(True);report['active']=[dataset,index,'ordinary_input_backward'];save()
        # Reference helper has no parameter gradients and uses the same native
        # FA, actual target trajectory and native selected output head.
        reference=ordinary_input_backward(model,ids,mask,plen)
        row['ordinary_backward']=reference
        report['native_root_forwards']+=1;report['native_vjps']+=1;report['ordinary_reference_forwards']+=1
        del ids,mask;save()
        if [dataset,index]==p['profile_case']:
            native_method_audit(True);report['active']=[dataset,index,'finite_profile'];save()
            with torch.profiler.profile(activities=list(torch.profiler.supported_activities()),record_shapes=True,profile_memory=True) as prof:
                value=strong_run(ex,expected,capture_mode='finite')
            trace='long_finite_FA_trace.json';prof.export_chrome_trace(str(HERE/trace))
            names=[x.name for x in prof.events() if str(x.device_type)=='DeviceType.CUDA']
            record={'trace':trace,'sha256':hashlib.sha256((HERE/trace).read_bytes()).hexdigest(),'result':value,
                 'actual_default_FA_forward_kernels':sum('flash_fwd_kernel' in n for n in names),
                 'actual_finite_kernels':sum('deltatrace_fa_finite_p1_kernel' in n for n in names)}
            assert record['actual_default_FA_forward_kernels']==record['actual_finite_kernels']==108
            row['profiles']['finite']=record;del prof,value,names;save()
    report['checkpoint_after']=checkpoint_receipt();assert report['checkpoint_before']==report['checkpoint_after']
    report['native_sources_after']=native_sources();assert report['native_sources_before']==report['native_sources_after']
    native_method_audit(True)
    for key,value in p['budget'].items():assert report[key]==value,(key,report[key],value)
    report['status']='complete';report.pop('active',None)
except Exception:
    report['status']='failed';report['error']=traceback.format_exc();raise
finally:
    report['elapsed_seconds']=time.time()-entry_started;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','results.json']+list(p['sources'])+[x.name for x in HERE.glob('*_FA_trace.json')]:z.write(HERE/name,name)
'''
ast.parse(driver);study=A/'vendor_fa_original_long_cost_20260907.py';study.write_text(driver,encoding='utf-8')
p=json.loads((A/'vendor_fa_end_to_end_protocol_20260907.json').read_text())
p.update(study_sha256=sha(study),wait_for_pid=138339,wait_for_script='${ARTIFACT_ROOT}/codex_batch_endpoint_diagnostic_20260907_v1/study.py',
    purpose='Original FT cache inputs only: NI2 length1241 control, maximum Hotpot original context3470, maximum MATH original rollout3762. Same-job dense P1 vs traceable FA finite P1, full costs and signed vectors; same-job ordinary native FA input-backward memory. No quality selection or independent-quality claim, no truncation or invented long benchmark.',
    selection=[['niah_mq_q2',2],['hotpotqa_long',22],['math',92]],profile_case=['math',92],
    capacity_file='${ARTIFACT_ROOT}/codex_qwen_official_cache_capacity_20260906_v1/results.json',capacity_sha256='4db8f66946de9ad66f3d1313a472f6d5e626173715d201a5fecac6c6a90dd250',
    precision_policy='Preserve actual native FP16/default FA. Finite formulas and vector residuals reviewed without bitwise or FP32 accuracy gate; never silently fall back to dense. Full original quality remains separately required.',
    budget={'native_root_forwards':28,'native_vjps':3,'evaluation_forwards':0,'ft_attribution_forwards':0,'ordinary_reference_forwards':3,
       'native_attribution_forwards':25,'manual_passes':25,'extra_layer_replay_calls':900,'extra_native_fa_attention_calls':900,
       'native_attribution_endpoint_trajectories':50,'extra_layer_replay_endpoint_trajectories':1800,'finite_FA_calls_attempted':468,'finite_FA_calls_enqueued':468})
protocol=A/'vendor_fa_original_long_cost_protocol_20260907.json';protocol.write_text(json.dumps(p,indent=2))
files=['study.py='+str(study),'protocol.json='+str(protocol)]+[name+'='+str(A/name) for name in p['sources']]
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_vendor_fa_original_long_cost_20260907_v1',*files,'--request',str(A/'launch_vendor_fa_original_long_cost_20260907.json')],check=True)
print('Frozen 28 root forwards,3 native input VJPs,25 finite passes; original inputs,zero quality queries.')
