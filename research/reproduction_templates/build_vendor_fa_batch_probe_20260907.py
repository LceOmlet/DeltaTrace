"""Freeze genuine B1/B2/B4 example throughput and batched original scoring."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
original=(A/'vendor_fa_end_to_end_probe_20260907.py').read_text()
driver=original[:original.index('\ndef strong_run(')]
driver+=original[original.index('\nfrom functools import partial'):original.index('\nclass Recorder(')]
driver+=original[original.index('\ndef strong_run('):original.index('\nfrom functools import partial')]
driver+='''
from qwen_signed_secant_batched_vendor_fa_public import capture_batch,propagate_batch
from original_ft_coroutine_evaluation import evaluate_curves_batched
report.update(batch_runs=[],batch_profiles={},original_batched_evaluation_activity={})

def batch_run(examples,expected):
    activity={'auxiliary_attempts':0,'auxiliary_completed':0,'metadata':[]};finite_activity=[]
    batch=len(examples);torch.cuda.empty_cache();preprop_peak=0;finished=False
    report['manual_attempts']=report.get('manual_attempts',0)+1;save()
    try:
        with measured() as cost:
            requests=[];identities=[]
            for ex,old in zip(examples,expected):
                engine=both.LLMIFRAttributionBoth(model,tokenizer,show_progress=False)
                ids,mask,plen,glen=engine._ensure_generation(ex.prompt,ex.target)
                positions=list(engine.user_prompt_indices);keep=both.keep_token_indices(engine.user_prompt_tokens)
                eligible=[positions[j] for j in keep]
                identity={'input_ids':ids[0].tolist(),'prompt_len':plen,'user_positions':positions,'keep_local_indices':keep,'eligible_positions':eligible}
                assert all(value==old[key] for key,value in identity.items())
                baseline=ids.clone();baseline[0,torch.tensor(eligible,device=ids.device)]=tokenizer.eos_token_id
                requests.append((baseline,ids,plen));identities.append(identity)
                del engine,ids,mask,baseline
            before,after=capture_batch(model,requests,tokenizer.eos_token_id)
            preprop_peak=torch.cuda.max_memory_allocated()
            result=propagate_batch(model,before,after,activity=activity,finite_attention=extension,finite_activity=finite_activity)
            result['propagation_peak_before_full_max']=result['peak_allocated_bytes']
            result['capture_peak_before_propagation']=preprop_peak
            del before,after,requests
        finished=True
    finally:
        cost['public_FA_activity']=activity;cost['finite_FA_activity']=finite_activity
        cost['peak_allocated_bytes']=max(preprop_peak,cost['peak_allocated_bytes'])
        report['extra_native_fa_attention_calls']+=activity['auxiliary_completed']
        report['native_root_forwards']+=cost['native_forwards'];report['native_attribution_forwards']+=cost['native_forwards']
        report['native_attribution_endpoint_trajectories']+=cost['native_forward_trajectories']
        report['extra_layer_replay_calls']+=cost['extra_replay_calls']
        report['extra_layer_replay_endpoint_trajectories']+=cost['extra_replay_trajectories']
        report['finite_FA_calls_attempted']+=sum(x.get('calls_attempted',0) for x in finite_activity)
        report['finite_FA_calls_enqueued']+=sum(x.get('calls_enqueued',0) for x in finite_activity)
        report.setdefault('native_attempt_costs',[]).append({'capture_mode':'batch'+str(batch),'completed':finished,'cost':dict(cost)})
        save()
    assert cost['native_forwards']==1 and cost['native_forward_trajectories']==2*batch
    assert cost['native_decoder_layer_calls']==72 and cost['native_decoder_layer_trajectories']==144*batch
    assert cost['extra_replay_calls']==36 and cost['extra_replay_trajectories']==72*batch
    report['manual_passes']+=1
    result.update(end_to_end_cost=cost,propagation_internal_seconds=result['seconds'],seconds=cost['seconds'],peak_allocated_bytes=cost['peak_allocated_bytes'])
    return result

save()
try:
    model.set_attn_implementation('flash_attention_2');native_method_audit(True)
    rows=[];examples=[]
    for dataset,index in p['selection']:
        path=ROOT/f'exp/exp2/data/{dataset}.jsonl';assert hashlib.sha256(path.read_bytes()).hexdigest()==p['official_cache_sha256'][dataset]
        examples.append(runner.ds_utils.load_cached(path)[index])
        old=next(r for r in parent['records'] if (r['dataset'],r['idx'])==(dataset,index))
        rows.append(old)
        row={key:old[key] for key in ['dataset','idx','input_ids','input_ids_sha256','prompt_len','user_positions','keep_local_indices','eligible_positions','gold_full_local','gold_eligible_local']}
        row.update(single_runs=[],scores={},metrics={},evaluation_masks={},evaluation_costs={},recovery_topk_local={})
        report['records'].append(row)
    for repeat in range(4):
        order=['single','batch2','batch4'];offset=repeat%3;order=order[offset:]+order[:offset]
        for mode in order:
            groups=[[i] for i in range(4)] if mode=='single' else ([[0,1],[2,3]] if mode=='batch2' else [[0,1,2,3]])
            for group in groups:
                native_method_audit(True);report['active']=[repeat,mode,group];save()
                if mode=='single':result=strong_run(examples[group[0]],rows[group[0]],capture_mode='finite')
                else:result=batch_run([examples[i] for i in group],[rows[i] for i in group])
                entry={'repeat':repeat,'warmup':repeat==0,'mode':mode,'indices':group,'result':result}
                if mode=='single':report['records'][group[0]]['single_runs'].append(entry)
                else:report['batch_runs'].append(entry)
                save();print('REAL_BATCH_DONE',repeat,mode,group,result['seconds'],result['peak_allocated_bytes'],flush=True)
    with torch.profiler.profile(activities=list(torch.profiler.supported_activities()),record_shapes=True) as prof:
        value=batch_run(examples,rows)
    trace='batch4_FA_trace.json';prof.export_chrome_trace(str(HERE/trace));names=[e.name for e in prof.events() if str(e.device_type)=='DeviceType.CUDA']
    report['batch_profiles']['batch4']={'trace':trace,'sha256':hashlib.sha256((HERE/trace).read_bytes()).hexdigest(),'result':value,
        'native_FA_forward_kernels':sum('flash_fwd_kernel' in n for n in names),'finite_FA_kernels':sum('deltatrace_fa_finite_p1_kernel' in n for n in names)}
    assert report['batch_profiles']['batch4']['native_FA_forward_kernels']==report['batch_profiles']['batch4']['finite_FA_kernels']==108
    del prof,value
    for i,row in enumerate(report['records']):
        values={'single':next(x['result']['signed_full_sequence'] for x in row['single_runs'] if x['repeat']==1)}
        for mode in ['batch2','batch4']:
            item=next(x for x in report['batch_runs'] if x['repeat']==1 and x['mode']==mode and i in x['indices'])
            sample=item['indices'].index(i);values[mode]=item['result']['signed_full_sequence'][sample][:len(row['input_ids'])]
        for mode,signed in values.items():row['scores'][mode]=torch.tensor(signed,dtype=torch.float64)[row['user_positions']].clamp_min(0).float().tolist()
        ft='flashtrace_legacy_hop0' if row['dataset']=='niah_mq_q2' else 'flashtrace_both_hop2'
        row['scores']['historical_FT']=rows[i]['scores'][ft]
        row['historical_FT_source']={'name':ft,'parent_sha256':p['required_parent_sha256'],'fresh_attribution':False}
    report['all_native_vectors_frozen_before_any_quality']=True;save()
    model.set_attn_implementation(p['official_evaluation_backend']);native_method_audit(True)
    evaluator=runner.llm_attr_eval.LLMAttributionEvaluator(model,tokenizer)
    jobs={};job_rows={};expected_tensors={}
    for i,(row,ex) in enumerate(zip(report['records'],examples)):
        ids=torch.tensor([row['input_ids']],dtype=torch.long,device='cuda:0');plen=row['prompt_len']
        expected_tensors[i]=(ids[:,:plen],ids[:,plen:])
        for mode in ['single','batch2','batch4','historical_FT']:
            key=str(i)+'_'+mode;job_rows[key]=(i,mode);row['evaluation_masks'][mode]=[]
            jobs[key]={'attribution':torch.tensor(row['scores'][mode],dtype=torch.float32)[None],
                'prompt':ex.prompt,'generation':ex.target,'keep_prompt_token_indices':row['keep_local_indices'],'user_prompt_indices':row['user_positions']}
    def record_request(key,prompt,response):
        i,mode=job_rows[key];row=report['records'][i];original_prompt,original_response=expected_tensors[i]
        assert torch.equal(response,original_response)
        changed=(prompt[0]!=original_prompt[0]).nonzero().flatten().tolist()
        inverse={a:j for j,a in enumerate(row['user_positions'])}
        assert all(a in inverse and int(prompt[0,a])==tokenizer.eos_token_id for a in changed)
        local=[inverse[a] for a in changed];assert set(local)<=set(row['keep_local_indices'])
        row['evaluation_masks'][mode].append(local)
    activity=report['original_batched_evaluation_activity'];report['active']=['original_curves','batch4'];save()
    try:
        with measured() as cost:
            metrics=evaluate_curves_batched(evaluator,both,jobs,4,activity,request_observer=record_request)
    finally:
        report['native_root_forwards']+=cost['native_forwards'];report['evaluation_forwards']+=cost['native_forwards']
        report['batched_evaluation_cost']=cost;save()
    assert cost['native_forwards']==activity['physical_evaluation_forwards']==84
    assert cost['native_forward_trajectories']==activity['evaluation_trajectories']==336
    assert set(activity['actual_batch_sizes'])=={4}
    for key,values in metrics.items():
        i,mode=job_rows[key];row=report['records'][i];ex=examples[i]
        gold=runner.ds_utils.ruler_gold_prompt_token_indices(ex,tokenizer);assert gold==row['gold_full_local']
        score=torch.tensor(row['scores'][mode],dtype=torch.float32)
        recovery=None if not gold else both.evaluate_attr_recovery_skip_tokens(score[None],keep_prompt_token_indices=row['keep_local_indices'],gold_prompt_token_indices=gold)
        row['metrics'][mode]={'rise':values[0],'mas':values[1],'mas_unnormalized':values[2],'recovery':recovery,'raw_curve':activity['curves'][key]}
        row['evaluation_costs'][mode]={'shared_batch_record':'batched_evaluation_cost','evaluation_trajectories':21,'physical_calls_not_exclusive':True}
        if recovery is not None:
            keep=row['keep_local_indices'];k=max(1,math.ceil(.1*len(keep)));indices=torch.topk(score[torch.tensor(keep)].clamp_min(0),k).indices.tolist()
            row['recovery_topk_local'][mode]=[keep[j] for j in indices]
    native_method_audit(True);model.set_attn_implementation('flash_attention_2')
    report['checkpoint_after']=checkpoint_receipt();assert report['checkpoint_before']==report['checkpoint_after']
    report['native_sources_after']=native_sources();assert report['native_sources_before']==report['native_sources_after']
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
ast.parse(driver);study=A/'vendor_fa_batch_probe_20260907.py';study.write_text(driver,encoding='utf-8')
p=json.loads((A/'vendor_fa_end_to_end_protocol_20260907.json').read_text())
sources=list(p['sources'])+['compiled_secant_batched_layout.py','qwen_signed_secant_batched_vendor_fa.py',
 'qwen_signed_secant_batched_vendor_fa_public.py','original_ft_batched_evaluation.py','original_ft_coroutine_evaluation.py']
p.update(purpose='Actual mini-batch attribution: four distinct original samples, B1 existing finite-FA control, B2 pairs, B4 together. Preserve actual default FA and finite P1. Trailing EOS only after complete fixed response; per-example seeds. Original curve code suspended only at real native scoring sites and resumed with actual token logprobs; evaluate16original curves using B4 native scorer. No placeholder scores, alternate metric or shadow model.',
 study_sha256=sha(study),sources={name:sha(A/name) for name in sources},
 selection=[['niah_mq_q2',0],['niah_mq_q2',3],['niah_mq_q2',6],['morehopqa',1]],
 wait_for_pid=128474,wait_for_script='codex_vendor_fa_development16_20260907_v1',
 budget={'native_root_forwards':113,'native_vjps':0,'evaluation_forwards':84,'ft_attribution_forwards':0,
 'ordinary_reference_forwards':0,'native_attribution_forwards':29,'manual_passes':29,'extra_layer_replay_calls':1044,
 'extra_native_fa_attention_calls':1044,'native_attribution_endpoint_trajectories':104,
 'extra_layer_replay_endpoint_trajectories':3744,'finite_FA_calls_attempted':1044,'finite_FA_calls_enqueued':1044},
 attribution_batch_scope='B=1/2/4 distinct examples; physical native endpoint batch2/4/8. One warm+three measured per grouping, rotated. Extra B4 profile charged. No sample duplication or self-created input benchmark.',
 repeats='B1fourcalls/B2twocalls/B4onecall per repeat,1warm+3measured, plusoneB4profile:29attributions.16curves21steps scored in exact-length B4 groups:84physical native scoring calls,336trajectories.',
 predeclared_review={'precision':'Default FA mixed precision accepted. Report actual B2/B4 native score and whole signed-vector differences, zeros/padding, signs, residuals and original metrics; no bitwise gate or hidden fallback.',
 'throughput':'Compare summed time of four existing B1 attributions, two true B2 calls and one true B4 call for the same four examples. Report total and per-example peak/cost, all endpoint/physical counts.',
 'evaluation':'Original deletion and metric AST identical after restoring two suspended original native-score calls. Actual native token tensors only, no placeholders. Same-job B4 scorer for all candidate modes and historical FT score control; actual batch_sizes must be4.',
 'scope':'Four development examples only. Source supports arbitrary leading batch but actual validation limited to tested sizes/lengths. No independent or long-input quality claim.'})
protocol=A/'vendor_fa_batch_protocol_20260907.json';protocol.write_text(json.dumps(p,indent=2),encoding='utf-8')
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_vendor_fa_batch_20260907_v1',
 f'study.py={study}',f'protocol.json={protocol}',*[f'{name}={A/name}' for name in sources],
 '--request',str(A/'vendor_fa_batch_launch_20260907.json')],check=True)
request=json.loads((A/'vendor_fa_batch_launch_20260907.json').read_text());assert len(request['cmd'])<128000
print('Frozen true mini-batch probe:',len(request['cmd']),'chars;113rootF,29attributions,16originalB4curves.')
