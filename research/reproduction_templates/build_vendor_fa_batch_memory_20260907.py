"""Freeze same-job genuine example-batch ordinary-backward memory references."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=(A/'vendor_fa_batch_probe_20260907.py').read_text()
driver=base[:base.index('\nsave()\ntry:\n')]
driver=driver.replace('json.dumps(report,ensure_ascii=False,indent=2)','json.dumps(report,ensure_ascii=False,separators=(",",":"))')
driver+=r'''
from native_backward_memory_reference_batched import ordinary_batch_input_backward
report.update(ordinary_attempts=[],paired_groups=[],quality_evaluated=False)
def run_ordinary(rows):
    inputs=[(torch.tensor([row['input_ids']],device='cuda:0',dtype=torch.long),row['prompt_len']) for row in rows]
    native_method_audit(True)
    value=ordinary_batch_input_backward(model,inputs,tokenizer.eos_token_id)
    report['ordinary_attempts'].append(value)
    report['native_root_forwards']+=1;report['native_vjps']+=1;report['ordinary_reference_forwards']+=1
    save();return value
save()
try:
    model.set_attn_implementation('flash_attention_2');native_method_audit(True)
    rows=[];examples=[]
    for dataset,index in p['selection']:
        path=ROOT/f'exp/exp2/data/{dataset}.jsonl';assert hashlib.sha256(path.read_bytes()).hexdigest()==p['official_cache_sha256'][dataset]
        examples.append(runner.ds_utils.load_cached(path)[index])
        rows.append(next(r for r in parent['records'] if (r['dataset'],r['idx'])==(dataset,index)))
    report['records']=[{k:r[k] for k in ['dataset','idx','input_ids','input_ids_sha256','prompt_len','eligible_positions']} for r in rows]
    groups=[[0],[1],[2],[3],[0,1],[2,3],[0,1,2,3]]
    for repeat in range(4):
        for group_index,group in enumerate(groups):
            order=['finite','ordinary'] if (repeat+group_index)%2==0 else ['ordinary','finite']
            pair={'repeat':repeat,'warmup':repeat==0,'indices':group};report['paired_groups'].append(pair)
            for mode in order:
                native_method_audit(True);report['active']=[repeat,group,mode];save()
                if mode=='ordinary':value=run_ordinary([rows[i] for i in group])
                elif len(group)==1:value=strong_run(examples[group[0]],rows[group[0]],capture_mode='finite')
                else:value=batch_run([examples[i] for i in group],[rows[i] for i in group])
                pair[mode]=value;save();print('SAME_BATCH_MEMORY',repeat,group,mode,value['seconds'],value['peak_allocated_bytes'],flush=True)
    report['profiles']={}
    for mode in ['ordinary','finite']:
        native_method_audit(True);report['active']=['profile',mode];save()
        with torch.profiler.profile(activities=list(torch.profiler.supported_activities()),record_shapes=True) as prof:
            value=run_ordinary(rows) if mode=='ordinary' else batch_run(examples,rows)
        trace=mode+'_batch4_FA_trace.json';prof.export_chrome_trace(str(HERE/trace))
        names=[e.name for e in prof.events() if str(e.device_type)=='DeviceType.CUDA']
        report['profiles'][mode]={'trace':trace,'sha256':hashlib.sha256((HERE/trace).read_bytes()).hexdigest(),'result':value,
            'device_kernel_names':names,'native_FA_forwards':sum('flash_fwd_kernel' in n for n in names),
            'finite_FA_kernels':sum('deltatrace_fa_finite_p1_kernel' in n for n in names),
            'native_FA_backward_named_kernels':sum('flash_bwd' in n for n in names)}
        assert report['profiles'][mode]['native_FA_forwards']==(36 if mode=='ordinary' else 108)
        assert report['profiles'][mode]['finite_FA_kernels']==(0 if mode=='ordinary' else 108)
        if mode=='ordinary':assert report['profiles'][mode]['native_FA_backward_named_kernels']>=36
        else:assert report['profiles'][mode]['native_FA_backward_named_kernels']==0
        del prof,value,names;save()
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
ast.parse(driver);study=A/'vendor_fa_batch_memory_20260907.py';study.write_text(driver,encoding='utf-8')
p=json.loads((A/'vendor_fa_batch_protocol_20260907.json').read_text())
p['sources']['native_backward_memory_reference_batched.py']=sha(A/'native_backward_memory_reference_batched.py')
p.update(study_sha256=sha(study),wait_for_pid=138481,wait_for_script='${ARTIFACT_ROOT}/codex_vendor_fa_original_long_cost_20260907_v1/study.py',
    purpose='Same-job actual B1/B2/B4 finite FA attribution versus ordinary native FA input backward with identical per-example original fixed response targets, native selected output head and frozen parameters. Four original short cases,one warmup/three paired repeats,actual B4 profiles. No synthetic memory inflation,quality retuning or native attention replacement.',
    budget={'native_root_forwards':58,'native_vjps':29,'evaluation_forwards':0,'ft_attribution_forwards':0,'ordinary_reference_forwards':29,
       'native_attribution_forwards':29,'manual_passes':29,'extra_layer_replay_calls':1044,'extra_native_fa_attention_calls':1044,
       'native_attribution_endpoint_trajectories':104,'extra_layer_replay_endpoint_trajectories':3744,'finite_FA_calls_attempted':1044,'finite_FA_calls_enqueued':1044})
protocol=A/'vendor_fa_batch_memory_protocol_20260907.json';protocol.write_text(json.dumps(p,indent=2))
files=['study.py='+str(study),'protocol.json='+str(protocol)]+[name+'='+str(A/name) for name in p['sources']]
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_vendor_fa_batch_memory_20260907_v1',*files,'--request',str(A/'launch_vendor_fa_batch_memory_20260907.json')],check=True)
print('Frozen 58 root forwards,29 native input VJPs,29 finite passes; same-job B1/B2/B4 references,zero quality queries.')
