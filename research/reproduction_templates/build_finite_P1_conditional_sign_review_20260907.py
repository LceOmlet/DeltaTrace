"""Freeze native conditional sign checks on development cases, no quality retuning."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
parent_file=A/'snapshot${ARTIFACT_ROOT}/codex_vendor_fa_development16_20260907_v1/results.json'
assert sha(parent_file)=='1598dfb8586b771e276f98ce8ade2b09b8368eb230773821a19e74e51c838bc6'
parent=json.loads(parent_file.read_text());rng=np.random.default_rng(730907);selection=[]
for row in parent['records']:
    value=next(r['result'] for r in row['runs'] if r['mode']=='finite' and r['repeat']==1)
    x=np.asarray(value['signed_full_sequence']);eligible=row['eligible_positions']
    positive=sorted((j for j in eligible if x[j]>0),key=lambda j:(-x[j],j))[:8]
    negative=sorted((j for j in eligible if x[j]<0),key=lambda j:(x[j],j))[:8]
    assert len(positive)==len(negative)==8
    rest=sorted(set(eligible)-set(positive)-set(negative));random=rng.choice(rest,8,replace=False).tolist()
    selection.append({'dataset':row['dataset'],'idx':row['idx'],'input_ids_sha256':row['input_ids_sha256'],
        'selected':[{'position':int(j),'selection':kind,'finite_P1_score':float(x[j])} for kind,indices in [('positive',positive),('negative',negative),('random',random)] for j in indices]})
base=(A/'batch_endpoint_diagnostic_20260907.py').read_text();driver=base[:base.index('\nimport zipfile\n')]
driver+=r'''
import zipfile
parent_file=Path(p['parent']);assert hashlib.sha256(parent_file.read_bytes()).hexdigest()==p['parent_sha256']
parent=json.loads(parent_file.read_text());assert parent['status']=='complete'
assert report['checkpoint_before']==parent['checkpoint_before']==parent['checkpoint_after']
def native_sources():return {name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in parent['native_sources_before']}
assert native_sources()==parent['native_sources_before']==parent['native_sources_after']
report.update(scope=p['purpose'],native_sources_before=native_sources(),native_forwards=0,trajectories=0,native_vjps=0,quality_queries=0,records=[])
def save():
    tmp=HERE/'results.partial';tmp.write_text(json.dumps(report,separators=(',',':')));tmp.replace(HERE/'results.json')
def score(ids,plen):
    assert ids.shape[0]==4
    selected=torch.arange(plen-1,ids.shape[1]-1,device=ids.device)
    with measured() as cost,torch.no_grad():
        output=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,output_attentions=False,logits_to_keep=selected)
        values=output.logits.float().log_softmax(-1).gather(2,ids[:,plen:,None]).squeeze(-1)
        scores=values.double().sum(-1)
    assert cost['native_forwards']==1 and cost['native_forward_trajectories']==4 and cost['native_decoder_layer_calls']==36 and cost['extra_replay_calls']==0
    assert torch.isfinite(values).all()
    report['native_forwards']+=1;report['trajectories']+=4
    return {'scores32_sum64':scores.cpu().tolist(),'target_logprobs32':values.cpu().tolist(),'cost':cost}
save()
try:
    model.set_attn_implementation('flash_attention_2');torch.backends.cuda.matmul.allow_tf32=False
    for case_no,selection in enumerate(p['selection']):
        source=next(r for r in parent['records'] if (r['dataset'],r['idx'])==(selection['dataset'],selection['idx']))
        assert source['input_ids_sha256']==selection['input_ids_sha256']
        selected=selection['selected'];positions=[s['position'] for s in selected]
        assert len(positions)==len(set(positions))==24 and set(positions)<=set(source['eligible_positions'])
        finite=next(r['result'] for r in source['runs'] if r['mode']=='finite' and r['repeat']==1)
        assert all(s['finite_P1_score']==finite['signed_full_sequence'][s['position']] for s in selected)
        ids=torch.tensor([source['input_ids']],dtype=torch.long,device='cuda:0');plen=source['prompt_len']
        clean=ids.repeat(4,1);eos=clean.clone();eos[:,torch.tensor(source['eligible_positions'],device=ids.device)]=tokenizer.eos_token_id
        row={'dataset':source['dataset'],'idx':source['idx'],'input_ids_sha256':source['input_ids_sha256'],'N':ids.shape[1],'prompt_len':plen,
            'selected':selected,'baseline_repeats':{},'interventions':[]};report['records'].append(row)
        for name,baseline in [('clean',clean),('eos',eos)]:
            report['active']=[case_no,name,'baseline_first'];save();row['baseline_repeats'][name]=[score(baseline,plen)];save()
            for offset in range(0,24,4):
                group=positions[offset:offset+4];changed=baseline.clone()
                for lane,j in enumerate(group):changed[lane,j]=tokenizer.eos_token_id if name=='clean' else clean[lane,j]
                assert torch.equal(changed[:,plen:],clean[:,plen:])
                for lane,j in enumerate(group):
                    actual=(changed[lane]!=baseline[lane]).nonzero().flatten().tolist();assert actual==[j]
                value=score(changed,plen)
                base_scores=row['baseline_repeats'][name][0]['scores32_sum64']
                effects=[(a-b if name=='clean' else b-a) for a,b in zip(base_scores,value['scores32_sum64'])]
                row['interventions'].append({'background':name,'positions':group,'lanes':[0,1,2,3],'result':value,'conditional_effects':effects,
                    'actual_changed_positions_checked':True,'original_target_preserved':True})
                report['active']=[case_no,name,offset];save()
            row['baseline_repeats'][name].append(score(baseline,plen));save()
        if case_no==0:
            with torch.profiler.profile(activities=list(torch.profiler.supported_activities())) as prof:extra=score(clean,plen)
            trace='native_conditional_FA_trace.json';prof.export_chrome_trace(str(HERE/trace))
            row['profile']={'trace':trace,'sha256':hashlib.sha256((HERE/trace).read_bytes()).hexdigest(),'result':extra};del prof,extra
        print('CONDITIONAL_SIGN_DONE',row['dataset'],row['idx'],flush=True)
        del ids,clean,eos,changed
    report['checkpoint_after']=checkpoint_receipt();assert report['checkpoint_before']==report['checkpoint_after']
    report['native_sources_after']=native_sources();assert report['native_sources_before']==report['native_sources_after']
    for k,v in p['budget'].items():assert report[k]==v
    report['status']='complete';report.pop('active',None)
except Exception:
    report['status']='failed';report['error']=traceback.format_exc();raise
finally:
    report['elapsed_seconds']=time.time()-entry_started;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','results.json']+[f.name for f in HERE.glob('*_FA_trace.json')]:z.write(HERE/name,name)
'''
ast.parse(driver);study=A/'finite_P1_conditional_sign_review_20260907.py';study.write_text(driver)
base_p=json.loads((A/'batch_endpoint_diagnostic_protocol_20260907.json').read_text())
p={k:v for k,v in base_p.items() if k in ['checkpoint','checkpoint_receipt','checkpoint_receipt_sha256','native_model_source_sha256','official_normalized_sources','maximum_queue_seconds']}
p.update(study_sha256=sha(study),wait_for_pid=144048,wait_for_script='${ARTIFACT_ROOT}/codex_vendor_fa_batch_memory_20260907_v1/study.py',
    purpose='Validate fixed finite-P1 signed token allocations against actual native G32 conditional finite effects, on original16 development examples only. Select8largest positive,8most negative,8uniform remaining tokens without gold/curves. Both clean deletion and all-other-eligible-EOS restoration contexts. Matched B4 lanes, original defaultFA/native selected head, full fixed response; no surrogate, model gradients, generation or metric tuning.',
    parent='${ARTIFACT_ROOT}/codex_vendor_fa_development16_20260907_v1/results.json',parent_sha256=sha(parent_file),selection=selection,
    budget={'native_forwards':257,'trajectories':1028,'native_vjps':0,'quality_queries':0},
    baseline_cache_policy='Each actual B4 baseline shared across six B4 interventions with corresponding lane, then repeated at end. Preserve both actual baseline results and all token logprobs. Report drift and lane variation; do not repair values or treat observed drift as a proven universal error bound.',
    interpretation='Reference is s({j}|A)=G(I_A)-G(I_Aunion{j}). clean A=empty; eos A=eligible minus{j}. These are separate conditional signs, not universally additive token effects. This is a development diagnostic, not a substitute benchmark or independent confirmation.')
protocol=A/'finite_P1_conditional_sign_review_protocol_20260907.json';protocol.write_text(json.dumps(p,indent=2))
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_finite_P1_conditional_sign_review_20260907_v1','study.py='+str(study),'protocol.json='+str(protocol),'--request',str(A/'launch_finite_P1_conditional_sign_review_20260907.json')],check=True)
print('Frozen257actual native B4 forwards;384tokens,two conditional contexts;no candidate tuning or quality curves.')
