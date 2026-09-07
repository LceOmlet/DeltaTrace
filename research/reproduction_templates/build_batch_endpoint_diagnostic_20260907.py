"""Freeze a direct-original-evaluator check of repeated identical B4 endpoints."""
import ast, hashlib, json
from pathlib import Path
A=Path(__file__).resolve().parent
base=(A/'vendor_fa_batch_probe_20260907.py').read_text()
prefix=base.split('\nimport zipfile\n',1)[0]
tail=r'''
import zipfile
parent_path=Path(p['batch_parent'])
assert hashlib.sha256(parent_path.read_bytes()).hexdigest()==p['batch_parent_sha256']
parent=json.loads(parent_path.read_text())
assert report['checkpoint_before']==parent['checkpoint_before']==parent['checkpoint_after']
def native_sources():
    return {name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in parent['native_sources_before']}
assert native_sources()==parent['native_sources_before']==parent['native_sources_after']
report.update(scope=p['purpose'],native_sources_before=native_sources(),native_forwards=0,trajectories=0,native_decoder_layer_calls=0)
def save():
    temp=HERE/'results.partial';temp.write_text(json.dumps(report,indent=2));temp.replace(HERE/'results.json')
save()
try:
    torch.backends.cuda.matmul.allow_tf32=False
    model.set_attn_implementation(p['official_evaluation_backend'])
    evaluator=runner.llm_attr_eval.LLMAttributionEvaluator(model,tokenizer)
    assert evaluator.compute_logprob_response_given_prompt.__func__ is type(evaluator).compute_logprob_response_given_prompt
    for dataset,index in p['selection']:
        row=next(r for r in parent['records'] if (r['dataset'],r['idx'])==(dataset,index))
        ids=torch.tensor([row['input_ids']],dtype=torch.long,device='cuda:0')
        prompt=ids[:,:row['prompt_len']];response=ids[:,row['prompt_len']:]
        for endpoint in ['clean','deleted']:
            query=prompt.clone()
            if endpoint=='deleted':query[0,torch.tensor(row['eligible_positions'],device=query.device)]=tokenizer.eos_token_id
            for repeat in range(2):
                for batch in [1,4]:
                    queries=torch.cat([query]*batch);responses=torch.cat([response]*batch)
                    assert all(torch.equal(queries[0],x) for x in queries)
                    assert all(torch.equal(responses[0],x) for x in responses)
                    with measured() as cost,torch.no_grad():
                        value=evaluator.compute_logprob_response_given_prompt(queries,responses)
                    assert value.shape==responses.shape and torch.isfinite(value).all()
                    assert cost['native_forwards']==1 and cost['native_forward_trajectories']==batch and cost['native_decoder_layer_calls']==36 and cost['extra_replay_calls']==0
                    report['native_forwards']+=cost['native_forwards'];report['trajectories']+=cost['native_forward_trajectories'];report['native_decoder_layer_calls']+=cost['native_decoder_layer_calls']
                    report['records'].append({'dataset':dataset,'idx':index,'endpoint':endpoint,'repeat':repeat,'batch':batch,'input_ids_sha256':row['input_ids_sha256'],
                        'identical_inputs_within_batch':True,'token_logprobs':value.cpu().tolist(),'sums':[float(x.sum()) for x in value],'cost':cost})
                    save();print('DIRECT_ENDPOINT',dataset,index,endpoint,repeat,batch,report['records'][-1]['sums'],flush=True)
    for key,value in p['budget'].items():assert report[key]==value
    report['checkpoint_after']=checkpoint_receipt();assert report['checkpoint_after']==report['checkpoint_before']
    report['native_sources_after']=native_sources();assert report['native_sources_after']==report['native_sources_before']
    report['status']='complete'
except Exception:
    report['status']='failed';report['error']=traceback.format_exc();raise
finally:
    save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','results.json']:z.write(HERE/name,name)
'''
study=(prefix+tail).encode();ast.parse(study)
(A/'batch_endpoint_diagnostic_20260907.py').write_bytes(study)
p=json.loads((A/'vendor_fa_batch_protocol_20260907.json').read_text())
p={k:p[k] for k in ['checkpoint','checkpoint_receipt','checkpoint_receipt_sha256','native_model_source_sha256','official_normalized_sources','official_evaluation_backend','maximum_queue_seconds']}
p.update(study_sha256=hashlib.sha256(study).hexdigest(),wait_for_pid=133858,wait_for_script='${ARTIFACT_ROOT}/codex_vendor_fa_batch_20260907_v1/study.py',
    purpose='Direct unchanged original evaluator only: identical inputs repeated within actual B4 and B1. Diagnose batch-row endpoint nonidentity observed in completed original mini-batch evaluation; no attribution, coroutine, scheduler, changed dtype or changed metric.',
    batch_parent='${ARTIFACT_ROOT}/codex_vendor_fa_batch_20260907_v1/results.json',batch_parent_sha256=hashlib.sha256((A/'snapshot${ARTIFACT_ROOT}/codex_vendor_fa_batch_20260907_v1/results.json').read_bytes()).hexdigest(),
    selection=[['niah_mq_q2',0],['morehopqa',1]],budget={'native_forwards':16,'trajectories':40,'native_decoder_layer_calls':576})
(A/'batch_endpoint_diagnostic_protocol_20260907.json').write_text(json.dumps(p,indent=2))
print('Frozen direct endpoint diagnostic:16native forwards,40trajectories,zero attribution.')
