"""Frozen original six-curve batch equivalence/cost probe, not a new benchmark."""
import hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent
prior=(A/'pv_interaction_development16_20260907.py').read_text()
driver=prior[:prior.index('\nclass Recorder(')]
driver+=r'''
import numpy as np, zipfile
from original_ft_batched_evaluation import evaluate_requests
for name,digest in p['sources'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
parent_path=Path(p['required_parent']);assert hashlib.sha256(parent_path.read_bytes()).hexdigest()==p['required_parent_sha256']
parent=json.loads(parent_path.read_bytes());assert parent['status']=='complete'
assert report['checkpoint_before']==parent['checkpoint_before']==parent['checkpoint_after']
import flash_attn.flash_attn_interface as fa_native
import transformers.integrations.flash_attention as fa_adapter
import transformers.utils.generic as generic
def native_sources():
    return {str(Path(m.__file__)):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in [fa_native,fa_adapter,fa_native.flash_attn_cuda,generic]}
assert native_sources()==parent['native_sources_before']==parent['native_sources_after']
original_methods={name:type(m).forward for name,m in model.named_modules()}
def audit_methods():
    for name,m in model.named_modules():
        assert type(m).forward is original_methods[name]
        assert getattr(m.forward,'__func__',None) is original_methods[name] and 'forward' not in m.__dict__
    assert hooks()==0
evaluator=runner.llm_attr_eval.LLMAttributionEvaluator(model,tokenizer)
original_score_method=runner.llm_attr_eval.LLMAttributionEvaluator.compute_logprob_response_given_prompt
def save():
    t=HERE/'results.partial';t.write_text(json.dumps(report,indent=2));t.replace(HERE/'results.json')
report.update(scope=p['purpose'],native_sources_before=native_sources(),physical_evaluation_forwards=0,
 evaluation_trajectories=0,native_decoder_layer_calls=0,native_decoder_layer_trajectories=0,
 model_backwards=0,attribution_calls=0,attempts=[])
def make_requests(row,methods):
    ids=torch.tensor(row['input_ids'],device=model.device,dtype=torch.long)[None]
    prompt=ids[:,:row['prompt_len']];response=ids[:,row['prompt_len']:]
    requests={}
    for method in methods:
        for step,mask in enumerate(row['evaluation_masks'][method]):
            value=prompt.clone()
            positions=[row['user_positions'][j] for j in mask]
            if positions:value[0,positions]=tokenizer.eos_token_id
            requests[(method,step)]=(value,response)
    assert len(requests)==42
    return requests
def call(row,methods,batch_size,warm=False):
    assert evaluator.compute_logprob_response_given_prompt.__func__ is original_score_method
    audit_methods();torch.cuda.empty_cache();cost={};done=False
    attempt={'dataset':row['dataset'],'idx':row['idx'],'batch_size':batch_size,'warm':warm,'complete':False}
    report['attempts'].append(attempt)
    try:
        with measured() as cost:
            requests=make_requests(row,methods)
            if warm:requests=dict(list(requests.items())[:batch_size])
            values,counts=evaluate_requests(evaluator,requests,batch_size)
        done=True;attempt.update(complete=True,cost=cost,scheduler_counts=counts)
        assert counts['physical_evaluation_forwards']==cost['native_forwards']
        assert counts['evaluation_trajectories']==cost['native_forward_trajectories']
        assert cost['extra_replay_calls']==cost['extra_replay_trajectories']==cost['vjps']==0
        assert cost['native_decoder_layer_calls']==36*cost['native_forwards']
        assert cost['native_decoder_layer_trajectories']==36*cost['native_forward_trajectories']
        return {m:[values[(m,i)] for i in range(21)] for m in methods} if not warm else None,dict(cost)
    finally:
        attempt['cost']=dict(cost)
        for src,dst in [('native_forwards','physical_evaluation_forwards'),('native_forward_trajectories','evaluation_trajectories'),
                        ('native_decoder_layer_calls','native_decoder_layer_calls'),('native_decoder_layer_trajectories','native_decoder_layer_trajectories')]:
            report[dst]+=cost.get(src,0)
        save();audit_methods()
save()
try:
    # Original benchmark evaluator/backend unchanged for every batch size.
    model.set_attn_implementation(p['official_evaluation_backend']);audit_methods()
    for case_number,(dataset,index) in enumerate(p['selection']):
        row=next(r for r in parent['records'] if (r['dataset'],r['idx'])==(dataset,index))
        methods=['strong_secant_pv_content_P1',p['FT_curve_control'][dataset]]
        record={'dataset':dataset,'idx':index,'methods':methods,'parent_input_ids_sha256':hashlib.sha256(json.dumps(row['input_ids']).encode()).hexdigest(),
                'prompt_len':row['prompt_len'],'N':len(row['input_ids']),'runs':[],
                'original_curves':{m:row['metrics'][m]['raw_curve'] for m in methods}}
        report['records'].append(record)
        for batch_size in p['batch_sizes']:call(row,methods,batch_size,warm=True)
        for repeat in range(p['measured_repeats']):
            sizes=p['batch_sizes'];offset=(case_number+repeat)%len(sizes);order=sizes[offset:]+sizes[:offset]
            for batch_size in order:
                curves,cost=call(row,methods,batch_size)
                record['runs'].append({'repeat':repeat,'batch_size':batch_size,'cost':cost,'curves':curves})
                save()
        print('BATCH_CASE_DONE',dataset,index,flush=True)
    for key,value in p['budget'].items():assert report[key]==value,(key,report[key],value)
    report['checkpoint_after']=checkpoint_receipt();assert report['checkpoint_after']==report['checkpoint_before']
    report['native_sources_after']=native_sources();assert report['native_sources_after']==report['native_sources_before']
    report['status']='complete';audit_methods()
except Exception:
    report['status']='failed';report['error']=traceback.format_exc();raise
finally:
    report['elapsed_seconds']=time.time()-entry_started;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','original_ft_batched_evaluation.py','results.json']:
            z.write(HERE/name,name)
'''
study=A/'original_ft_batch_probe_20260907.py';study.write_text(driver,encoding='utf-8')
p=json.loads((A/'pv_interaction_development16_protocol_20260907.json').read_text())
p={k:p[k] for k in ['checkpoint','checkpoint_receipt','checkpoint_receipt_sha256','native_model_source_sha256',
 'official_normalized_sources','official_evaluation_backend','maximum_queue_seconds']}
p.update(purpose='Original FT batch evaluation equivalence and actual throughput on frozen P1 and strongest-RISE FT curves. Same untouched original evaluator, model, inputs/masks, precision and eager backend. No attribution change or new benchmark.',
 study_sha256=hashlib.sha256(study.read_bytes()).hexdigest(),
 sources={'original_ft_batched_evaluation.py':hashlib.sha256((A/'original_ft_batched_evaluation.py').read_bytes()).hexdigest()},
 required_parent='${ARTIFACT_ROOT}/codex_pv_interaction_development16_20260907_v1/results.json',
 required_parent_sha256='679842f0ccfc32769a416868f6c08f70ddffff84298b80b02f02ca9da1457476',
 selection=[['niah_mq_q2',0],['niah_mq_q2',2],['morehopqa',0]],batch_sizes=[1,2,4],measured_repeats=2,
 FT_curve_control={'niah_mq_q2':'flashtrace_legacy_hop0','morehopqa':'flashtrace_both_hop2'},
 wait_for_pid=106350,wait_for_script='codex_vendor_fa_source_gpu_language_20260907_v1',
 budget={'physical_evaluation_forwards':453,'evaluation_trajectories':777,'native_decoder_layer_calls':16308,'native_decoder_layer_trajectories':27972,'model_backwards':0,'attribution_calls':0},
 diagnostic_tolerances={'max_raw_curve_error_absolute_or_relative':'max(0.5, 0.001*abs(single_case_curve_value))','max_RISE_absolute_difference':0.002,'max_MAS_absolute_difference':0.002},
 acceptance_scope='Diagnose original-curve numerical equivalence before using batch execution in new comparisons; failure remains failure, no relaxed thresholds. Repeats, warmups, failed/OOM calls all charged. Select batch only after complete evidence, not estimated physical-call reduction.')
protocol=A/'original_ft_batch_probe_protocol_20260907.json';protocol.write_text(json.dumps(p,indent=2),encoding='utf-8')
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_original_ft_batch_probe_20260907_v1',
 f'study.py={study}',f'protocol.json={protocol}',f'original_ft_batched_evaluation.py={A/"original_ft_batched_evaluation.py"}',
 '--request',str(A/'original_ft_batch_probe_launch_20260907.json')],check=True)
