"""Freeze only original rescoring of saved current-C vectors, without new DT calls."""
import ast,base64,copy,hashlib,json,shlex,zlib
from pathlib import Path
A=Path(__file__).resolve().parent;R=A.parent/'DeltaTrace'
sha=lambda b:hashlib.sha256(b).hexdigest()
pilot=A/'snapshot${ARTIFACT_ROOT}/codex_dt_GDN1_K_whole_pilot_20260909_v1'
base=json.loads((pilot/'protocol.json').read_bytes())
source=R/'research/reproduction_templates/dt_GDN1_K_remaining_20260909.py'
text=source.read_text(encoding='utf-8')
# Reuse the reviewed study's source/input/model preparation, not a model or FT implementation.
text=text[:text.index('    original_fa=VendorFAFiniteP1BF16D256')]
text=text.replace(text[:text.index('import os,sys')],'"""User-authorized positive-score original RISE/MAS; saved fixed8 current C,0DT168scores."""\n')
for line in [
    '    from qwen35_answer_finite import PackedAnswerTargets\n',
    '    from qwen35_dense_finite_runner import Qwen35DenseFiniteRunner\n',
    '    from vendor_fa_finite_bf16_d256 import VendorFAFiniteP1BF16D256\n',
    '    from layer0_fla_endpoint_average_20260908 import Layer0FLAEndpointAverage\n']:
    assert line in text;text=text.replace(line,'')
text=text.replace('from finite_fla_gpu import make_compiled_finite_pullback,verify_native_sources','from finite_fla_gpu import verify_native_sources')
text=text.replace('from normal_finite_study_utils_20260908 import CountFinite,check_counts,deletion_audit','from normal_finite_study_utils_20260908 import deletion_audit\n    from metric_score_view import positive_metric_scores')
start=text.index('    class BudgetedFinite(');end=text.index('    assert sha(Path(native.__file__)',start)
text=text[:start]+'''    assert p['quality_cases']==[f'{d}_{i}' for d in ['niah_mq_q2','morehopqa'] for i in range(4)]
    assert p['case_indices']==[[k.rsplit('_',1)[0],int(k.rsplit('_',1)[1])] for k in p['quality_cases']]
    r['scope']='Same current-C signed vectors; positive metric view explicitly requested by user. No new attribution or FT.'
'''+text[end:]
text=text.replace('Frozen4DT100FLA32FA84score input-supported GDN1 K budget expired.','Frozen fixed8 positive-metric 168-score budget expired.')
text+='''    evaluator=LLMAttributionEvaluator(model,tokenizer)
    for key in p['quality_cases']:
        case=cases[key];info=case['input'];src=p['score_sources'][key]
        assert sha(Path(src['vectors_path']).read_bytes())==src['vectors_sha256']
        assert sha(Path(src['results_path']).read_bytes())==src['results_sha256']
        with np.load(src['vectors_path'],allow_pickle=False) as prior:
            signed=torch.from_numpy(prior[key+'_control_evaluated'].copy()).float()
        assert signed.shape==(info['prompt_length'],) and bool(torch.isfinite(signed).all())
        assert sha(signed.numpy().tobytes())==src['signed_FP32_sha256']
        score=positive_metric_scores(signed)
        assert bool(score.eq(signed.clamp_min(0)).all()) and bool(score.ge(0).all())
        audit=deletion_audit(score,case['ids'],info['keep'],tokenizer.eos_token_id)
        vectors[key+'_signed']=signed.numpy();vectors[key+'_positive']=score.numpy()
        view=FixedInputMetricView(evaluator,case['record']['prompt'])
        assert view.compute_logprob_response_given_prompt.__func__ is LLMAttributionEvaluator.compute_logprob_response_given_prompt
        needle_args=dict(keep_prompt_token_indices=info['keep'],gold_prompt_token_indices=case['gold'],top_fraction=.1)
        needle=float(evaluate_attr_recovery_skip_tokens(score[None],**needle_args)) if case['gold'] else None
        signed_needle=float(evaluate_attr_recovery_skip_tokens(signed[None],**needle_args)) if case['gold'] else None
        assert needle==signed_needle==src['historical_needle']
        curve={'input_receipts':[],'returned_forwards':0,'needle':needle,'signed_input_needle':signed_needle}
        row={'input':info,'gold':case['gold'],'source':src,'score_view':'positive_part_before_original_metrics',
             'curve':curve,'deletion_audit':audit,'signed_stats':{'net':float(signed.double().sum()),'negative_count':int((signed<0).sum())},
             'historical_FT_reference':p['historical_FT_references'][key]}
        r['cases'][key]=row
        def before_score(_module,args,kw):
            step=len(curve['input_receipts']);assert step<21 and r['scorer_entered']<168
            x=kw['input_ids'].detach().cpu();assert x.shape==(1,len(case['ids'])) and bool(kw['attention_mask'].eq(1).all())
            changed=(x[0]!=case['ids']).nonzero().flatten().tolist()
            receipt={'input_sha256':sha(x.numpy().tobytes()),'deleted_positions':changed}
            assert receipt==audit['input_receipts'][step]
            curve['input_receipts'].append(receipt);r['scorer_entered']+=1
        def after_score(_module,args,output):
            if output is not None:
                assert output.logits.dtype==torch.bfloat16;r['scorer_returned']+=1;curve['returned_forwards']+=1
        def observe(frame,event,arg):
            if frame.f_code is faithfulness_test_skip_tokens.__code__ and event=='return' and arg is not None:
                loc=frame.f_locals
                for name in ('scores','density','normalized_model_response','alignment_penalty','corrected_scores'):
                    curve[name]=np.asarray(loc[name]).copy().tolist()
                curve['sorted_keep']=[int(x) for x in loc['sorted_keep']];curve['attr_sum']=float(loc['attr_sum'])
        handles=[model.register_forward_pre_hook(before_score,with_kwargs=True),model.register_forward_hook(after_score,always_call=True)]
        assert sys.getprofile() is None;r['status']='positive_original_curve_'+key;save()
        torch.cuda.reset_peak_memory_stats();sys.setprofile(observe)
        try:
            with torch.no_grad():returned=timed('positive_original_curve_'+key,lambda:faithfulness_test_skip_tokens(view,score[None],
                case['record']['prompt'],case['record']['target'],keep_prompt_token_indices=info['keep'],user_prompt_indices=list(range(info['prompt_length'])),k=20))
        finally:
            sys.setprofile(None)
            for handle in handles:handle.remove()
            handles=[]
        curve['return_metrics']=[float(x) for x in returned]
        assert curve['returned_forwards']==21 and curve['sorted_keep']==audit['sorted_keep']
        assert all(np.isfinite(curve[name]).all() for name in ('scores','density','normalized_model_response','alignment_penalty','corrected_scores','return_metrics'))
        row['peak_allocated_full_model_resident']=torch.cuda.max_memory_allocated()
        row['historical_signed_curve_endpoint_drift']=[curve['scores'][i]-src['historical_signed_scores'][i] for i in (0,20)]
        curve['status']='complete';save()
    assert r['scorer_entered']==r['scorer_returned']==168 and r['DT_entered']==r['DT_returned']==0
    np.savez_compressed(A/'vectors.npz',**vectors);r['vectors_sha256']=sha((A/'vectors.npz').read_bytes())
    r['sources_after']=sources();r['weight_stats_after']=weights()
    assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
    r['status']='positive_metric_view_fixed8_0DT168originalscores_complete'
except Exception:r['status']='failed';r['error']=traceback.format_exc()
finally:
    sys.setprofile(None);signal.alarm(0)
    for handle in handles:handle.remove()
    r['seconds']=time.perf_counter()-started
    r['cost_scope']='1original model load,1NI0 eager initialization,8original B1 metric curves=168scorers;0DT,0FT,0generation. No attribution speed or new batch claim.'
    save()
    with zipfile.ZipFile(A/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in [*p['files_sha256'],'protocol.json','results.json','vectors.npz']:
            if (A/name).is_file():archive.write(A/name,name)
    print(json.dumps({'status':r['status'],'seconds':r['seconds'],'scorer_returned':r['scorer_returned'],'error':r.get('error')}),flush=True)
'''
ast.parse(text)
study=R/'research/reproduction_templates/dt_positive_metrics_fixed8_20260909.py'
if study.exists():assert study.read_text(encoding='utf-8')==text
else:study.write_text(text,encoding='utf-8')
common={'study.py':study.read_bytes()}
for name in ['official_span_mapping.py','fixed_input_metric_view.py','normal_finite_study_utils_20260908.py','finite_fla_gpu.py']:
    raw=(pilot/name).read_bytes();assert sha(raw)==base['files_sha256'][name];common[name]=raw
common['metric_score_view.py']=(R/'research/runtime/metric_score_view.py').read_bytes()
records={};refs={};score_sources={};protected=[]
jobs=['dt_GDN1_K_whole_pilot_20260909_v1']+[f'dt_GDN1_K_remaining_20260909_s{i}_v1' for i in range(3)]
import numpy as np
for job in jobs:
    directory=A/'snapshot/tmp'/('codex_'+job)
    pp=json.loads((directory/'protocol.json').read_bytes());rr=json.loads((directory/'results.json').read_bytes());rec=json.loads((directory/'terminal_receipt.json').read_bytes())
    assert rec['proc_exists'] is False and sha((directory/'results.json').read_bytes())==rec['files']['results.json']['sha256']
    for k in ['checkpoint','cache_paths','cache_hashes','checkpoint_config_tokenizer_sha256','native_model_sha256','official_source_blob_sha1','span_source_sha256']:assert pp[k]==base[k]
    vp=directory/'vectors.npz';assert sha(vp.read_bytes())==rr['vectors_sha256']
    with np.load(vp,allow_pickle=False) as vec:
        for key,case in rr['cases'].items():
            source_curve=case['curves']['control'];record=copy.deepcopy(pp['fixed_records'][key]);record.update(expected_input=case['input'],expected_gold=case['gold']);records[key]=record
            refs[key]=pp['historical_FT_references'][key]
            score_sources[key]={'vectors_path':'/tmp/'+directory.name+'/vectors.npz','vectors_sha256':sha(vp.read_bytes()),
                'results_path':'/tmp/'+directory.name+'/results.json','results_sha256':sha((directory/'results.json').read_bytes()),
                'signed_FP32_sha256':sha(vec[key+'_control_evaluated'].astype(np.float32).tobytes()),
                'historical_signed_scores':source_curve['scores'],'historical_signed_metrics':source_curve['return_metrics'],
                'historical_needle':source_curve['needle']}
    for name in ['protocol.json','results.json','vectors.npz']:
        protected.append({'path':'/tmp/'+directory.name+'/'+name,'sha256':sha((directory/name).read_bytes())})
needed={node.slice.value for node in ast.walk(ast.parse(text)) if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name) and node.value.id=='p' and isinstance(node.slice,ast.Constant) and isinstance(node.slice.value,str)}
p={k:copy.deepcopy(base[k]) for k in needed|{'FT_commit','method_target_semantics'} if k in base}
keys=[f'{d}_{i}' for d in ['niah_mq_q2','morehopqa'] for i in range(4)]
p.update(quality_cases=keys,case_indices=[[k.rsplit('_',1)[0],int(k.rsplit('_',1)[1])] for k in keys],fixed_records=records,historical_FT_references=refs,
    score_sources=score_sources,protected_sources=protected,files_sha256={n:sha(b) for n,b in common.items()},
    budget={'wall_time_seconds':360,'model_loads':1,'native_eager_initializations':1,'DT_calls':0,'FT_calls':0,'scoring_forwards':168,'generation_calls':0},
    user_instruction='For Qwen3.5 RISE/MAS also remove negative values. Preserve original signed output and original FT; use the same positive score convention as historical Qwen3.',
    scope='Existing fixed8 current-C score vectors only; author original curves newly scored under positive ordering,0newDT. Prior K candidate not evaluated.',
    method_score_view='positive_part_before_original_RISE_MAS',
    attribution_source_profile='Unchanged current C:layer0 symmetric normgate and FLA endpoint average;allFA P1.',
    decision='Finish fixed8 unless input/source/numerical/nonfinite/runtime failure. Report all cases, endpoint drift and needle. No new method or per-layer candidate.',
    prepared_from_source_sha256=sha(source.read_bytes()))
for k in ['segment_index','call_schedule','quality_schedule','completed_pilot','fixed_segment_plan','executed_subset']:p.pop(k,None)
remote='${ARTIFACT_ROOT}/codex_dt_positive_metrics_fixed8_20260909_v1';python='${ARTIFACT_ROOT}/codex_qwen35_isolated_import_20260908_v1/env/bin/python'
files=dict(common);files['protocol.json']=json.dumps(p,indent=2).encode()
blob=base64.b64encode(zlib.compress(json.dumps({n:base64.b64encode(b).decode() for n,b in files.items()}).encode())).decode()
loader='import pathlib,subprocess,json,base64,zlib;d=pathlib.Path('+repr(remote)+');d.mkdir(exist_ok=False);files=json.loads(zlib.decompress(base64.b64decode('+repr(blob)+')));[(d/k).write_bytes(base64.b64decode(v)) for k,v in files.items()];f=(d/"driver.log").open("w");j=subprocess.Popen(['+repr(python)+',"-B",str(d/"study.py")],stdout=f,stderr=subprocess.STDOUT,start_new_session=True);(d/"pid").write_text(str(j.pid));print(json.dumps({"pid":j.pid,"directory":str(d)}))'
(A/'dt_positive_metrics_fixed8_protocol_20260909.json').write_bytes(files['protocol.json'])
(A/'launch_dt_positive_metrics_fixed8_20260909.json').write_text(json.dumps({'cmd':python+' -B -c '+shlex.quote(loader),'timeout':10}),encoding='utf-8')
print(json.dumps({'status':'prepared_not_launched','protocol_sha256':sha(files['protocol.json']),'cases':keys,'sources':len(common),'model_loads':1,'DT':0,'original_scores':168}))
