"""Freeze a fair seq-only original FT control using original author functions."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
base=(A/'finite_FA_FT_cost16_20260907.py').read_text();assert base.count('\nsave()\ntry:')==1
source=base[:base.index('\nsave()\ntry:')]
assert source.count('json.dumps(report,ensure_ascii=False,indent=2)')==1
source=source.replace('json.dumps(report,ensure_ascii=False,indent=2)',"json.dumps(report,ensure_ascii=False,separators=(',',':'))")
source+=r'''
def original_seq_only(ex,expected):
    model.set_attn_implementation(p['official_evaluation_backend']);native_method_audit(True);torch.cuda.empty_cache()
    positions=expected['user_positions'];keep=expected['keep_local_indices']
    with measured() as cost:
        engine=both.LLMIFRAttributionBoth(model,tokenizer,chunk_tokens=128,sink_chunk_tokens=32,show_progress=False)
        value=engine.calculate_ifr_multi_hop_both(ex.prompt,target=ex.target,sink_span=tuple(ex.sink_span),thinking_span=tuple(ex.thinking_span),n_hops=1)
        # These are the original get_all_token_attrs seq-expression and the
        # already used prompt-side aggregation, calling the original method.
        seq=value.normalize_sum_to_one(value.attribution_matrix)
        score=seq[:,:len(positions)].sum(0).cpu().float().tolist()
        assert engine.user_prompt_indices==positions and both.keep_token_indices(list(value.prompt_tokens))==keep
        del seq,value,engine
    assert cost['native_forwards']==1 and cost['native_forward_trajectories']==1 and cost['extra_replay_calls']==0
    assert cost['actual_root_input_ids']==[[expected['input_ids']]]
    report['native_root_forwards']+=1;report['ft_attribution_forwards']+=1;report['fresh_attribution_calls']+=1
    restore_author_bindings();model.set_attn_implementation('flash_attention_2');native_method_audit(True)
    return {'score':score,'end_to_end_cost':cost}

def invoke(ex,expected,mode):
    return original_seq_only(ex,expected) if mode=='seq_1' else run_mode(ex,expected,mode)

save()
try:
    modes=p['methods']
    for case_number,(dataset,index) in enumerate(p['selection']):
        path=ROOT/f'exp/exp2/data/{dataset}.jsonl';assert hashlib.sha256(path.read_bytes()).hexdigest()==p['official_cache_sha256'][dataset]
        ex=runner.ds_utils.load_cached(path)[index]
        expected=next(r for r in parent['records'] if (r['dataset'],r['idx'])==(dataset,index))
        qp=next(r for r in quality_parent['records'] if (r['dataset'],r['idx'])==(dataset,index))
        row={k:expected[k] for k in ['dataset','idx','input_ids_sha256','prompt_len','user_positions','keep_local_indices','eligible_positions']}
        row.update(N=len(expected['input_ids']),runs=[],profiles={},historical_quality_reuse={})
        report['records'].append(row)
        for repeat in range(4):
            offset=(case_number+repeat)%3;order=modes[offset:]+modes[:offset]
            for mode in order:
                report['active']=[dataset,index,repeat,mode];save();result=invoke(ex,expected,mode)
                row['runs'].append({'repeat':repeat,'warmup':repeat==0,'mode':mode,'result':result})
                save();print('SEQ_FAIR_COST',dataset,index,repeat,mode,result['end_to_end_cost']['seconds'],flush=True)
        if case_number==0:
            for mode in ['both_1','seq_1']:
                observed={}
                def observe(frame,event,arg):
                    if event=='call' and frame.f_code.co_filename.startswith(str(ROOT)):
                        name=frame.f_code.co_name
                        if name in ['get_all_token_attrs','normalize_sum_to_one','compute_CAGE_token_attr']:
                            observed[name]=observed.get(name,0)+1
                try:
                    sys.setprofile(observe);extra=invoke(ex,expected,mode)
                finally:sys.setprofile(None)
                row['profiles'][mode]={'result':extra,'original_python_calls':observed,'scope':'Extra passive Python call observation, excluded from normal timing medians; real original FT call counted.'}
                del extra;save()
        for mode in modes:
            chosen=next(x['result']['score'] for x in row['runs'] if x['mode']==mode and x['repeat']==1)
            prior=qp['scores']['finite'] if mode=='finite' else expected['scores']['flashtrace_both_hop1']
            metric=qp['metrics']['finite'] if mode=='finite' else expected['metrics']['flashtrace_both_hop1']
            equal=chosen==prior
            row['historical_quality_reuse'][mode]={'exact_projected_score_match':equal,'metrics':metric if equal else None,'new_quality_queries':0}
        row['complete']=True;save()
    model.set_attn_implementation('flash_attention_2');native_method_audit(True)
    report['checkpoint_after']=checkpoint_receipt();assert report['checkpoint_before']==report['checkpoint_after']
    report['native_sources_after']=native_sources();assert report['native_sources_before']==report['native_sources_after']
    for k,v in p['budget'].items():assert report[k]==v,(k,report[k],v)
    report['status']='complete';report.pop('active',None)
except Exception:
    report['status']='failed';report['error']=traceback.format_exc();raise
finally:
    report['elapsed_seconds']=time.time()-entry_started;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','results.json',*p['sources']]:z.write(HERE/name,name)
'''
ast.parse(source);study=A/'original_FT_seq_cost16_20260907.py';study.write_text(source)
p=json.loads((A/'finite_FA_FT_cost16_protocol_20260907.json').read_text())
p.update(study_sha256=sha(study),wait_for_pid=148653,wait_for_script='${ARTIFACT_ROOT}/codex_finite_FA_FT_cost16_20260907_v1/study.py',
    methods=['finite','both_1','seq_1'],
    purpose='Check whether original FT run_attribution unused row/rec views bias the cost comparison. Original16 development; fresh unchanged finite-P1, full original FTboth1 runner, and original class+original normalize_sum_to_one seq-only expression. No attention/forward/math replacement, gradients, generation, quality retuning or new curves. One warm+three rotated measured repeats, plus NI0 passive Python-call profiles for the two FT paths. Exact full score comparisons required before any quality reuse; default FA candidate unchanged.',
    budget={'native_root_forwards':194,'native_vjps':0,'evaluation_forwards':0,'ft_attribution_forwards':130,'ordinary_reference_forwards':0,'native_attribution_forwards':64,'manual_passes':64,'extra_layer_replay_calls':2304,'extra_native_fa_attention_calls':2304,'native_attribution_endpoint_trajectories':128,'extra_layer_replay_endpoint_trajectories':4608,'finite_FA_calls_attempted':2304,'finite_FA_calls_enqueued':2304,'fresh_attribution_calls':194},
    repeats='16cases x3methods x4calls +2extra realFT Python call profiles =194; do not include instrumentation profiles in medians.',
    quality_cost_claim='A speed claim against FT must distinguish original full runner overhead from required seq output. Report complete paired costs to both; if seq-only original primitives preserve scores and are faster, use this stronger baseline going forward. No shadow replacement or intentional FT handicap.',
    serialization_policy='Standard json compact encoding for checkpoint reports, all values retained. Verified against complete previous report on host and by independent byte reconstruction; no old raw/protocol rewrite.')
protocol=A/'original_FT_seq_cost16_protocol_20260907.json';protocol.write_text(json.dumps(p,indent=2))
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_original_FT_seq_cost16_20260907_v1',f'study.py={study}',f'protocol.json={protocol}',*[f'{name}={A/name}' for name in p['sources']],'--request',str(A/'launch_original_FT_seq_cost16_20260907.json')],check=True)
req=json.loads((A/'launch_original_FT_seq_cost16_20260907.json').read_text());assert len(req['cmd'])<128000
print('Frozen original-interface seq-only fairness control;194real attributions;',len(req['cmd']),'characters.')
