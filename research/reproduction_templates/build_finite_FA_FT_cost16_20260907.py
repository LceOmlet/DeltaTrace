"""Freeze fresh isolated FT0-3 and current finite-P1 complete development costs."""
import ast,hashlib,json,subprocess,sys
from pathlib import Path
A=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
base=(A/'vendor_fa_development16_20260907.py').read_text()
source=base[:base.index('\nclass Recorder(')]
needle="        cost['native_forward_trajectories'] += value.shape[0]"
assert source.count(needle)==1
source=source.replace(needle,needle+"\n        assert kwargs.get('input_ids') is not None\n        cost.setdefault('actual_root_input_ids',[]).append(value.detach().cpu().tolist())")
source+=r'''
report.update(fresh_attribution_calls=0,evaluation_forwards=0)
quality_path=Path(p['quality_parent']);assert hashlib.sha256(quality_path.read_bytes()).hexdigest()==p['quality_parent_sha256']
quality_parent=json.loads(quality_path.read_text());assert quality_parent['status']=='complete'
assert quality_parent['checkpoint_before']==report['checkpoint_before']==quality_parent['checkpoint_after']

def restore_author_bindings():
    restored=native_method_audit()
    for name,m in model.named_modules():
        if 'forward' in m.__dict__:
            assert name in restored and getattr(m.forward,'__func__',None) is original_methods[name]
            delattr(m,'forward')
    native_method_audit(True)

def ft_run(ex,expected,family,hop):
    model.set_attn_implementation(p['official_evaluation_backend']);native_method_audit(True);torch.cuda.empty_cache()
    positions=expected['user_positions'];keep=expected['keep_local_indices']
    with measured() as cost:
        if family=='both':
            values,_,fp,fk=runner.run_attribution({'model':model,'tokenizer':tokenizer,'attr_func':'ifr_multi_hop_both','chunk_tokens':128,'sink_chunk_tokens':32,'n_hops':hop},ex,ex.target)
            score=values[0][:,:len(positions)].sum(0).cpu().float().tolist();del values
            assert fp==positions and fk==keep
        else:
            engine=attr.LLMIFRAttribution(model,tokenizer,chunk_tokens=128,sink_chunk_tokens=32,show_progress=False)
            value=engine.calculate_ifr_multi_hop(ex.prompt,target=ex.target,sink_span=tuple(ex.sink_span),thinking_span=tuple(ex.thinking_span),n_hops=hop,renorm_threshold=0.0)
            obs=value.metadata['ifr']['raw'].observation;cumulative=obs['base'].clone()
            assert len(obs['per_hop'])==hop and engine.user_prompt_indices==positions
            for delta in obs['per_hop']:cumulative=cumulative+delta
            score=cumulative.index_select(0,torch.tensor(positions,dtype=torch.long)).float().tolist()
            del value,obs,cumulative,engine
    assert cost['native_forwards']==1 and cost['native_forward_trajectories']==1 and cost['extra_replay_calls']==0
    assert cost['actual_root_input_ids']==[[expected['input_ids']]]
    report['native_root_forwards']+=1;report['ft_attribution_forwards']+=1
    restore_author_bindings();model.set_attn_implementation('flash_attention_2');native_method_audit(True)
    return {'score':score,'end_to_end_cost':cost}

def run_mode(ex,expected,mode):
    if mode=='finite':
        model.set_attn_implementation('flash_attention_2');native_method_audit(True)
        result=strong_run(ex,expected,capture_mode='finite')
        actual=result['end_to_end_cost']['actual_root_input_ids'];baseline=list(expected['input_ids'])
        for j in expected['eligible_positions']:baseline[j]=tokenizer.eos_token_id
        assert actual==[[baseline,expected['input_ids']]]
        signed=torch.tensor(result['signed_full_sequence'],dtype=torch.float64)
        result['score']=signed[expected['user_positions']].clamp_min(0).float().tolist()
        del signed
    else:
        family,hop=mode.rsplit('_',1);result=ft_run(ex,expected,family,int(hop))
    report['fresh_attribution_calls']+=1
    return result

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
            offset=(case_number+repeat)%len(modes);order=modes[offset:]+modes[:offset]
            for mode in order:
                report['active']=[dataset,index,repeat,mode];save()
                result=run_mode(ex,expected,mode)
                row['runs'].append({'repeat':repeat,'warmup':repeat==0,'mode':mode,'result':result})
                save();print('FRESH_COST',dataset,index,repeat,mode,result['end_to_end_cost']['seconds'],flush=True)
        if case_number==0:
            for mode in ['finite','both_1']:
                with torch.profiler.profile(activities=list(torch.profiler.supported_activities())) as prof:extra=run_mode(ex,expected,mode)
                trace=mode+'_profile.json';prof.export_chrome_trace(str(HERE/trace))
                row['profiles'][mode]={'trace':trace,'sha256':hashlib.sha256((HERE/trace).read_bytes()).hexdigest(),'result':extra}
                del prof,extra;save()
        for mode in modes:
            chosen=next(r['result']['score'] for r in row['runs'] if r['mode']==mode and r['repeat']==1)
            if mode=='finite':prior=qp['scores']['finite'];metrics=qp['metrics']['finite'];source=p['quality_parent_sha256']
            else:
                family,hop=mode.rsplit('_',1);key=f'flashtrace_{family}_hop{hop}'
                prior=expected['scores'][key];metrics=expected['metrics'][key];source=p['required_parent_sha256']
            equal=chosen==prior
            row['historical_quality_reuse'][mode]={'exact_projected_score_match':equal,'source_sha256':source,'metrics':metrics if equal else None,'new_quality_queries':0}
        row['complete']=True;save()
    model.set_attn_implementation('flash_attention_2');native_method_audit(True)
    report['checkpoint_after']=checkpoint_receipt();assert report['checkpoint_before']==report['checkpoint_after']
    report['native_sources_after']=native_sources();assert report['native_sources_before']==report['native_sources_after']
    assert hashlib.sha256(Path(p['library']).read_bytes()).hexdigest()==p['library_sha256']
    for k,v in p['budget'].items():assert report[k]==v,(k,report[k],v)
    report['status']='complete';report.pop('active',None)
except Exception:
    report['status']='failed';report['error']=traceback.format_exc();raise
finally:
    report['elapsed_seconds']=time.time()-entry_started;save()
    with zipfile.ZipFile(HERE/'review_bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in ['study.py','protocol.json','results.json',*p['sources']]+[f.name for f in HERE.glob('*_profile.json')]:z.write(HERE/name,name)
'''
ast.parse(source);study=A/'finite_FA_FT_cost16_20260907.py';study.write_text(source)
p=json.loads((A/'vendor_fa_development16_protocol_20260907.json').read_text())
p.pop('required_integration_pilot',None)
p.update(study_sha256=sha(study),wait_for_pid=148506,wait_for_script='${ARTIFACT_ROOT}/codex_finite_P1_conditional_sign_review_20260907_v1/study.py',
    purpose='Same-job complete B1 attribution latency and memory of current unchanged traceable-FA finite P1 and all original FlashTrace both0-3/legacy0-3 separately. Original16 development only. Each call independently prepares inputs/captures/propagates/projects/returns actual CPU scores; no cross-method capture sharing. One warm plus three rotated measured calls each, separate two NI0 profiles. Quality may be linked to prior verified original curves only if current projected vector exactly matches; otherwise unavailable until fresh original evaluation. No new quality queries or independent confirmation in this cost job.',
    quality_parent='${ARTIFACT_ROOT}/codex_vendor_fa_development16_20260907_v1/results.json',quality_parent_sha256='1598dfb8586b771e276f98ce8ade2b09b8368eb230773821a19e74e51c838bc6',
    methods=['finite']+[f'{family}_{hop}' for family in ['both','legacy'] for hop in range(4)],
    repeats='16original cases x9methods x(1warm+3measured);NI0 extra finite and originalboth1 profiles. Every hop has isolated original capture, not cumulative family time.',
    budget={'native_root_forwards':578,'native_vjps':0,'evaluation_forwards':0,'ft_attribution_forwards':513,'ordinary_reference_forwards':0,'native_attribution_forwards':65,'manual_passes':65,'extra_layer_replay_calls':2340,'extra_native_fa_attention_calls':2340,'native_attribution_endpoint_trajectories':130,'extra_layer_replay_endpoint_trajectories':4680,'finite_FA_calls_attempted':2340,'finite_FA_calls_enqueued':2340,'fresh_attribution_calls':578},
    quality_cost_claim='Report observed quality-cost frontier from all eight original controls and finite candidate. At each candidate budget, compare all FT controls whose actual per-case median fits, as a diagnostic oracle upper envelope, not a runnable selector. Main preregistered FTboth1 speed claim uses paired per-case median ratio<=1; report every violation, aggregate, peak, cold initialization/warmup and extra diagnostic costs. No post-hoc speed gate relaxation.',
    backend_scope='Original author FT attention capture and propagation execute verbatim under its original eager configuration; current candidate uses actual default nativeFA plus explicitly separate finite extension. No FT-source optimization or disguised FA replacement. Matched checkpoint/input/FP16/response/eligible set; numerical implementation differences are explicit.')
p['predeclared_review']['quality']='No new curves. Previously verified original metrics reusable only after exact score vector match, with source hash; failed matches remain unavailable.'
protocol=A/'finite_FA_FT_cost16_protocol_20260907.json';protocol.write_text(json.dumps(p,indent=2))
subprocess.run([sys.executable,str(A/'prepare_remote_experiment.py'),'${ARTIFACT_ROOT}/codex_finite_FA_FT_cost16_20260907_v1',f'study.py={study}',f'protocol.json={protocol}',*[f'{name}={A/name}' for name in p['sources']],'--request',str(A/'launch_finite_FA_FT_cost16_20260907.json')],check=True)
request=json.loads((A/'launch_finite_FA_FT_cost16_20260907.json').read_text());assert len(request['cmd'])<128000
print('Frozen578attributions; no quality/API queries;',len(request['cmd']),'command characters')
