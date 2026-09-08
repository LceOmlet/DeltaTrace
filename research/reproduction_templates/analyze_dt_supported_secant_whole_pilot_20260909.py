"""CPU audit of the frozen supported/P1 whole pilot, including both real masks."""
import hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
from analyze_dt_original_regression_20260909 import close,stats,auc,needle,masks,curve_audit
from analyze_dt_PV_layer19_whole_pilot_20260909 import inputs_audit,drift

A=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
local=lambda p:A/'snapshot'/str(p).lstrip('/')

def provenance(D,p,r):
    receipt=read(D/'terminal_receipt.json')
    assert receipt['proc_exists'] is False
    assert p==r['protocol']==read(A/'dt_supported_secant_whole_pilot_protocol_20260909.json')
    for n,v in receipt['files'].items():
        assert sha(D/n)==v['sha256'] and (D/n).stat().st_size==v['bytes'],n
    for n,v in p['files_sha256'].items():assert sha(D/n)==v,n
    with zipfile.ZipFile(D/'review_bundle.zip') as z:
        for n in z.namelist():assert z.read(n)==(D/n).read_bytes(),n
    for n,v in p['official_source_blob_sha1'].items():
        raw=(local(p['official_root'])/n).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==v,n
        assert r['sources_before']['official/'+n]==hashlib.sha256(raw).hexdigest()
    for n,v in p['runtime_source_sha256'].items():
        assert r['sources_before']['native/'+n]==v
        f=local(p['isolated_site'])/n
        if f.exists() and sha(f)!=v:
            fresh=A/'snapshot${ARTIFACT_ROOT}/codex_dt_current_FLA_source_receipts_20260909_v1'
            fr=read(fresh/'receipt.json')[n];ff=fresh/Path(n).name
            assert fr['sha256']==sha(ff)==v and fr['bytes']==ff.stat().st_size
        elif f.exists():assert sha(f)==v
    for i in p['protected_sources']:
        assert r['sources_before'][i['path']]==i['sha256']
        if local(i['path']).exists():assert sha(local(i['path']))==i['sha256']
    assert r['sources_before']==r['sources_after']
    assert r['weight_stats_before']==r['weight_stats_after']==p['expected_weight_stats']
    return {'before_after_equal':True,'files_verified':len(p['files_sha256']),
        'original_metric_blob_sha1':p['official_source_blob_sha1']['flashtrace/improved.py'],
        'original_FA_library_sha256':p['finite_FA_library_sha256'],
        'supported_FA_library_sha256':p['candidate_library_sha256'],
        'native_model_sha256':p['native_model_sha256'],
        'scope':'Actual source receipts and local frozen copies checked; native weight size/mtime guards retained, no full weight rehash.'}

def audit_run(run,p,r,z,inputs):
    assert run['status']=='complete' and run['root_forwards']==1
    d=run['details'];c=run['counts'];method=run['method']
    want={'native_root':1,'native_decoder_replays':32,'finite_decoder_calls':32,
        'public_FA_auxiliary_calls':8,'finite_FA_calls':8,'finite_FLA_calls':24,'finite_FLA_backend_calls':25}
    assert {k:c[k] for k in want}==want
    assert run['finite_callback_counts']=={'FA_entered':8,'FA_returned':8,'FLA_backend_entered':25,'FLA_backend_returned':25}
    assert run['native_stage_accounting']=={'returned_FLA_backend_calls_times_two':50,'stages_inside_nonreturned_backend':0}
    assert d['norm_gate_rules']=={'0':'symmetric'} and d['finite_fla_by_layer']==[0] and d['select_output_rows']
    assert 'attention_pv_rules' not in d  # Exact pre-PV frozen runner.
    wrap=run['candidate_layer0_receipts'];assert len(wrap)==1
    assert wrap[0]['status']=='returned' and wrap[0]['endpoint_permutation']==[1,0]
    assert [x['orientation'] for x in wrap[0]['backend_calls']]==['original','swapped']
    assert all(x['status']=='returned' for x in wrap[0]['backend_calls'])
    kinds=[x['kind'] for x in d['calls']]
    for pref in ('native_replay_','finite_decoder_'):
        assert [k for k in kinds if k.startswith(pref)]==[pref+str(i) for i in reversed(range(32))]
    assert [k for k in kinds if k.startswith('public_FA_LSE_')]==['public_FA_LSE_'+str(i) for i in reversed(p['expected_FA_layers'])]
    for i,l in d['layers'].items():
        assert l['decoder_calls']=={k:1 for k in ('input_norm','post_norm','gate','silu','up','down','mlp','decoder')}
        fa=int(i) in p['expected_FA_layers']
        assert l['mixer_calls']==({'module':1,'interface':1,'native_varlen':0,'native_dense':1} if fa else {'module':1,'conv':1,'FLA':1,'stage':1})
    rr=run['supported_FA_receipts'];assert len(rr)==(8 if method=='candidate' else 0)
    for x in rr:
        assert x['status']=='returned' and x['kernel_phases_on_success']==3 and x['row_field_count']==21
        assert x['row_state_shape'][0]==21 and x['row_state_shape'][1]==1
        assert x['row_state_bytes']==int(np.prod(x['row_state_shape']))*4 and x['original_wrapper_synchronization']
    key=run['case'];info=r['cases'][key]['input'];full=z[run['vector_key']+'_full'];w=z[run['vector_key']+'_evaluated']
    assert full.dtype==np.float64 and w.dtype==np.float32 and np.isfinite(full).all()
    assert full.shape==(info['total_length'],) and w.shape==(info['prompt_length'],)
    assert np.array_equal(full[:len(w)].astype(np.float32),w)
    close(full.sum(),d['signed_sum'],1e-7)
    for field in ('net','positive','negative'):close(stats(full)[field],run['signed_summary'][field],1e-7)
    nt=needle(w,info['keep'],r['cases'][key]['gold'],run['signed_summary']['needle'])
    masks(w,info,*inputs[key],run['deletion_audit'])
    lp0=np.asarray(d['target_logp0']);lp1=np.asarray(d['target_logp1'])
    assert lp0.shape==lp1.shape==(info['target_length'],)
    assert d['selected_predictor_rows']==list(range(info['prompt_length']-1,info['total_length']-1))
    close((lp1-lp0).sum(),d['root_effect'],1e-7)
    mem=run['memory_cost'];assert mem['peak_allocated_full_model_resident']==d['peak_allocated']
    assert mem['peak_reserved_full_model_resident']==d['peak_reserved']
    assert mem['peak_allocated_minus_before_pair']==d['peak_allocated']-run['GPU_allocated_before_pair']
    return {'case':key,'method':method,'needle':nt,'outer_seconds':run['outer_attribute_seconds'],
        'runner_seconds':d['complete_attribution_seconds_with_diagnostics'],'memory_cost':mem,
        'signed':stats(full),'root_effect':d['root_effect'],'seed_effect':d['seed_effect'],
        'relative_residual':d['relative_residual'],'supported_FA_receipts':rr}

def error_summary(e):
    return {'MAE_AUC':auc(abs(e)),'max_absolute':float(abs(e).max()),'early_step3':float(e[3]),
            'middle_step10':float(e[10]),'allEOS_step20':float(e[20])}

def main():
    D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_supported_secant_whole_pilot_20260909_v1'
    p=read(D/'protocol.json');r=read(D/'results.json')
    assert r['status']=='all8FA_supported_secant_current_layer0_4DT100FLA84score_complete',r.get('error')
    out={'status':'supported_whole_pilot_actual_original_metrics_and_two_masks_CPU_audited','results_sha256':sha(D/'results.json'),
        'protocol_sha256':sha(D/'protocol.json'),'receipt_sha256':sha(D/'terminal_receipt.json'),
        'analyzer_sha256':sha(__file__),'sources':provenance(D,p,r),'wall_seconds':r['seconds'],
        'cases':{},'frozen_budget':p['budget'],'counts':{'models':1,'eager_initializations':1,'DT':4,'FLA':100,
            'native_FLA_adjoint_stages':200,'finite_FA':32,'public_FA_LSE':32,'native_decoder_replays':128,'original_scorer':84,'FT':0,'generation':0}}
    assert r['model_loads']==r['native_eager_diagnostics']==1 and r['FT_calls']==r['generation_calls']==0
    assert r['DT_entered']==r['DT_returned']==4 and r['scorer_entered']==r['scorer_returned']==84
    assert [[x['case'],x['method'],x['phase']] for x in r['runs']]==p['call_schedule']
    assert r['finite_counts']['FLA_backend']=={'entered':100,'returned':100,'native_adjoint_stages_from_returned_calls':200,'native_stages_inside_nonreturned_calls':0}
    for m in ('control','candidate'):assert r['finite_counts'][m]=={'entered':16,'returned':16}
    sc=r['finite_counts']['supported_FA_native_library'];assert sc['entered']==sc['returned']==16 and sc['successful_phases']==48
    assert r['finite_counts']['layer0_average_wrapper']['entered']==r['finite_counts']['layer0_average_wrapper']['returned']==4
    assert sha(D/'vectors.npz')==r['vectors_sha256'];out['vectors_sha256']=r['vectors_sha256']
    z=np.load(D/'vectors.npz',allow_pickle=False)
    inputs,out['input_provenance']=inputs_audit(p,r)
    out['runs']=[audit_run(x,p,r,z,inputs) for x in r['runs']]
    for key,case in r['cases'].items():
        rr={m:next(x for x in r['runs'] if x['case']==key and x['method']==m) for m in ('control','candidate')}
        methods={};ws={m:z[key+'_'+m+'_evaluated'].astype(np.float64) for m in rr}
        for m in rr:
            alias='DT_'+m;v={key+'_'+alias+t:z[key+'_'+m+t] for t in ('_full','_evaluated')}
            methods[m]=curve_audit(key,alias,case['curves'][m],case,rr[m]['deletion_audit'],v,inputs)
        root={field:drift(rr['candidate']['details'][field],rr['control']['details'][field]) for field in ('target_logp0','target_logp1')}
        assert all(v['equal'] for v in root.values())
        close(rr['candidate']['details']['seed_effect'],rr['control']['details']['seed_effect'],1e-7)
        c={'methods':methods,'root_pair_drift':root,'candidate_minus_control':{k:methods['candidate'][k]-methods['control'][k] for k in ('RISE','MAS')},'backgrounds':{}}
        for background in rr:
            curve=case['curves'][background];g=np.asarray(curve['scores']);actual=g[0]-g
            pred={m:np.array([w[i['deleted_positions']].sum() for i in curve['input_receipts']]) for m,w in ws.items()}
            e={m:pred[m]-actual for m in rr};change=abs(e['candidate'])-abs(e['control'])
            if background=='control':
                for step,row in enumerate(case['fixed_control_masks']['points']):
                    for m in rr:close(row[m]['prediction_minus_actual'],e[m][step],1e-7)
            c['backgrounds'][background]={'actual_drop':actual.tolist(),'predicted':{m:v.tolist() for m,v in pred.items()},
                'errors':{m:v.tolist() for m,v in e.items()},'summary':{m:error_summary(v) for m,v in e.items()},
                'candidate_better_equal_worse_interior19':[int((change[1:20]<-1e-9).sum()),int((abs(change[1:20])<=1e-9).sum()),int((change[1:20]>1e-9).sum())],
                'input_receipts':curve['input_receipts']}
        same=[]
        for i,x in enumerate(case['curves']['control']['input_receipts']):
            for j,y in enumerate(case['curves']['candidate']['input_receipts']):
                if x['input_sha256']==y['input_sha256']:
                    diff=case['curves']['candidate']['scores'][j]-case['curves']['control']['scores'][i]
                    same.append({'control_step':i,'candidate_step':j,'score_difference':diff})
        assert all(x['score_difference']==0 for x in same)
        c['same_input_score_checks']=same
        c['scope']='Two real observed deletion paths only; MAE is diagnostic logprob error, not MAS. No additional scoring or final-score adjustment.'
        out['cases'][key]=c
    out['cost_scope']='One observation per method/case, opposite order across cases, includes compilation warm effects. No stable speedup/slowdown or true-multisample inference.'
    out['numerical_scope']='Whole input finite values and seed/root/source identities checked. Per-row supported diagnostics allocated and count-verified but not copied by this pilot; do not claim all row constraints independently checked at every layer.'
    target=A/'dt_supported_secant_whole_pilot_summary_20260909.json'
    target.write_text(json.dumps(out,indent=2,allow_nan=False))
    print(json.dumps({'summary_sha256':sha(target),'counts':out['counts'],'seconds':out['wall_seconds'],
        'cases':{k:{'delta':v['candidate_minus_control'],'metrics':{m:{x:y[x] for x in ('RISE','MAS','needle')} for m,y in v['methods'].items()},
        'mask_error':{b:d['summary'] for b,d in v['backgrounds'].items()}} for k,v in out['cases'].items()}},ensure_ascii=False))

if __name__=='__main__':main()
