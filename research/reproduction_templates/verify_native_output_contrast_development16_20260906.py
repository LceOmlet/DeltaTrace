"""Audit original metrics, full native gradient artifacts, provenance and gates."""
import argparse,hashlib,io,json,math,statistics,zipfile
from pathlib import Path
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--remote-artifacts',action='store_true');args=parser.parse_args()
root=Path(__file__).resolve().parent;folder=root if args.remote_artifacts else root/'snapshot${ARTIFACT_ROOT}/codex_native_output_contrast_development16_20260906_v1'
protocol_path=root/('protocol.json' if args.remote_artifacts else 'native_output_contrast_development16_protocol_20260906.json')
study_path=root/('study.py' if args.remote_artifacts else 'native_output_contrast_development16_20260906.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads((folder/'results.json').read_text(encoding='utf-8'));p=d['protocol']
assert p==json.loads(protocol_path.read_text(encoding='utf-8'))
assert sha(protocol_path)=='87dc09465e28f0c91840d317fed818eaadd16a9a7e5643c76ff5860716af7cbc'
assert sha(folder/'study.py')==p['study_sha256'] and (folder/'study.py').read_bytes()==study_path.read_bytes()
for name,digest in p['sources'].items():assert sha(root/name)==digest
assert d['status']=='complete' and d['checkpoint_before']==d['checkpoint_after'] and d['native_sources_before']==d['native_sources_after']
assert d['all_native_vectors_frozen_before_any_quality']
expected={(dataset,i) for dataset,indices in p['selection'].items() for i in indices}
assert len(d['records'])==16 and {(r['dataset'],r['idx']) for r in d['records']}==expected
for key in ['native_root_forwards','native_vjps','native_attribution_forwards','ordinary_reference_forwards','ft_attribution_forwards','evaluation_forwards']:assert d[key]==p['budget'][key]
cap=Path('${ARTIFACT_ROOT}/codex_qwen_official_cache_capacity_20260906_v1/results.json') if args.remote_artifacts else root/'snapshot${ARTIFACT_ROOT}/codex_qwen_official_cache_capacity_20260906_v1/results.json'
assert sha(cap)=='4db8f66946de9ad66f3d1313a472f6d5e626173715d201a5fecac6c6a90dd250'
capacity=json.loads(cap.read_text());inputs={(dataset,r['idx']):r for dataset,e in capacity['datasets'].items() for r in e['records']}
methods=p['native_methods']+p['baselines'];candidates=p['candidates'];baselines=p['baselines']
out={'status':'verified_complete','raw_sha256':sha(folder/'results.json'),'records':16,'native_root_forwards':d['native_root_forwards'],'native_vjps':d['native_vjps'],
    'vector_verification_location':'remote CPU NumPy against archived actual gradients' if args.remote_artifacts else 'local CPU NumPy against downloaded gradient archive',
    'curves_verified':0,'signed_vectors_reconstructed':0,'max_signed_reconstruction_error':0.,'max_metric_reconstruction_error':0.,'means':{},'quality_gates':{},'paired':{},'costs':{},'per_case':[],
    'scope':'Original16 development only. Same unchanged original eager evaluator for all methods, original FA native candidate vectors; no independent confirmation or negative-sign certification.'}
def area(x):return float((x.sum()-(x[0]+x[-1])/2)/(len(x)-1))
def metrics(row,name):
    w=np.array(row['scores'][name],dtype=np.float32);keep=set(row['keep_local_indices']);m=row['metrics'][name]
    assert len(w)==len(row['user_positions']) and np.isfinite(w).all() and (w>=0).all()
    masks=row['evaluation_masks'][name];assert len(masks)==21 and not masks[0] and set(masks[-1])==keep
    curve=np.array(m['raw_curve']);assert len(curve)==21 and np.isfinite(curve).all()
    assert [curve[0],curve[-1]]==row['common_eager_evaluation_endpoints16']
    total=float(w[sorted(keep)].sum(dtype=np.float32));density=[1.]
    for step in range(1,21):
        assert len(masks[step])==len(set(masks[step])) and set(masks[step-1])<=set(masks[step])<=keep
        group=set(masks[step])-set(masks[step-1]);rest=keep-set(masks[step])
        assert len(group)==len(keep)//20+int(step<=len(keep)%20)
        if rest:assert min(w[list(group)])>=max(w[list(rest)])
        density.append(density[-1]-float(w[sorted(group)].sum(dtype=np.float32))/total if total>0 else 1-step/20)
    norm=np.minimum.accumulate(np.clip((curve-curve[-1])/abs(curve[0]-curve[-1]),0,1));penalty=np.abs(norm-np.array(density))
    corrected=np.clip(norm+penalty,0,1)
    corrected=(corrected-corrected.min())/(corrected.max()-corrected.min()) if corrected.max()>corrected.min() else np.linspace(1,0,21)
    error=max(abs(a-m[k]) for a,k in zip([area(norm),area(corrected),area(norm+penalty)],['rise','mas','mas_unnormalized']))
    assert error<5e-6,(row['dataset'],row['idx'],name,error)
    out['max_metric_reconstruction_error']=max(out['max_metric_reconstruction_error'],error);out['curves_verified']+=1
    if m['recovery'] is not None:
        gold=set(row['gold_eligible_local']);chosen=row['recovery_topk_local'][name]
        assert len(chosen)==len(set(chosen))==max(1,math.ceil(.1*len(keep))) and set(chosen)<=keep
        assert min(w[chosen])>=max(w[list(keep-set(chosen))])
        assert len(set(chosen)&gold)/len(gold)==m['recovery']
    cost=row['evaluation_costs'][name];assert cost['native_forwards']==21 and cost['native_decoder_layer_calls']==756 and cost['vjps']==0
with zipfile.ZipFile(folder/'vectors.zip') as z:
    aliases=json.loads(z.read('aliases.json'))
    for row in d['records']:
        assert row['complete'] and row['native_complete']
        payload={k:v for k,v in row.items() if k!='record_digest_sha256'}
        assert hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest()==row['record_digest_sha256']
        assert hashlib.sha256(json.dumps(row['input_ids']).encode()).hexdigest()==row['input_ids_sha256']==inputs[(row['dataset'],row['idx'])]['input_ids_sha256']
        assert row['eligible_positions']==[row['user_positions'][i] for i in row['keep_local_indices']]
        assert row['gold_eligible_local']==sorted(set(row['gold_full_local'])&set(row['keep_local_indices']))
        assert set(row['scores'])==set(row['metrics'])==set(methods)
        assert len(row['restored_native_forward_attributes'])==36
        assert row['memory_reference']['native_root_forwards']==row['memory_reference']['native_vjps']==1
        details={'dataset':row['dataset'],'idx':row['idx'],'N':len(row['input_ids']),'M':len(row['input_ids'])-row['prompt_len'],'metrics':{},'native':{}}
        for name,native in row['native'].items():
            arrays={}
            for key,a in native['vector_artifacts'].items():
                raw=z.read(aliases[a['file']]);assert hashlib.sha256(raw).hexdigest()==a['sha256']
                arrays[key]=np.load(io.BytesIO(raw),allow_pickle=False)
            assert arrays['gradient'].dtype==np.float16 and arrays['delta'].dtype==np.float32
            assert arrays['gradient'].shape==arrays['delta'].shape==(len(row['eligible_positions']),4096)
            signed=np.zeros(len(row['input_ids']),dtype=np.float64)
            signed[row['eligible_positions']]=(arrays['gradient'].astype(np.float64)*arrays['delta'].astype(np.float64)).sum(-1)
            error=float(np.max(np.abs(signed-np.array(native['signed_full_sequence']))));assert error<1e-10
            out['max_signed_reconstruction_error']=max(out['max_signed_reconstruction_error'],error);out['signed_vectors_reconstructed']+=1
            assert np.array_equal(np.maximum(signed[row['user_positions']],0).astype(np.float32),np.array(row['scores'][name],dtype=np.float32))
            assert math.isclose(float(signed.sum()),native['signed_sum'],abs_tol=1e-9,rel_tol=1e-12)
            assert not native['gradient_replacement'] and native['backend']==p['candidate_backend']
            count=1 if name=='native_G_clean' else 2
            assert native['cost']=={'native_root_forwards':count,'native_decoder_calls':count*36,'native_vjps':1}
            for key in ['point_scores']+([] if name=='native_G_clean' else ['base_scores','clean_scores']):
                score=native[key];assert abs(math.fsum(score['target_logprobs32'])-score['G32'])<1e-9
            if name!='native_G_clean':
                head=native['head_seed_audit'];delta=native['clean_scores']['G32']-native['base_scores']['G32'];assert delta==head['delta_G32']
                for prefix,label in [('float32','real'),('native_float16','half')]:
                    residual=abs(delta-head[prefix+'_seed_credit'])/max(abs(delta),1)
                    assert residual==head['relative_'+label+'_head_residual' if label=='real' else 'relative_half_seed_residual'] and residual<=1e-3
                assert head['native_float16_seed_credit']-native['signed_sum']==native['body_linearization_remainder']
            seconds=native['seconds']+row['input_preparation_seconds'];assert seconds==native['comparable_attribution_seconds']
            ratio=seconds/row['ft_costs']['flashtrace_both_hop1']['seconds'];assert ratio==native['time_ratio_to_ft'] and native['time_gate_pass']==(ratio<=1)
            excess=native['peak_allocated_bytes']-row['memory_reference']['peak_allocated_bytes'];assert excess==native['peak_bytes_above_reference'] and native['memory_gate_pass']==(excess<=72500000)
            details['native'][name]={k:native[k] for k in ['comparable_attribution_seconds','time_ratio_to_ft','time_gate_pass','peak_allocated_bytes','peak_bytes_above_reference','memory_gate_pass','signed_sum','signed_absolute_mass']}
            details['native'][name]['body_linearization_remainder']=native.get('body_linearization_remainder')
            details['native'][name]['delta_G32']=native.get('head_seed_audit',{}).get('delta_G32')
            details['native'][name]['zero_support_tokens']=int(np.sum(np.array(row['scores'][name])[row['keep_local_indices']]==0))
        for c in list(row['ft_costs'].values())+[row['legacy_joint_cost']]:assert c['native_forwards']==1 and c['native_decoder_layer_calls']==36 and c['vjps']==0
        for name in methods:
            metrics(row,name);details['metrics'][name]={k:row['metrics'][name][k] for k in ['rise','mas','recovery']}
        out['per_case'].append(details)
out['vectors_zip_sha256']=sha(folder/'vectors.zip')
for dataset in p['selection']:
    rows=[r for r in d['records'] if r['dataset']==dataset]
    out['means'][dataset]={name:{metric:statistics.mean(r['metrics'][name][metric] for r in rows) if all(r['metrics'][name][metric] is not None for r in rows) else None for metric in ['recovery','rise','mas']} for name in methods}
    means=out['means'][dataset];out['paired'][dataset]={};rng=np.random.default_rng(20260906);samples=rng.integers(0,len(rows),size=(10000,len(rows)))
    for candidate in p['native_methods']:
        out['paired'][dataset][candidate]={}
        for metric in ['recovery','rise','mas']:
            if means[candidate][metric] is None:continue
            best=(max if metric=='recovery' else min)(baselines,key=lambda b:means[b][metric])
            for baseline in dict.fromkeys([best,'flashtrace_both_hop1','native_G_clean']):
                if baseline==candidate:continue
                gains=np.array([(r['metrics'][candidate][metric]-r['metrics'][baseline][metric])*(1 if metric=='recovery' else -1) for r in rows])
                out['paired'][dataset][candidate][metric+'|'+baseline]={'mean_improvement':float(gains.mean()),'wins':int((gains>0).sum()),'ties':int((gains==0).sum()),'losses':int((gains<0).sum()),'bootstrap95_descriptive':np.quantile(gains[samples].mean(1),[.025,.975]).tolist(),'per_case_improvements':gains.tolist(),'strongest_ft_for_metric':best}
ni=out['means']['niah_mq_q2'];mh=out['means']['morehopqa']
for candidate in p['native_methods']:
    old=['flashtrace_both_hop1']+[f'flashtrace_legacy_hop{i}' for i in range(4)]
    groups={'expanded8':baselines,'historical5':old};out['quality_gates'][candidate]={}
    for label,bs in groups.items():
        gates={'niah_recovery':ni[candidate]['recovery']>=max(ni[b]['recovery'] for b in bs),'morehop_rise':mh[candidate]['rise']<=min(mh[b]['rise'] for b in bs),'morehop_mas':mh[candidate]['mas']<=min(mh[b]['mas'] for b in bs)}
        gates['joint_pass']=all(gates.values());out['quality_gates'][candidate][label]=gates
    rows=[r['native'][candidate] for r in d['records']];ratios=[r['time_ratio_to_ft'] for r in rows]
    out['costs'][candidate]={'median_time_ratio_to_ft':statistics.median(ratios),'p95_time_ratio_to_ft':float(np.quantile(ratios,.95)),
        'max_peak_bytes':max(r['peak_allocated_bytes'] for r in rows),'max_excess_bytes':max(r['peak_bytes_above_reference'] for r in rows),
        'time_failures':[(r['dataset'],r['idx']) for r in d['records'] if not r['native'][candidate]['time_gate_pass']],
        'memory_failures':[(r['dataset'],r['idx']) for r in d['records'] if not r['native'][candidate]['memory_gate_pass']]}
    out['costs'][candidate]['all_resources_pass']=not out['costs'][candidate]['time_failures'] and not out['costs'][candidate]['memory_failures']
out['eligible_candidates']=[c for c in candidates if out['quality_gates'][c]['expanded8']['joint_pass'] and out['costs'][c]['all_resources_pass']]
assert out['curves_verified']==176 and out['signed_vectors_reconstructed']==48
out['decision']='Advance only eligible frozen candidates to separately defined conditional-sign and independent confirmation work.' if out['eligible_candidates'] else 'No eligible candidate: preserve full failures, do not launch reserved/cross-task confirmation or tune endpoint signs/normalization. Reconsider body-nonlinearity versus fixed-budget conditional finite measurement.'
(root/'native_output_contrast_development16_summary_20260906.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ['per_case','paired']},ensure_ascii=False,indent=2))
