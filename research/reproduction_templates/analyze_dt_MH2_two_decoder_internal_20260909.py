"""No-model NumPy audit of actual MH2 FA19/GDN1 internal ledger receipts."""
import hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH2_two_decoder_internal_20260909_v1'
started=time.perf_counter();sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();read=lambda f:json.loads(f.read_bytes())
def close(a,b,label=''):
    error=float(np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0));assert error<1e-7,(label,error)
    return error
p,r=read(D/'protocol.json'),read(D/'results.json')
assert p==r['protocol']==read(A/'dt_MH2_two_decoder_internal_protocol_20260909.json')
assert sha(D/'protocol.json')=='148a2212a58ac852b09cc54b3ed8d9405e7fcf620e0cea00985614319a542882'
assert p['files_sha256']['study.py']=='04c12c859e1391d89915720970772b064f29536f59ff1ab0f11c35aa12c9e008'
assert r['status']=='MH2_FA19_GDN1_1native10replay2finite_internal_complete',r.get('error')
receipt=read(D/'terminal_receipt.json');assert receipt['proc_exists'] is False
for name,item in receipt['files'].items():assert sha(D/name)==item['sha256'] and (D/name).stat().st_size==item['bytes']
for name,want in p['files_sha256'].items():assert sha(D/name)==want
with zipfile.ZipFile(D/'review_bundle.zip') as z:
    assert not any(n.endswith('.pt') for n in z.namelist())
    for n in z.namelist():assert z.read(n)==(D/n).read_bytes()
assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']==p['expected_weight_stats']
assert r['model_loads']==r['whole_entered']==r['whole_returned']==r['aux_FA_entered']==r['aux_FA_returned']==1
assert r['replay_entered']==r['replay_returned']==10 and r['finite_decoder_entered']==r['finite_decoder_returned']==2
assert r['finite_counts']=={'FA':{'entered':1,'returned':1},'FLA':{'entered':1,'returned':1}}
assert all(r[k]==0 for k in ['DT_calls','scorer_calls','FT_calls','generation_calls'])
assert all(c['status']=='returned' and c['seconds']>=0 for c in r['calls'])
assert r['root_layer_counts']=={str(i):1 for i in range(32)}
assert r['native_counts_from_returned_calls']=={'root_FA':8,'root_FLA':24,'root_conv':24,'selected_FA_replays':5,'selected_FLA_and_conv_replays':5,'finite_FLA_adjoint_stages':2,'finite_GDN_conv_preactivation_and_autograd_each':1}
src=p['source_boundary'];S=(A/'snapshot'/src['results_path'].lstrip('/')).parent
for n in ['results','protocol','vectors']:assert sha(S/(n+('.npz' if n=='vectors' else '.json')))==src[n+'_sha256']
prior=read(S/'results.json');pv=np.load(S/'vectors.npz',allow_pickle=False);v=np.load(D/'vectors.npz',allow_pickle=False)
assert r['input']==p['input']==prior['cases']['morehopqa_2']['input'];T=r['input']['total_length']
out={'status':'MH2_two_decoder_internal_independent_CPU_audit_passed','analyzer_sha256':sha(Path(__file__)),
    'results_sha256':sha(D/'results.json'),'protocol_sha256':sha(D/'protocol.json'),'vectors_sha256':sha(D/'vectors.npz'),
    'source_boundary':src,'private_artifact':r['private_artifact'],'actual_budget':dict(p['budget'],job_seconds=r['seconds']),
    'layers':{},'next':'Root review actual largest internal terms before selecting any new propagation rule; no automatic dispatch.'}
keys=set()
for index in ['19','1']:
    lr=r['layers'][index];assert set(lr['points'])=={'B2','0','3','10','20'}
    assert set(p['required_decoder_kwargs']).issubset(lr['native_kwargs_keys'])
    for name,row in lr['points'].items():
        assert row['decoder_calls']=={k:1 for k in ['input_norm','post_norm','gate','up','silu','down','mlp','decoder']}
        expected={'module':1,'interface':1,'native_varlen':0,'native_dense':1} if index=='19' else {'module':1,'conv':1,'FLA':1,'stage':1}
        assert row['mixer_calls']==expected and row['initial_cache']['provided']==(name!='B2')
        if index=='1':assert row['initial_cache']['has_previous_state'] is False and row['initial_cache']['layer_index']==1
    dst={k:lr[k] for k in ['kind','root_capture_input_drift','root_capture_output_drift','saved_input_coeff_drift']}
    dst['native_output_replay_drift']={s:q['output_replay_drift'] for s,q in lr['points'].items()};dst['points']={};out['layers'][index]=dst
    if index=='19':assert lr['FA_activity']['GQA_input_expansion'] is False
    for step in ['3','10','20','B2']:
        row=lr['ledgers'][step];dec=row['internal'];terms=dec['terms'];assert len(terms)==(9 if index=='19' else 16)
        assert dec['sign_convention']=='prediction_minus_actual'
        close(sum(terms.values()),dec['prediction_minus_actual']);close(dec['input_contraction']-dec['output_contraction'],dec['prediction_minus_actual'])
        names=['saved_minus_replayed_coeff','input_replay_transfer','output_replay_transfer','saved_boundary_error']
        arrays={n:v[f'{index}_{step}_{n}'] for n in names};keys.update(f'{index}_{step}_{n}' for n in names)
        for a in arrays.values():assert a.shape==(1,T) and a.dtype==np.float64 and np.isfinite(a).all()
        for n,want in row['transfers'].items():close(arrays[n].sum(),want)
        measured=arrays['saved_boundary_error'];close(measured.sum(),row['saved_boundary_error'])
        close(sum(terms.values())+sum(row['transfers'].values()),row['saved_boundary_error'])
        if step=='B2':expected=prior['runs'][0]['B2_endpoint_boundary_contractions'][index]-prior['runs'][0]['B2_endpoint_boundary_contractions'][str(int(index)+1)]
        else:
            expected=-prior['cases']['morehopqa_2']['decomposition'][step]['decoder_errors'][index]
            close(measured[0],-pv[f'step{step}_decoder_{index}_actual_minus_predicted'])
        close(measured.sum(),expected)
        assert set(row['groups'])=={'deleted','other_prompt','response'}
        for term,value in terms.items():close(sum(g['terms'][term] for g in row['groups'].values()),value)
        for group in row['groups'].values():close(sum(group['terms'].values()),group['prediction_minus_actual'])
        dst['points'][step]={'saved_boundary_error':row['saved_boundary_error'],'transfers':row['transfers'],'transfer_net':sum(row['transfers'].values()),
            'terms':terms,'positive_terms_ranked':sorted([(k,x) for k,x in terms.items() if x>0],key=lambda z:-z[1]),
            'negative_terms_ranked':sorted([(k,x) for k,x in terms.items() if x<0],key=lambda z:z[1]),'groups':row['groups']}
assert set(v.files)==keys
out['limits']=['No candidate, metric improvement or production timing measured. Signed compensations retained.',
    'Scalar and grouped internal ledgers are checked algebraically against public per-token boundary/transfer vectors; private activation/coefficient dot products remain remote and are not independently recomputed by this NumPy audit.',
    'Group locations are contraction coordinates, not independent source-token effects. Source root/replay drift and transfer remain explicit; no bitwise requirement.',
    'Budget includes actual one whole B2 kwargs forward,10selected replays,2finite decoders,1public FA LSE; whole-graph FA/FLA counts inferred from verified native graph and all32 observed entries.']
out['CPU_audit_seconds']=time.perf_counter()-started
path=A/'dt_MH2_two_decoder_internal_summary_20260909.json';path.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'status':out['status'],'sha256':sha(path),'seconds':r['seconds'],
    'layers':{i:{s:{'error':x['saved_boundary_error'],'transfer':x['transfer_net'],'positive':x['positive_terms_ranked'][:4],'negative':x['negative_terms_ranked'][:3]} for s,x in z['points'].items()} for i,z in out['layers'].items()}}))
