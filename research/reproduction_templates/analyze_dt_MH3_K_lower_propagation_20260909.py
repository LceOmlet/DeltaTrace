"""Independent same-upstream change-ledger checks; no model call."""
import hashlib,json
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH3_K_lower_propagation_20260909_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes())
p,r,rec=[read(D/f) for f in ['protocol.json','results.json','terminal_receipt.json']]
assert rec['proc_exists'] is False and p==r['protocol']==read(A/'dt_MH3_K_lower_propagation_20260909_protocol.json')
assert r['status']=='MH3_K_lower_propagation_2DT0score_complete'
for f,v in rec['files'].items():assert sha(D/f)==v['sha256'] and (D/f).stat().st_size==v['bytes']
for f,h in p['files_sha256'].items():assert sha(D/f)==h
assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
assert r['DT_entered']==r['DT_returned']==2 and r['scorer_entered']==r['scorer_returned']==r['FT_calls']==r['generation_calls']==0
for n in ['control','candidate']:assert r['finite_counts'][n]=={'entered':8,'returned':8}
assert r['finite_counts']['FLA_backend']['entered']==r['finite_counts']['FLA_backend']['returned']==50
assert len(r['key_norm_calls'])==1 and r['key_norm_calls'][0]['status']=='returned'
assert r['runs'][0]['details']['key_norm_by_layer']==[] and r['runs'][1]['details']['key_norm_by_layer']==[1]
z=np.load(D/'vectors.npz',allow_pickle=False);assert sha(D/'vectors.npz')==r['vectors_sha256']
def close(x,y):assert np.max(np.abs(np.asarray(x)-np.asarray(y)),initial=0)<1e-7
def stats(x):return dict(net=float(x.sum()),positive=float(x.clip(min=0).sum()),negative=float(x.clip(max=0).sum()),absolute=float(np.abs(x).sum()))
v=r['conditional_propagation'];assert v['same_process_upstream_and_native_endpoints_equal'] is True
S=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH3_native_K_boundaries_20260909_v1';source=read(S/'results.json')
assert sha(S/'results.json')==p['native_state_source']['results_sha256']
for step,row in v['points'].items():
    f={n:z[step+'_'+n] for n in row['fields']}
    close(f['boundary_2'],0);close(f['GDN0'],f['boundary_0']-f['boundary_1']);close(f['GDN1_below_K'],f['boundary_1']-f['K_norm'])
    for n,s in row['fields'].items():
        for k,w in s.items():close(stats(f[n])[k],w)
    ids=source['protocol']['frozen_capture_receipts'][step]['deleted_positions']
    cw=z['morehopqa_3_control_run0_evaluated'].astype(np.float64);kw=z['morehopqa_3_candidate_run1_evaluated'].astype(np.float64)
    close(row['prediction_change'],(kw-cw)[ids].sum());close(row['prediction_change'],sum(row['terms'].values()))
    close(row['terms']['input_map_and_score_storage'],row['prediction_change']-f['boundary_0'].sum())
    for n in ['K_norm','GDN1_below_K','GDN0']:close(row['terms'][n],f[n].sum())
    close(row['current_prediction'],cw[ids].sum());close(row['candidate_prediction'],kw[ids].sum())
out={'status':'MH3_K_lower_propagation_independent_CPU_audit_passed','results_sha256':sha(D/'results.json'),'protocol_sha256':sha(D/'protocol.json'),'vectors_sha256':sha(D/'vectors.npz'),'analyzer_sha256':sha(Path(__file__)),'seconds':r['seconds'],'ledger':v,
    'decision':'Do not promote input-supported K. Same-upstream lower propagation can reverse an early local reduction and reinforces the mid-step overprediction. This is a propagation change ledger on actual saved states, not proof of the whole MAS cause; source-state drift remains material. Prior eight-case regression remains the quality decision.',
    'limits':'Original scorer was not rerun. Between the two native processes, boundary coefficient relative drift is about10%; native B2 boundaries1/2 differ by0.03125/0.0625 maximum. Within the C/candidate pair, pre-K coefficients and endpoint tensors match exactly. These scopes cannot be merged.'}
path=A/'dt_MH3_K_lower_propagation_summary_20260909.json';path.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'status':out['status'],'points':{s:q['terms'] for s,q in v['points'].items()}}))
