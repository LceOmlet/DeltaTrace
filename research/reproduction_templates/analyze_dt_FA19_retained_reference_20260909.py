"""Compare native route contrasts in identical output-query coordinates; zero GPU."""
import hashlib,json
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_FA19_retained_reference_20260909_v1';read=lambda p:json.loads(p.read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p,r,rec=[read(D/f) for f in ['protocol.json','results.json','terminal_receipt.json']]
assert rec['proc_exists'] is False and p==r['protocol']==read(A/'dt_FA19_retained_reference_20260909_protocol.json')
assert r['status']=='three_FA19_retained_references_6publicFA_complete'
for f,v in rec['files'].items():assert sha(D/f)==v['sha256'] and (D/f).stat().st_size==v['bytes']
for f,h in p['files_sha256'].items():assert sha(D/f)==h
assert r['native_FA_entered']==r['native_FA_returned']==6 and len(r['calls'])==6
assert r['model_calls']==r['DT_calls']==r['scorer_calls']==r['FT_calls']==r['generation_calls']==r['backward_calls']==0
assert all(c['status']=='returned' for c in r['calls'])
z=np.load(D/'vectors.npz',allow_pickle=False);assert sha(D/'vectors.npz')==r['vectors_sha256']
def close(a,b):assert np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<1e-7
def stats(x):return dict(net=float(x.sum()),positive=float(x.clip(min=0).sum()),negative=float(x.clip(max=0).sum()),absolute=float(np.abs(x).sum()))
out={'status':'retained_reference_independent_CPU_audit_passed_fixed_endpoint_mixtures_insufficient','results_sha256':sha(D/'results.json'),'protocol_sha256':sha(D/'protocol.json'),'vectors_sha256':sha(D/'vectors.npz'),'analyzer_sha256':sha(Path(__file__)),'seconds':r['seconds'],'cases':{}}
for key,case in r['cases'].items():
    source=A/'snapshot'/case['source']['reference_vectors'].lstrip('/');assert sha(source)==case['source']['reference_vectors_sha256'];old=np.load(source,allow_pickle=False)
    out['cases'][key]={}
    for step,row in case['points'].items():
        f={n:z[key+'_'+step+'_'+n] for n in row['fields']};prefix='FA19_'+step+'_' if key=='MH3' else step+'_'
        for n in ['R0','RA']:close(f[n],old[prefix+n].reshape(-1))
        close(f['EOS_reference_error'],f['R0']-f['RA']);close(f['clean_reference_error'],f['RC']-f['RA']);close(f['actual_total'],f['RC']+f['actual_P_A_content'])
        for n,s in row['fields'].items():
            for k,v in s.items():close(stats(f[n])[k],v)
        fields={n:stats(v) for n,v in f.items()}
        out['cases'][key][step]={'fields':fields,'retained_below_both_fixed_reference_nets':fields['RA']['net']<min(fields['R0']['net'],fields['RC']['net']),
            'minimum_global_convex_reference_net_gap':min(fields['R0']['net'],fields['RC']['net'])-fields['RA']['net'],
            'clean_reference_absolute_error_minus_EOS':fields['clean_reference_error']['absolute']-fields['EOS_reference_error']['absolute']}
out['decision']='All six retained-reference contrasts lie below both fixed-reference scalar contrasts, and clean-reference query-wise absolute errors are larger. In ideal real-arithmetic FA with fixed routing, a single convex V0/VC reference cannot produce RA. Do not turn this into a fitted rowwise weight or assume the clean endpoint fixes the mismatch.'
out['limits']='RA and RC describe different valid conditional PV orders, not a unique intrinsic route sign. Convex-family statement follows the ideal operator linearity in V plus observed scalar ordering; arbitrary BF16 mixtures were not executed or uniformly error-bounded. No new whole attribution or MAS curve.'
path=A/'dt_FA19_retained_reference_summary_20260909.json';path.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({'status':out['status'],'cases':{k:{s:v['minimum_global_convex_reference_net_gap'] for s,v in row.items()} for k,row in out['cases'].items()}}))
