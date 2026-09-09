"""Independent signed contractions and native-call accounting for two rejected path rules."""
import hashlib,json
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_FA19_joint_path_scout_20260909_v1';read=lambda p:json.loads(p.read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p,r,rec=[read(D/f) for f in ['protocol.json','results.json','terminal_receipt.json']]
assert rec['proc_exists'] is False and p==r['protocol']==read(A/'dt_FA19_joint_path_scout_20260909_protocol.json')
assert r['status']=='three_FA19_joint_paths_12nativeFA9nativebackward_complete'
for f,v in rec['files'].items():assert sha(D/f)==v['sha256'] and (D/f).stat().st_size==v['bytes']
for f,h in p['files_sha256'].items():assert sha(D/f)==h
assert r['native_FA_entered']==r['native_FA_returned']==12 and r['native_autograd_entered']==r['native_autograd_returned']==9
assert r['model_calls']==r['DT_calls']==r['scorer_calls']==r['FT_calls']==r['generation_calls']==r['compiler_calls']==0
assert all(c['status']=='returned' and c['autograd_function']=='FlashAttnFuncBackward' for c in r['calls'])
z=np.load(D/'vectors.npz',allow_pickle=False);assert sha(D/'vectors.npz')==r['vectors_sha256']
def close(a,b):assert np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<1e-7
def stats(x):return dict(net=float(x.sum()),positive=float(x.clip(min=0).sum()),negative=float(x.clip(max=0).sum()),absolute=float(np.abs(x).sum()))
out={'status':'joint_FA19_paths_independent_CPU_audit_passed_both_rejected','results_sha256':sha(D/'results.json'),'protocol_sha256':sha(D/'protocol.json'),'vectors_sha256':sha(D/'vectors.npz'),'analyzer_sha256':sha(Path(__file__)),'seconds':r['seconds'],'cases':{},'native_calls':{'FA':12,'autograd':9}}
for key,case in r['cases'].items():
    out['cases'][key]={'input_endpoint_replay':case['input_endpoint_replay'],'points':{}}
    for step,row in case['points'].items():
        f={n:z[key+'_'+step+'_'+n] for n in row['fields']}
        for method in ['current','joint_midpoint','joint_gauss2']:
            close(f[method+'_prediction'],f[method+'_qk']+f[method+'_value']);close(f[method+'_error'],f[method+'_prediction']-f['actual'])
        for n,s in row['fields'].items():
            for k,v in s.items():close(stats(f[n])[k],v)
        out['cases'][key]['points'][step]={'actual':float(f['actual'].sum()),'errors':{m:float(f[m+'_error'].sum()) for m in ['current','joint_midpoint','joint_gauss2']}}
    endpoint=out['cases'][key]['points']['B2'];out['cases'][key]['endpoint_relative_signed_errors']={m:e/abs(endpoint['actual']) for m,e in endpoint['errors'].items()}
out['decision']='Reject both fixed cheap approximations; Gauss2 worsens every early/mid case and full endpoint errors remain large, midpoint also worsens MH2/MH3. Do not add nodes or calibrate scores. This rejects these approximations, not all native VJP or all possible operator paths.'
out['limits']=r['limits'];out['peak_allocated_bytes']=r['peak_allocated_bytes']
path=A/'dt_FA19_joint_path_scout_summary_20260909.json';path.write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps({'status':out['status'],'cases':out['cases']}))
