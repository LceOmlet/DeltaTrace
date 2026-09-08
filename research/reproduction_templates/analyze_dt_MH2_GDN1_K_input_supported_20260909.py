"""Independent local contraction checks for the input-supported K scout."""
import hashlib,json
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH2_GDN1_K_input_supported_20260909_v1';G=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH2_GDN1_QK_norm_geometry_20260909_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes())
p,r,rec=[read(D/f) for f in ['protocol.json','results.json','terminal_receipt.json']]
assert rec['proc_exists'] is False and r['protocol']==p and r['status']=='MH2_GDN1_K_input_supported_CPU_scout_complete'
for f,v in rec['files'].items():assert sha(D/f)==v['sha256'] and (D/f).stat().st_size==v['bytes']
for f,h in p['files_sha256'].items():assert sha(D/f)==h
z,g=np.load(D/'vectors.npz',allow_pickle=False),np.load(G/'vectors.npz',allow_pickle=False)
assert sha(D/'vectors.npz')==r['vectors_sha256'] and sha(G/'results.json')==p['geometry_result_sha256']
def close(a,b):assert np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<1e-7
def stats(x):return {'net':float(x.sum()),'positive':float(x.clip(min=0).sum()),'negative':float(x.clip(max=0).sum()),'absolute':float(np.abs(x).sum()),'max_absolute':float(np.abs(x).max())}
out={'status':'MH2_K_input_supported_independent_CPU_audit_passed_local_improvement_only','results_sha256':sha(D/'results.json'),'protocol_sha256':sha(D/'protocol.json'),'vectors_sha256':sha(D/'vectors.npz'),'analyzer_sha256':sha(Path(__file__)),'seconds':r['seconds'],'points':{},'endpoint_real_closure_max':r['endpoint_real_closure_max'],'candidate32_vs64_relative_L2':r['candidate32_vs64_relative_L2']}
for step,row in r['points'].items():
    f={name:z[step+'_'+name] for name in row['fields']}
    close(f['current_error'],g[step+'_k_error'])
    for method in ['current','candidate','candidate32']:close(f[method+'_error'],f[method+'_prediction']-f['actual_native'])
    close(f['candidate_off_chord'],f['J1_off_chord']);close(f['candidate_prediction'],f['endpoint_chord_component']+f['candidate_off_chord'])
    for name,expected in row['fields'].items():
        for key,value in expected.items():close(stats(f[name])[key],value)
    for group,sl in [('prompt',slice(0,480)),('response',slice(480,None))]:
        for name,expected in row['groups'][group].items():
            for key,value in expected.items():close(stats(f[name][:,sl])[key],value)
    out['points'][step]=row
out['decision']='Actual early/mid signed and token-head absolute errors improve, sufficient to justify one frozen NI0/MH2 whole original-metric pilot only. Mid negative error magnitude increases; local net improvement is not whole-method repair. No promotion, no automatic fixed-eight/batch expansion.'
out['limits']='J1 is calculated from the native mathematical normalization definition as an attribution operator, not an executed native backward. Private actual tensors remain remote; local NPZ checks the exposed contractions. Same principle failed prior distinct operators and is not universally validated.'
path=A/'dt_MH2_GDN1_K_input_supported_summary_20260909.json';path.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'status':out['status'],'sha256':sha(path)}))
