"""Audit the rejected analytic normalization path and actual radius backgrounds."""
import hashlib,json
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH2_GDN1_K_path_scout_20260909_v1';G=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH2_GDN1_QK_norm_geometry_20260909_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes())
p,r,rec=[read(D/f) for f in ['protocol.json','results.json','terminal_receipt.json']]
assert rec['proc_exists'] is False and r['protocol']==p and r['status']=='MH2_GDN1_K_analytic_path_CPU_scout_complete'
for f,v in rec['files'].items():assert sha(D/f)==v['sha256'] and (D/f).stat().st_size==v['bytes']
for f,h in p['files_sha256'].items():assert sha(D/f)==h
z,g=np.load(D/'vectors.npz',allow_pickle=False),np.load(G/'vectors.npz',allow_pickle=False)
assert sha(D/'vectors.npz')==r['vectors_sha256'] and sha(G/'results.json')==p['geometry_result_sha256']
def close(a,b):assert np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<1e-7
def stats(x):return {'net':float(x.sum()),'positive':float(x.clip(min=0).sum()),'negative':float(x.clip(max=0).sum()),'absolute':float(np.abs(x).sum()),'max_absolute':float(np.abs(x).max())}
out={'status':'MH2_K_path_scout_independent_CPU_audit_passed_rejected','results_sha256':sha(D/'results.json'),'protocol_sha256':sha(D/'protocol.json'),'vectors_sha256':sha(D/'vectors.npz'),'analyzer_sha256':sha(Path(__file__)),'seconds':r['seconds'],'points':{},'endpoint_real_closure_max':r['endpoint_real_closure_max'],'independent_quadrature':r['independent_quadrature'],'candidate32_vs64_relative_L2':r['candidate32_vs64_relative_L2']}
for step,row in r['points'].items():
    f={name:z[step+'_'+name] for name in row['fields']}
    close(f['current_error'],g[step+'_k_error'])
    for method in ['current','candidate','candidate32']:close(f[method+'_error'],f[method+'_prediction']-f['actual_native'])
    for name,expected in row['fields'].items():
        for key,value in expected.items():close(stats(f[name])[key],value)
    for group,sl in [('prompt',slice(0,480)),('response',slice(480,None))]:
        for name,expected in row['groups'][group].items():
            for key,value in expected.items():close(stats(f[name][:,sl])[key],value)
    out['points'][step]=row
out['actual_partial_radius_geometry']={}
for step in ['3','10']:
    f={k:g[step+'_k_'+k] for k in ['error','r0','r1','rC','rA','partial_off_chord_norm']};w=np.abs(f['error'])
    near=np.abs(f['rA']-f['r1'])<np.abs(f['rA']-f['r0'])
    out['actual_partial_radius_geometry'][step]={'closer_input_radius_absolute_error_weighted_fraction':float((w*near).sum()/w.sum()),
        'relative_radius_distance_to_input_weighted':float((w*np.abs(f['rA']/f['r1']-1)).sum()/w.sum()),
        'relative_radius_distance_to_EOS_weighted':float((w*np.abs(f['rA']/f['r0']-1)).sum()/w.sum()),
        'relative_off_chord_distance_to_input_radius_weighted':float((w*f['partial_off_chord_norm']/f['r1']).sum()/w.sum())}
out['decision']='Reject analytic fixed-path K replacement: actual early/mid signed and token-head absolute errors both increase. Independent integral and endpoint closure validate formula only. Default/runtime unchanged; no full model, metric or batch claim.'
out['limits']='Local artifacts independently verify contractions/identities and coordinate-consistent statistics, not private large native states. Radius-background weighting is descriptive on the fixed observed masks, never used to fit a coefficient or choose cases.'
path=A/'dt_MH2_GDN1_K_path_scout_summary_20260909.json';path.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'status':out['status'],'sha256':sha(path),'geometry':out['actual_partial_radius_geometry']}))
