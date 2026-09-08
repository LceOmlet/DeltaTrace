"""Independent NumPy audit of coordinate-consistent normalization residuals."""
import hashlib,json
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH2_GDN1_QK_norm_geometry_20260909_v1'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();read=lambda f:json.loads(f.read_bytes())
p,r,rec=[read(D/f) for f in ['protocol.json','results.json','terminal_receipt.json']]
assert r['protocol']==p and rec['proc_exists'] is False and r['status']=='MH2_GDN1_QK_native_norm_scale_radial_CPU_complete'
for f,v in rec['files'].items():assert sha(D/f)==v['sha256'] and (D/f).stat().st_size==v['bytes']
for f,h in p['files_sha256'].items():assert sha(D/f)==h
z=np.load(D/'vectors.npz',allow_pickle=False);assert sha(D/'vectors.npz')==r['vectors_sha256']
def stats(x):return {'net':float(x.sum()),'positive':float(x.clip(min=0).sum()),'negative':float(x.clip(max=0).sum()),'absolute':float(np.abs(x).sum()),'max_absolute':float(np.abs(x).max())}
def close(a,b):assert np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<1e-7
out={'status':'MH2_QK_norm_geometry_independent_CPU_audit_passed','results_sha256':sha(D/'results.json'),'protocol_sha256':sha(D/'protocol.json'),'vectors_sha256':sha(D/'vectors.npz'),'analyzer_sha256':sha(Path(__file__)),'seconds':r['seconds'],'points':{},'endpoint':r['endpoint']}
terms=['coefficient_precision','inverse_scale_mismatch','radial_feedback_mismatch','native_normalization_difference']
for step,rows in r['points'].items():
    out['points'][step]={}
    for key,row in rows.items():
        f={name[len(step+'_'+key+'_'):]:z[name] for name in z.files if name.startswith(step+'_'+key+'_')}
        assert all(v.shape==f['error'].shape and v.dtype==np.float64 and np.isfinite(v).all() for v in f.values())
        for t,a,b in [('inverse_scale_mismatch','fixed_scale_response','actual_scale_response'),('radial_feedback_mismatch','fixed_radial_response','actual_radial_response'),('native_normalization_difference','analytic_native_input_normalization','actual_native'),('error','prediction','actual_native')]:close(f[t],f[a]-f[b])
        close(f['coefficient_precision'],f['prediction']-f['fixed_scale_response']-f['fixed_radial_response']);close(sum(f[t] for t in terms),f['error'])
        for name,expected in {**row['terms'],**row['responses'],'error':row['error']}.items():
            for stat,v in expected.items():close(stats(f[name])[stat],v)
        for group,sl in [('prompt',slice(0,480)),('response',slice(480,None))]:
            for name,expected in row['groups'][group].items():
                for stat,v in expected.items():close(stats(f[name][:,sl])[stat],v)
        for top in row['top_token_head_rows']:
            for name,x in f.items():close(top[name],x[0,top['token_position'],top['compact_head']])
        out['points'][step][key]=row
out['interpretation']='K mid error +3.96545 separates inverse scale +1.53172 and radial feedback +2.44973; native analytic/output difference -.015997 and coefficient precision ~-2e-7. All terms share actual token/compact-head coordinates; positive/negative/absolute contributions retained. No direction-specific source attribution or whole MAS improvement inferred.'
out['next']='One analytically integrated L2 Jacobian candidate can address both changing inverse-radius and radial direction along the same fixed EOS/input chord, without numerical path steps or model forwards. It remains a hypothesis on partial directions; freeze and test on saved native K before full propagation.'
path=A/'dt_MH2_GDN1_QK_norm_geometry_summary_20260909.json';path.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'status':out['status'],'sha256':sha(path),'mid_K':out['points']['10']['k']['terms']}))
