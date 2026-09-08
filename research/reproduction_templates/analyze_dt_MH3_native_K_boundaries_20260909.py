"""Independent native-mask, boundary telescoping and actual K formula audit, NumPy only."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH3_native_K_boundaries_20260909_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes())
p,r,rec=[read(D/f) for f in ['protocol.json','results.json','terminal_receipt.json']]
assert p==r['protocol']==read(A/'dt_MH3_native_K_boundaries_20260909_protocol.json')
assert rec['proc_exists'] is False and r['status']=='MH3_native_K_boundaries_1DT4score_complete'
for f,row in rec['files'].items():assert sha(D/f)==row['sha256'] and (D/f).stat().st_size==row['bytes']
for f,h in p['files_sha256'].items():assert sha(D/f)==h
assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
assert r['DT_entered']==r['DT_returned']==r['model_loads']==r['native_eager_diagnostics']==1
assert r['scorer_entered']==r['scorer_returned']==4 and r['FT_calls']==r['generation_calls']==0
assert r['finite_counts']['DT']=={'entered':8,'returned':8}
assert r['finite_counts']['FLA_backend']['entered']==r['finite_counts']['FLA_backend']['returned']==25
z=np.load(D/'vectors.npz',allow_pickle=False);assert sha(D/'vectors.npz')==r['vectors_sha256']
c=r['cases']['morehopqa_3'];key='morehopqa_3';info=c['input'];P,T=info['prompt_length'],info['total_length']
source=p['source_standalone'];S=A/'snapshot'/Path(source['results_path']).as_posix().lstrip('/');S=S.parent
for kind in ['results','vectors','protocol']:assert sha(S/(kind+('.npz' if kind=='vectors' else '.json')))==source[kind+'_sha256']
prior=read(S/'results.json');assert info==prior['cases'][key]['input']
curve=prior['cases'][key]['curves']['control'];old=np.load(S/'vectors.npz',allow_pickle=False)
def close(x,y):assert np.max(np.abs(np.asarray(x)-np.asarray(y)),initial=0)<1e-7
def stats(x):return dict(net=float(x.sum()),positive=float(x.clip(min=0).sum()),negative=float(x.clip(max=0).sum()),absolute=float(np.abs(x).sum()),max_absolute=float(np.abs(x).max()))
out={'status':'MH3_native_K_boundaries_independent_CPU_audit_passed','results_sha256':sha(D/'results.json'),'protocol_sha256':sha(D/'protocol.json'),'vectors_sha256':sha(D/'vectors.npz'),'analyzer_sha256':sha(Path(__file__)),'seconds':r['seconds'],'points':{},'source_vector_drift':r['runs'][0]['source_vector_drift_report_only']}
frozen=r['input_freeze_before_model_load'][key];ids=np.asarray(frozen['input_ids'],dtype=np.int64);assert hashlib.sha256(ids.tobytes()).hexdigest()==info['input_sha256']
recovery=read(D/'EOS_export_recovery_receipt.json')
assert recovery['results_sha256']==sha(D/'results.json') and recovery['output_sha256']==sha(D/'recovered_EOS_K.npz')
assert recovery['private_sha256']==c['private_artifact']['sha256'] and recovery['model_calls']==recovery['GPU_calls']==0
raw={n:z['K_raw_'+n] for n in ['x1','f','current_m','candidate_m','candidate32_m']}
raw['x0']=np.load(D/'recovered_EOS_K.npz',allow_pickle=False)['x0'].astype(np.float64)
x0,x1,f=[raw[n] for n in ['x0','x1','f']];radius=lambda x:np.sqrt((x*x).sum(-1,keepdims=True)+1e-6)
r0,r1=radius(x0),radius(x1);d=x1-x0;J=f/r1-x1*(x1*f).sum(-1,keepdims=True)/r1**3
defect=(f*(x1/r1-x0/r0)).sum(-1,keepdims=True)-(J*d).sum(-1,keepdims=True)
length2=(d*d).sum(-1,keepdims=True);candidate=J+d*defect/np.where(length2>0,length2,1)
close(candidate,raw['candidate_m']);close((candidate*d).sum(-1),(f*(x1/r1-x0/r0)).sum(-1))
for step in ['3','10','20']:
    i=int(step);rec=c['points'][step]['input_receipt'];assert rec==curve['input_receipts'][i]==p['frozen_capture_receipts'][step]
    changed=ids.copy();changed[rec['deleted_positions']]=ids[-1];assert hashlib.sha256(changed.tobytes()).hexdigest()==rec['input_sha256']
    assert c['native_K_capture_receipts'][step]['calls']=={'module':1,'conv':1,'FLA':1,'stage':1}
    point=c['decomposition'][step];boundary={n:z['step'+step+'_boundary_'+n] for n in [*[str(i) for i in range(33)],'norm']}
    for layer in range(32):
        values=z['step'+step+'_decoder_'+str(layer)+'_actual_minus_predicted'];close(values,boundary[str(layer+1)]-boundary[str(layer)]);close(values.sum(),point['decoder_errors'][str(layer)])
    w=z[key+'_DT_evaluated'];deleted=rec['deleted_positions'];pred=float(w[deleted].astype(np.float64).sum())
    actual=c['points']['0']['original_native_score']-c['points'][step]['original_native_score']
    close(actual-pred,point['actual_minus_predicted']);close(sum(point['terms'].values()),actual-pred)
    close(point['conditional_error_drift_from_new_score']+point['conditional_error_drift_from_new_vector'],point['actual_minus_predicted']-point['source_actual_minus_predicted'])
    fields={n:z['K_'+step+'_'+n] for n in c['K_diagnostic']['points'][step]['fields']}
    delta=z['K_raw_x0']-z['K_raw_x'+step]
    # K_raw_x0 here is the actual B1 clean value (the saved endpoint x0 name is
    # overwritten by B1 step0 in the export); endpoint EOS remains in private tensors.
    for name,m in [('current',raw['current_m']),('candidate',raw['candidate_m']),('candidate32',raw['candidate32_m'])]:
        close(fields[name+'_prediction'],(m*delta).sum(-1));close(fields[name+'_error'],fields[name+'_prediction']-fields['actual_native'])
    close(fields['current_error'],sum(fields[n] for n in ['scale_error','radial_error','native_precision','current_coefficient_precision']))
    for n,expected in c['K_diagnostic']['points'][step]['fields'].items():
        for k,v in expected.items():close(stats(fields[n])[k],v)
    out['points'][step]={'K':c['K_diagnostic']['points'][step],'current_root_error':-point['actual_minus_predicted'],
        'source_score_drift':point['conditional_error_drift_from_new_score'],'source_vector_error_drift':point['conditional_error_drift_from_new_vector'],
        'decoder_errors_ranked_absolute':point['decoder_errors_ranked_absolute'],
        'prior_same_process_candidate_minus_control_input_prediction':float((old[key+'_candidate_evaluated'].astype(np.float64)-old[key+'_control_evaluated'].astype(np.float64))[deleted].sum())}
out['EOS_export_recovery']=recovery
out['limitations']='K contraction, candidate formula and scalar telescoping checked independently. Original numeric export overwrote only its EOS array name with B1 step0; the missing actual EOS tensor was recovered separately from the SHA-bound private capture, with zero GPU/model calls. Original computation and results preserved. No whole-metric or production-speed claim.'
public={n:z[n] for n in z.files if not n.startswith('K_raw_')}
np.savez_compressed(D/'public_vectors.npz',**public)
out['public_vectors']={'file':'public_vectors.npz','sha256':sha(D/'public_vectors.npz'),'bytes':(D/'public_vectors.npz').stat().st_size,'scope':'Every non-raw array, value-identical to the full source NPZ; full raw tensors remain remote/private and SHA-bound.'}
path=A/'dt_MH3_native_K_boundaries_summary_20260909.json';path.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'status':out['status'],'points':{s:{'root_error':v['current_root_error'],'old_root_change':v['prior_same_process_candidate_minus_control_input_prediction'],'K_current':v['K']['fields']['current_error'],'K_candidate':v['K']['fields']['candidate_error']} for s,v in out['points'].items()}}))
