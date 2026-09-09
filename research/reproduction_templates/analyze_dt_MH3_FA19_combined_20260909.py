"""Independent actual-mask, full-boundary and native-FA contrast audit, NumPy only."""
import hashlib,json
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;D=A/'snapshot${ARTIFACT_ROOT}/codex_dt_MH3_FA19_combined_20260909_v1';read=lambda p:json.loads(p.read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p,r,receipt=[read(D/f) for f in ['protocol.json','results.json','terminal_receipt.json']]
assert p==r['protocol']==read(A/'dt_MH3_FA19_combined_20260909_protocol.json') and receipt['proc_exists'] is False
assert r['status']=='MH3_FA19_combined_1DT4score11FA_complete'
for f,v in receipt['files'].items():assert sha(D/f)==v['sha256'] and (D/f).stat().st_size==v['bytes']
for f,h in p['files_sha256'].items():assert sha(D/f)==h
assert r['sources_before']==r['sources_after'] and r['weight_stats_before']==r['weight_stats_after']
assert r['DT_entered']==r['DT_returned']==r['model_loads']==r['native_eager_diagnostics']==1
assert r['scorer_entered']==r['scorer_returned']==4 and r['FT_calls']==r['generation_calls']==0
assert r['finite_counts']['DT']=={'entered':8,'returned':8}
assert r['finite_counts']['FLA_backend']['entered']==r['finite_counts']['FLA_backend']['returned']==25
probe=r['FA19_native_probe'];assert probe['native_FA_entered']==probe['native_FA_returned']==len(probe['calls'])==11
assert all(x['status']=='returned' for x in probe['calls'])
z=np.load(D/'vectors.npz',allow_pickle=False);assert sha(D/'vectors.npz')==r['vectors_sha256']
case=r['cases']['morehopqa_3'];info=case['input'];P,T=info['prompt_length'],info['total_length']
def close(a,b):assert np.max(np.abs(np.asarray(a)-np.asarray(b)),initial=0)<1e-7
def stats(x):return dict(net=float(x.sum()),positive=float(x.clip(min=0).sum()),negative=float(x.clip(max=0).sum()),absolute=float(np.abs(x).sum()))
out={'status':'MH3_FA19_combined_independent_CPU_audit_passed','results_sha256':sha(D/'results.json'),'protocol_sha256':sha(D/'protocol.json'),'vectors_sha256':sha(D/'vectors.npz'),'analyzer_sha256':sha(Path(__file__)),'seconds':r['seconds'],'points':{},'replays':probe['replays'],'B1_B2_drift':probe['B1_B2_drift'],'source_vector_drift':r['runs'][0]['source_vector_drift_report_only']}
ids=np.asarray(r['input_freeze_before_model_load']['morehopqa_3']['input_ids'],dtype=np.int64);assert hashlib.sha256(ids.tobytes()).hexdigest()==info['input_sha256']
for step in ['3','10','20']:
    row=probe['points'][step];rec=row['input_receipt'];assert rec==p['frozen_capture_receipts'][step]==case['points'][step]['input_receipt']
    x=ids.copy();x[rec['deleted_positions']]=ids[-1];assert hashlib.sha256(x.tobytes()).hexdigest()==rec['input_sha256']
    for i in range(32):close(z['step'+step+'_decoder_'+str(i)+'_actual_minus_predicted'],z['step'+step+'_boundary_'+str(i+1)]-z['step'+step+'_boundary_'+str(i)])
    internal=row['internal'];close(sum(internal['terms'].values()),internal['prediction_minus_actual']);close(internal['prediction_minus_actual'],-case['decomposition'][step]['decoder_errors']['19'])
    fields={n:z['FA19_'+step+'_'+n] for n in row['fields']}
    close(fields['actual'],fields['RA']+fields['content_actual']);close(fields['route_error_at_V0'],fields['qk_prediction']-fields['R0'])
    close(fields['content_error_at_PC'],fields['v_prediction']-fields['content_actual']);close(fields['reference_interaction'],fields['R0']-fields['RA'])
    close(fields['core_error'],fields['qk_prediction']+fields['v_prediction']-fields['actual'])
    close(fields['core_error'],fields['route_error_at_V0']+fields['content_error_at_PC']+fields['reference_interaction'])
    close(fields['core_error'].sum(),internal['terms']['finite_FA_core_including_seed_cast'])
    for n,s in row['fields'].items():
        for k,v in s.items():close(stats(fields[n])[k],v)
    actual=case['points']['0']['original_native_score']-case['points'][step]['original_native_score']
    predicted=z['morehopqa_3_DT_evaluated'][rec['deleted_positions']].astype(np.float64).sum()
    close(actual-predicted,case['decomposition'][step]['actual_minus_predicted']);close(sum(case['decomposition'][step]['terms'].values()),actual-predicted)
    out['points'][step]={'fields':row['fields'],'internal':internal,'root_prediction_minus_actual':float(predicted-actual),'internal_minus_boundary':case['FA19_internal_minus_boundary'][step]}
out['decision']='Same actual process confirms PV reference-content and routing errors dominate MH3 FA19; current route prediction is positive while native route contrast on retained content is negative. Test a bounded joint-QKV native operator-path hypothesis rather than infer that a local PV endpoint change solves whole attribution.'
out['limits']='Default FA replay and exact ledger closure concern the observed operator/coordinates only. Hybrid values are not full model counterfactuals. Query and key coordinate roles must not be mixed into a source-error claim. No new needle/RISE/MAS curve, candidate propagation, or production timing.'
path=A/'dt_MH3_FA19_combined_summary_20260909.json';path.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'status':out['status'],'points':{s:{k:v['net'] for k,v in x['fields'].items()} for s,x in out['points'].items()}}))
