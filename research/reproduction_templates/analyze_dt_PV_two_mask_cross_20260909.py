"""CPU-only 2x2 actual-deletion-background audit of eight completed PV19 cases."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from analyze_dt_original_regression_20260909 import auc,curve_audit
A=Path(__file__).resolve().parent
read=lambda f:json.loads(Path(f).read_bytes())
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
def metrics(e):
    e=np.asarray(e,dtype=np.float64)
    return {'absolute_error_AUC':auc(abs(e)),'signed_error_AUC':auc(e),
        'max_absolute_error':float(abs(e).max()),'interior19_mean_absolute_error':float(abs(e[1:20]).mean()),
        'clean_error':float(e[0]),'allEOS_error':float(e[20])}
def compare(a,b):
    d=abs(b)-abs(a);inner=d[1:20]
    return {'candidate_minus_control_absolute_error_AUC':auc(d),
      'interior19_candidate_better_equal_worse':[int((inner<-1e-9).sum()),int((abs(inner)<=1e-9).sum()),int((inner>1e-9).sum())],
      'interior19_worse_steps':(np.flatnonzero(inner>1e-9)+1).tolist(),
      'max_point_absolute_error_change':float(d.max()),'absolute_error_change':d.tolist()}
def needle_positions(key,case,z):
    keep=case['input']['keep'];gold=set(case['gold'])&set(keep)
    if not gold:return None
    k=max(1,min(len(keep),math.ceil(.1*len(keep))));out={};sets={}
    for m in ('control','candidate'):
        w=z[key+'_'+m+'_evaluated'];v=np.maximum(w[keep],0);threshold=np.sort(v)[-k]
        above={keep[i] for i in np.flatnonzero(v>threshold)};ties={keep[i] for i in np.flatnonzero(v==threshold)}
        # This position ledger requires an unambiguous threshold; do not fake torch topk tie ordering.
        assert len(above)+len(ties)==k,(key,m,'ambiguous topK threshold')
        selected=above|ties;sets[m]=selected
        assert abs(len(selected&gold)/len(gold)-case['curves'][m]['needle'])<1e-12
        out[m]={'k':k,'threshold':float(threshold),'threshold_ties':len(ties),'selected':sorted(selected),
          'selected_gold':sorted(selected&gold),'hits':len(selected&gold),'gold_denominator':len(gold)}
    entered=sets['candidate']-sets['control'];exited=sets['control']-sets['candidate']
    def positions(ix):
        ans=[]
        for pos in sorted(ix):
            row={'position':pos,'eligible_gold':pos in gold}
            for m in ('control','candidate'):
                w=z[key+'_'+m+'_evaluated'];row[m]={'score':float(w[pos]),
                  'rank_min':1+int((w[keep]>w[pos]).sum()),'rank_max':int((w[keep]>=w[pos]).sum())}
            ans.append(row)
        return ans
    out.update(entered=positions(entered),exited=positions(exited),gold_positions_lost=sorted(exited&gold),
               gold_positions_gained=sorted(entered&gold),all_eligible_gold_positions=sorted(gold),
               scope='Actual FP32 scores with the original needle clamp/top10% budget. Threshold sets are unambiguous here; no torch tie ordering inferred, no token text decoded.')
    return out

jobs=['codex_dt_PV_layer19_whole_pilot_20260909_v1','codex_dt_PV_layer19_remaining_regression_20260909_s0_v1','codex_dt_PV_layer19_remaining_regression_20260909_s1_v1','codex_dt_PV_layer19_remaining_regression_20260909_s2_v1']
report={'status':'eight_completed_cases_two_mask_CPU_audit','GPU_model_scorer_calls':0,'source_actual_DT_calls':16,'source_actual_scorer_calls':336,'jobs':[],'cases':{},
  'definition':'E[a,b,t]=sum evaluated_FP32_score_a[deleted_positions_b,t] in CPU FP64 - (actual original B1 score_b,0 - score_b,t). Both b=C_masks and b=P0_masks use their OWN actually recorded scorer curve; no score substitution or additional forwards.',
  'AUC_axis':'Original uniformly spaced21-step axis t/20, trapezoidal. Interior19 excludes clean/allEOS. Comparison tolerance1e-9 is CPU arithmetic equality only, not a quality-significance threshold.',
  'selection_limit':'C_masks are chosen by C scores. Improvement on this selected family does not imply improvement on P0-selected states or arbitrary coalitions. Even both observed families improving does not certify all A. RISE evaluates the realized deletion ordering; MAS also depends on signed density normalization and author postprocessing.',
  'source_verifier_sha256':sha(A/'analyze_dt_original_regression_20260909.py')}
for name in jobs:
    D=A/'snapshot/tmp'/name;p=read(D/'protocol.json');r=read(D/'results.json');receipt=read(D/'terminal_receipt.json')
    assert r['protocol']==p and r['DT_returned']==4 and r['scorer_returned']==84
    assert 'complete' in r['status'] and r['sources_before']==r['sources_after']
    for f,v in receipt['files'].items():assert sha(D/f)==(v if isinstance(v,str) else v['sha256']),f
    for f,v in p['files_sha256'].items():assert sha(D/f)==v,f
    z=np.load(D/'vectors.npz');assert sha(D/'vectors.npz')==r['vectors_sha256']
    job={'directory':name,'results_sha256':sha(D/'results.json'),'vectors_sha256':sha(D/'vectors.npz'),
      'protocol_sha256':sha(D/'protocol.json'),'receipt_sha256':sha(D/'terminal_receipt.json')};report['jobs'].append(job)
    for key,case in r['cases'].items():
        assert key not in report['cases'];info=case['input'];f=r['input_freeze_before_model_load'][key]
        ids=np.asarray(f['input_ids'],dtype=np.int64);assert hashlib.sha256(ids.tobytes()).hexdigest()==info['input_sha256']
        inputs={key:(ids,int(f['target_ids'][-1]))};audits={};wm={};fm={}
        for m in ('control','candidate'):
            run=next(x for x in r['runs'] if x['case']==key and x['method']==m)
            alias='DT_'+m;v={key+'_'+alias+tail:z[key+'_'+m+tail] for tail in ('_full','_evaluated')}
            audits[m]=curve_audit(key,alias,case['curves'][m],case,run['deletion_audit'],v,inputs)
            wm[m]=z[key+'_'+m+'_evaluated'].astype(np.float64);fm[m]=z[key+'_'+m+'_full']
        out={'source_job':name,'input':info,'own_original_metrics':{m:{k:audits[m][k] for k in ('RISE','MAS','needle')} for m in audits},'backgrounds':{}}
        for background in ('control','candidate'):
            c=case['curves'][background];g=np.asarray(c['scores'],dtype=np.float64);actual=g[0]-g
            sets=[x['deleted_positions'] for x in c['input_receipts']]
            pred={m:np.asarray([wm[m][ix].sum() for ix in sets]) for m in wm}
            err={m:pred[m]-actual for m in wm}
            fullerr={m:np.asarray([fm[m][ix].sum() for ix in sets])-actual for m in fm}
            if background=='control':
                for i,point in enumerate(case['fixed_control_masks']['points']):
                    assert point['input_receipt']==c['input_receipts'][i]
                    for m in wm:assert abs(err[m][i]-point[m]['prediction_minus_actual'])<1e-7
            out['backgrounds'][background]={'real_score_curve':g.tolist(),'actual_drop':actual.tolist(),'input_receipts':c['input_receipts'],
              'predictions':{m:x.tolist() for m,x in pred.items()},'errors':{m:x.tolist() for m,x in err.items()},
              'summary':{m:metrics(x) for m,x in err.items()},'candidate_vs_control':compare(err['control'],err['candidate']),
              'full_FP64_vector_check':{m:{'summary':metrics(fullerr[m]),'max_effect_of_evaluated_FP32_cast':float(abs(fullerr[m]-err[m]).max())} for m in wm}}
        same=[]
        for i,rc in enumerate(case['curves']['control']['input_receipts']):
            for j,rp in enumerate(case['curves']['candidate']['input_receipts']):
                if rc['input_sha256']==rp['input_sha256']:
                    same.append({'control_step':i,'candidate_step':j,'score_candidate_minus_control':case['curves']['candidate']['scores'][j]-case['curves']['control']['scores'][i]})
        out['same_input_score_comparisons']=same
        out['own_candidate_minus_control']={k:audits['candidate'][k]-audits['control'][k] for k in ('RISE','MAS')}
        out['own_actual_drop_AUC']={m:auc(np.asarray(case['curves'][m]['scores'][0])-np.asarray(case['curves'][m]['scores'])) for m in wm}
        out['needle_positions']=needle_positions(key,case,z)
        bg=out['backgrounds'];cd=bg['control']['candidate_vs_control']['candidate_minus_control_absolute_error_AUC'];pd=bg['candidate']['candidate_vs_control']['candidate_minus_control_absolute_error_AUC']
        acc=bg['control']['summary']['control']['absolute_error_AUC'];apc=bg['control']['summary']['candidate']['absolute_error_AUC']
        acp=bg['candidate']['summary']['control']['absolute_error_AUC'];app=bg['candidate']['summary']['candidate']['absolute_error_AUC']
        out['own_path_MAE_decomposition']={'own_candidate_minus_own_control':app-acc,
          'fixed_P0_masks_coefficient_change':app-acp,'C_coefficient_fixed_mask_path_change':acp-acc,
          'fixed_C_masks_coefficient_change':apc-acc,'P0_coefficient_fixed_mask_path_change':app-apc,
          'closure_via_P0_background':(app-acc)-((app-acp)+(acp-acc)),
          'scope':'Exact algebra on observed MAE_AUC, not a causal model decomposition. A different selected mask path changes the actual states and drops. Both additive orders are saved; do not treat one chosen allocation as a unique root cause.'}
        out['observed_pattern']={'improves_C_background_AUC':cd<0,'improves_P0_background_AUC':pd<0,
          'candidate_own_RISE_regresses':out['own_candidate_minus_control']['RISE']>0,'candidate_own_MAS_regresses':out['own_candidate_minus_control']['MAS']>0,
          'cannot_infer':'AUC net improvement can coexist with worse individual points; inspected masks are selected, not all coalitions. Original MAS is not raw logprob MAE.'}
        report['cases'][key]=out
assert len(report['cases'])==8
path=A/'dt_PV_two_mask_cross_summary_20260909.json';path.write_text(json.dumps(report,indent=2,allow_nan=False))
for key,c in report['cases'].items():
    print(key,json.dumps({'own_delta':c['own_candidate_minus_control'],**{b:{'C':c['backgrounds'][b]['summary']['control']['absolute_error_AUC'],'P0':c['backgrounds'][b]['summary']['candidate']['absolute_error_AUC'],'wins_ties_losses':c['backgrounds'][b]['candidate_vs_control']['interior19_candidate_better_equal_worse']} for b in ('control','candidate')},'needle_loss':c['needle_positions']['gold_positions_lost'] if c['needle_positions'] else None}))
print(json.dumps({'summary':str(path),'sha256':sha(path),'script_sha256':sha(__file__)}))
