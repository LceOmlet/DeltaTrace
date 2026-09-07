"""Post-completion descriptive evidence only; never used to choose attribution weights."""
import hashlib,json,math
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent
path=A/'snapshot${ARTIFACT_ROOT}/codex_pv_interaction_development16_20260907_v1/results.json'
d=json.loads(path.read_text());s=json.loads((A/'pv_interaction_development16_summary_20260907.json').read_text())
assert d['status']=='complete' and s['status']=='verified_complete'
assert hashlib.sha256(path.read_bytes()).hexdigest()==s['raw_sha256']
p=d['protocol'];sym=p['native_methods'][0];rows=[]
for row in d['records']:
    base=np.array(row['native'][sym]['signed_full_sequence']);local=np.array(row['user_positions']);keep=np.array(row['keep_local_indices'])
    gold=np.array(row['gold_eligible_local'],dtype=int);reference_curve=np.array(row['metrics'][sym]['raw_curve'])
    details={'dataset':row['dataset'],'idx':row['idx'],'methods':{}}
    for method in p['native_methods']:
        value=np.array(row['native'][method]['signed_full_sequence']);fullscore=np.array(row['scores'][method]);eligible_value=value[local[keep]]
        reference_score=np.array(row['scores'][sym]);curve=np.array(row['metrics'][method]['raw_curve'])
        details['methods'][method]={
            'local_PV_rule':p['pv_rules'][method],
            'positive_eligible_tokens':int((eligible_value>0).sum()),'negative_eligible_tokens':int((eligible_value<0).sum()),
            'negative_absolute_mass_fraction':float(-eligible_value[eligible_value<0].sum()/max(np.abs(eligible_value).sum(),1e-30)),
            'original_deletion_mask_steps_changed_vs_symmetric':[i for i,(a,b) in enumerate(zip(row['evaluation_masks'][method],row['evaluation_masks'][sym])) if set(a)!=set(b)],
            'raw_curve_delta_vs_symmetric':(curve-reference_curve).tolist(),
            'max_absolute_raw_curve_delta':float(np.max(np.abs(curve-reference_curve))),
            'positive_evaluation_mass':float(fullscore[keep].sum()),
            'positive_evaluation_mass_vs_symmetric':float(fullscore[keep].sum()/max(reference_score[keep].sum(),1e-30))}
        if len(gold):
            chosen=set(row['recovery_topk_local'][method]);refchosen=set(row['recovery_topk_local'][sym]);goldset=set(gold)
            details['methods'][method].update(
                eligible_tokens=len(keep),original_topk_budget=len(chosen),
                original_recovery_combinatorial_ceiling=min(1.,len(chosen)/len(gold)),
                gold_tokens=len(gold),recovered_gold_tokens=len(chosen&goldset),
                newly_recovered_gold_local_indices=sorted((chosen-refchosen)&goldset),
                lost_recovered_gold_local_indices=sorted((refchosen-chosen)&goldset),
                gold_positive_tokens=int((value[local[gold]]>0).sum()),gold_negative_tokens=int((value[local[gold]]<0).sum()),
                gold_positive_score_mass_fraction=float(fullscore[gold].sum()/max(fullscore[keep].sum(),1e-30)))
    rows.append(details)
out={'status':'complete_descriptive_only','raw_sha256':s['raw_sha256'],'cases':rows,
 'scope':'Computed only after all16 original cases and176curves independently verified. Gold is used here for post-hoc interpretation only, never in attribution or parameter selection. No new model calls; no proof route allocation is the unique cause of per-token gains; no causal-sign certification.'}
(A/'pv_interaction_evidence_20260907.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print('Computed complete16 descriptive recovery/negative-mass/original-curve evidence;0newmodelcalls.')
