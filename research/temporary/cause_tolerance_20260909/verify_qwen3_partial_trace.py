"""Verify the original Qwen3 fixed sets, native observations and signed sums."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sha = lambda b: hashlib.sha256(b).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshots', type=Path, required=True)
    p.add_argument('--pilot-only', action='store_true')
    a = p.parse_args()
    rows = {}; reports = {}; cases = []
    jobs = [('qwen3_partial_trace',4)]
    if not a.pilot_only: jobs.append(('qwen3_partial_remaining12',12))
    for job,count in jobs:
        folder = HERE/job; r = json.loads((folder/'results.json').read_bytes())
        assert r['status']=='complete' and len(r['cases'])==count and len(r['calls'])==4*count+1
        assert r['script_sha256']==sha((HERE/'trace_qwen3_partial_deletion.py').read_bytes())
        assert r['vectors_sha256']==sha((folder/'vectors.npz').read_bytes())
        assert r['FT_calls']==r['generation_calls']==r['metric_calls']==0
        for path,digest in r['references']:
            raw = (a.snapshots/path.lstrip('/')).read_bytes(); assert sha(raw)==digest
            for c in json.loads(raw)['cases']: rows[f"{c['dataset']}_{c['index']}"]=c
        for path,digest in r['baseline_sources']['files'].items(): assert sha((ROOT/path).read_bytes())==digest
        assert r['observer_sources']==json.loads((HERE/'qwen3_trace_derivation.json').read_bytes())
        for name,info in r['observer_sources'].items():
            assert sha((HERE/name).read_bytes())==info['derived_sha256']
            assert sha((ROOT/info['original_path']).read_bytes())==info['original_sha256']
        z = np.load(folder/'vectors.npz',allow_pickle=False); assert len(z.files)==2*count
        for c in r['cases']:
            key=c['case']; old=rows[key]
            assert c['input_sha256']==old['input_sha256']
            plain,observed=z[key+'/plain'],z[key+'/observed']
            assert plain.shape==observed.shape==(1,len(old['input_ids']))
            assert np.array_equal(plain,observed) and np.isfinite(plain).all()
            assert len(c['layers'])==36 and {l['layer'] for l in c['layers']}==set(range(36))
            for method in ('DT','FT_K1'):
                indices=old['metrics'][method]['deleted_user_indices'][2]
                assert c['deleted_sets'][method]==indices
                ids=np.asarray(old['input_ids'],dtype=np.int64).copy()
                ids[[old['user_positions'][j] for j in indices]]=ids[-1]
                assert sha(ids.tobytes())==c['partial_input_hashes'][method]==old['metrics'][method]['actual_input_hashes'][2]
                mass=float(plain[0,[old['user_positions'][j] for j in indices]].sum())
                actual=sum(float(x)-float(y) for x,y in zip(c['full_target_logprobs'],c['partial_target_logprobs'][method]))
                saved=c['methods'][method]
                assert mass==saved['allocated_mass'] and actual==saved['actual_drop']
                assert abs(mass-c['boundaries']['0'][method])<1e-7
                groups={'head_seed':c['boundaries']['norm'][method]-actual,
                        'final_norm':c['boundaries']['36'][method]-c['boundaries']['norm'][method]}
                for layer in c['layers']:
                    v=layer['methods'][method]
                    assert abs(sum(v['errors'].values())-(v['input_prediction']-v['output_prediction']))<1e-8
                    replay=v['output_prediction']-c['boundaries'][str(layer['layer']+1)][method]
                    assert replay==v['native_replay_boundary_gap']
                    groups['native_replay_boundary']=groups.get('native_replay_boundary',0.)+replay
                    for name,value in v['errors'].items():
                        field=layer['mixer'] if name=='mixer' else name
                        groups[field]=groups.get(field,0.)+value
                assert groups==saved['signed_family_errors']
                assert abs(sum(groups.values())-(mass-actual))<1e-7
            d,f=c['methods']['DT'],c['methods']['FT_K1']
            family={k:d['signed_family_errors'][k]-f['signed_family_errors'][k] for k in d['signed_family_errors']}
            allocation_gap=d['allocated_mass']-f['allocated_mass']; drop_gap=d['actual_drop']-f['actual_drop']
            assert abs(sum(family.values())-(allocation_gap-drop_gap))<1e-7
            old_gap=(old['metrics']['DT']['scores'][0]-old['metrics']['DT']['scores'][2])-(old['metrics']['FT_K1']['scores'][0]-old['metrics']['FT_K1']['scores'][2])
            cases.append({'case':key,'dataset':old['dataset'],'index':old['index'],'target_tokens':len(c['full_target_logprobs']),
                          'allocation_gap_DT_minus_FT':allocation_gap,'native_FA_drop_gap_DT_minus_FT':drop_gap,
                          'frozen_record_drop_gap_DT_minus_FT':old_gap,
                          'drop_order_agrees_with_frozen_record':np.sign(drop_gap).item()==np.sign(old_gap).item(),
                          'excess_DT_preference':allocation_gap-drop_gap,'family_contributions_to_excess_preference':family})
        reports[job]={'raw_sha256':sha((folder/'results.json').read_bytes()),'vectors_sha256':r['vectors_sha256'],
                      'cases':count,'complete_DT_calls':2*count,'partial_paired_root_calls':2*count,'initialization_root_calls':0,
                      'charged_seconds_excluding_load':sum(c['seconds'] for c in r['calls'] if c['name']!='load')}
    expected=4 if a.pilot_only else 16
    assert len(cases)==len({c['case'] for c in cases})==expected
    summaries=[]
    for dataset in ('niah_mq_q2','morehopqa'):
        selected=[c for c in cases if c['dataset']==dataset]; assert len(selected)==expected//2
        family={k:{'mean_signed':statistics.mean(c['family_contributions_to_excess_preference'][k] for c in selected),
                   'positive_count':sum(c['family_contributions_to_excess_preference'][k]>0 for c in selected),
                   'mean_per_target_token':statistics.mean(c['family_contributions_to_excess_preference'][k]/c['target_tokens'] for c in selected)}
                for k in selected[0]['family_contributions_to_excess_preference']}
        summaries.append({'dataset':dataset,'cases':len(selected),'DT_set_less_destructive':sum(c['native_FA_drop_gap_DT_minus_FT']<0 for c in selected),
                          'drop_order_agreement_to_frozen_record':sum(c['drop_order_agrees_with_frozen_record'] for c in selected),
                          'mean_excess_preference':statistics.mean(c['excess_DT_preference'] for c in selected),'families':family})
    report={'status':'verified','runs':reports,'cases':cases,'summaries':summaries,
            'scope':'Same native B2 default FP16 FA and full cached target; original10%DT/FTK1 sets. FP32-logsoftmax/FP64-sum matches DT objective. Read-only own-code observer, no native implementation replacement.',
            'limits':'Error accounting is not causal mediation or a quality repair. Original Qwen3 metric used eager/native-half scores; native diagnostic drops do not replace its metric curves.'}
    output='qwen3_partial_trace_pilot_summary.json' if a.pilot_only else 'qwen3_partial_trace_summary.json'
    (HERE/output).write_text(json.dumps(report,indent=2)+'\n',newline='\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
