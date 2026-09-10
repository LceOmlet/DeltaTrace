"""Verify actual fixed-set internal traces and preserve prior-process drift."""
import argparse,hashlib,json,statistics
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sha=lambda b:hashlib.sha256(b).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshots',type=Path,required=True)
    p.add_argument('--family',choices=['qwen3','qwen35','all'],default='all')
    p.add_argument('--core-order',action='store_true');a=p.parse_args()
    if a.core_order:assert a.family=='qwen35'
    reports={};summaries=[]
    for family in (['qwen3','qwen35'] if a.family=='all' else [a.family]):
        run_name='qwen35_core_order_trace' if a.core_order else family+'_internal_trace'
        folder=HERE/run_name;r=json.loads((folder/'results.json').read_bytes())
        count=36 if family=='qwen3' else 32
        assert r['status']=='complete' and len(r['cases'])==4 and len(r['calls'])==(17 if family=='qwen3' else 18)
        driver='trace_qwen35_core_order.py' if a.core_order else 'trace_'+family+'_internal_normalization.py'
        assert r['script_sha256']==sha((HERE/driver).read_bytes())
        assert r['FT_calls']==r['generation_calls']==r['metric_calls']==0
        assert r['vectors_sha256']==sha((folder/'vectors.npz').read_bytes())
        for name,digest in r['baseline_sources']['files'].items():assert sha((ROOT/name).read_bytes())==digest
        if family=='qwen3':
            for name,d in r['observer_sources'].items():
                assert sha((HERE/name).read_bytes())==d['derived_sha256']
                assert sha((ROOT/d['original_path']).read_bytes())==d['original_sha256']
        else:
            prefix='qwen35_core_order_trace' if a.core_order else 'qwen35_internal_trace'
            d=r['observer_derivation'];assert d==json.loads((HERE/(prefix+'_derivation.json')).read_bytes())
            assert d['driver']['derived_sha256']==r['script_sha256']
            assert sha((HERE/(prefix+'_controller.py')).read_bytes())==d['controller']['derived_sha256']
            assert sha((ROOT/d['controller']['original_path']).read_bytes())==d['controller']['original_sha256']
        inputs={}
        for path,digest in r['references']:
            raw=(a.snapshots/path.lstrip('/')).read_bytes();assert sha(raw)==digest
            for c in json.loads(raw)['cases']:
                if c['status']=='complete':inputs[f"{c['dataset']}_{c['index']}"]=c
        z=np.load(folder/'vectors.npz',allow_pickle=False);assert len(z.files)==8
        prior_name='qwen35_internal_trace' if a.core_order else 'qwen3_partial_trace' if family=='qwen3' else 'qwen35_output_gate'
        prior=json.loads((HERE/prior_name/'results.json').read_bytes())
        oldz=np.load(HERE/prior_name/'vectors.npz',allow_pickle=False)
        oldcases={c['case']:c for c in prior['cases']}
        cases=[]
        for c in r['cases']:
            key=c['case'];original=inputs[key];old=oldcases[key]
            x,y=z[key+'/plain'],z[key+'/observed']
            assert np.array_equal(x,y) and np.isfinite(x).all() and x.shape==(1,len(original['input_ids']))
            assert c['input_sha256']==original['input_sha256'] and len(c['layers'])==count
            assert {l['layer'] for l in c['layers']}==set(range(count))
            for method in ('DT','FT_K1'):
                deleted=original['metrics'][method]['deleted_user_indices'][2];assert c['deleted_sets'][method]==deleted
                indices=[original['user_positions'][j] for j in deleted]
                ids=np.asarray(original['input_ids'],dtype=np.int64).copy();ids[indices]=ids[-1]
                assert sha(ids.tobytes())==c['partial_input_hashes'][method]==original['metrics'][method]['actual_input_hashes'][2]
                mass=float(x[0,indices].sum());actual=sum(float(u)-float(v) for u,v in zip(c['full_target_logprobs'],c['partial_target_logprobs'][method]))
                saved=c['methods'][method]
                assert abs(saved['allocated_mass']-mass)<1e-9 and saved['actual_drop']==actual
                assert abs(c['boundaries']['0'][method]-mass)<1e-7
                groups={'head_seed':c['boundaries']['norm'][method]-actual,
                        'final_norm':c['boundaries'][str(count)][method]-c['boundaries']['norm'][method]}
                for layer in c['layers']:
                    v=layer['methods'][method]
                    assert abs(sum(v['errors'].values())-(v['input_prediction']-v['output_prediction']))<1e-8
                    gap=v['output_prediction']-c['boundaries'][str(layer['layer']+1)][method]
                    assert gap==v['native_replay_boundary_gap']
                    groups['native_replay_boundary']=groups.get('native_replay_boundary',0.)+gap
                    for name,val in v['errors'].items():
                        field=layer['mixer'] if name=='mixer' else name;groups[field]=groups.get(field,0.)+val
                    assert abs(sum(v['mixer_parts'].values())-v['errors']['mixer'])<1e-8
                    assert abs(sum(v['internal_parts'].values())-v['mixer_parts']['remaining_mixer'])<1e-8
                assert groups==saved['signed_family_errors']
                assert abs(sum(groups.values())-(mass-actual))<1e-7
            totals={};by_mixer={};reconstruction=[]
            for layer in c['layers']:
                parts={}
                for section in ('mixer_parts','internal_parts'):
                    d,f=[layer['methods'][m][section] for m in ('DT','FT_K1')]
                    for k in d:
                        if k!='remaining_mixer':parts[k]=d[k]-f[k]
                group=by_mixer.setdefault(layer['mixer'],{})
                for k,v in parts.items():totals[k]=totals.get(k,0.)+v;group[k]=group.get(k,0.)+v
                if family=='qwen35':
                    descriptor=layer['normalization_coefficient_reconstruction']
                    if layer['mixer']=='full_attention':
                        assert np.isfinite(descriptor['relative_L2']) and np.isfinite(descriptor['max_absolute'])
                        reconstruction.append(descriptor)
                    else:assert descriptor=={'grouped_coefficients_exact':True}
            total_mixer=sum(c['methods']['DT']['signed_family_errors'][k]-c['methods']['FT_K1']['signed_family_errors'][k] for k in by_mixer)
            assert abs(sum(totals.values())-total_mixer)<1e-7
            previous=oldz[key+'/plain']
            cases.append({'case':key,'dataset':original['dataset'],'target_tokens':len(c['full_target_logprobs']),
                          'total_mixer_excess_preference':total_mixer,'parts':totals,'by_mixer':by_mixer,
                          'FA_coefficient_reconstruction':reconstruction,
                          'prior_vector_relative_L2':float(np.linalg.norm(x-previous)/np.linalg.norm(previous)),
                          'prior_vector_max_absolute':float(abs(x-previous).max()),
                          'prior_drop_differences':{m:c['methods'][m]['actual_drop']-old['methods'][m]['actual_drop'] for m in ('DT','FT_K1')}})
        means=[]
        for dataset in ('niah_mq_q2','morehopqa'):
            selected=[c for c in cases if c['dataset']==dataset];assert len(selected)==2
            entry={'model':family,'dataset':dataset,'n':2,
                   'total_mixer_excess_preference':statistics.mean(c['total_mixer_excess_preference'] for c in selected),
                   'parts':{k:statistics.mean(c['parts'][k] for c in selected) for k in selected[0]['parts']},
                   'by_mixer':{m:{k:statistics.mean(c['by_mixer'][m][k] for c in selected) for k in selected[0]['parts']} for m in selected[0]['by_mixer']}}
            means.append(entry);summaries.append(entry)
        report={'status':'verified','raw_sha256':sha((folder/'results.json').read_bytes()),'vectors_sha256':r['vectors_sha256'],
                'prior_raw_sha256':sha((HERE/prior_name/'results.json').read_bytes()),'cases':cases,'summaries':means,
                'cost':{'complete_DT_calls':8,'partial_paired_native_forwards':8,'native_initializations':int(family=='qwen35'),
                        'charged_seconds_excluding_load':sum(c['seconds'] for c in r['calls'] if c['name']!='load')},
                'limits':'Read-only accounting at original frozen10%sets/full targets. Same-process controls pass. Prior-process drift is reported and not a method effect. No rule candidate or quality metric is tested.'}
        output_name='qwen35_core_order_internal_summary.json' if a.core_order else family+'_internal_trace_summary.json'
        (HERE/output_name).write_text(json.dumps(report,indent=2)+'\n',newline='\n')
        reports[family]=report
    if a.family=='all':
        comparison={'status':'verified','sources':{f:sha((HERE/(f+'_internal_trace_summary.json')).read_bytes()) for f in reports},'summaries':summaries,
                    'scope':'Four matched original indices per model; error terms are not architectural causal effects or percentages of RISE. No expansion/adoption follows from magnitude alone.'}
        (HERE/'internal_trace_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n',newline='\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
