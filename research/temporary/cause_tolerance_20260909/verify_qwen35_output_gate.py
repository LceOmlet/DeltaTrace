"""Check native gate trace against the preceding whole-mixer pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda b:hashlib.sha256(b).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshots',type=Path,required=True);args=p.parse_args()
    folder=HERE/'qwen35_output_gate'
    r=json.loads((folder/'results.json').read_bytes())
    prior=json.loads((HERE/'qwen35_partial_trace/results.json').read_bytes())
    assert r['status']==prior['status']=='complete' and len(r['cases'])==4 and len(r['calls'])==18
    assert r['script_sha256']==sha((HERE/'trace_qwen35_output_gate.py').read_bytes())
    derivation=json.loads((HERE/'qwen35_output_gate_derivation.json').read_bytes())
    assert derivation['derived_sha256']==r['script_sha256']
    assert derivation['source_sha256']==sha((HERE/derivation['source']).read_bytes())
    assert r['references']==prior['references'] and r['baseline_sources']==prior['baseline_sources']
    for name,digest in r['baseline_sources']['files'].items(): assert sha((ROOT/name).read_bytes())==digest
    assert r['vectors_sha256']==sha((folder/'vectors.npz').read_bytes())
    old=np.load(HERE/'qwen35_partial_trace/vectors.npz',allow_pickle=False)
    new=np.load(folder/'vectors.npz',allow_pickle=False)
    assert sorted(new.files)==sorted(old.files)
    inputs={}
    for path,digest in r['references']:
        source=args.snapshots/path.lstrip('/');raw=source.read_bytes();assert sha(raw)==digest
        for c in json.loads(raw)['cases']:
            if c['status']=='complete':inputs[f"{c['dataset']}_{c['index']}"]=c
    drift=[]
    for name in new.files:
        x,y=old[name],new[name]
        assert x.shape==y.shape and np.isfinite(y).all()
        drift.append({'vector':name,'equal':np.array_equal(x,y),'max_absolute':float(abs(x-y).max()),
                      'relative_L2':float(np.linalg.norm(x-y)/np.linalg.norm(x))})
    rows=[]
    for c,previous in zip(r['cases'],prior['cases']):
        key=c['case'];original=inputs[key]
        for field in ('case','input_sha256','deleted_sets','partial_input_hashes','plain_observed_vector_equal'):
            assert c[field]==previous[field]
        assert np.array_equal(new[key+'/plain'],new[key+'/observed'])
        native_drift={}
        for method in ('DT','FT_K1'):
            positions=[original['user_positions'][i] for i in c['deleted_sets'][method]]
            mass=float(new[key+'/plain'][0,positions].sum())
            actual=sum(float(x)-float(y) for x,y in zip(c['full_target_logprobs'],c['partial_target_logprobs'][method]))
            d=c['methods'][method]
            assert abs(d['allocated_mass']-mass)<1e-9 and actual==d['actual_drop']
            assert abs(c['boundaries']['0'][method]-mass)<1e-7
            assert abs(sum(d['signed_family_errors'].values())-(mass-actual))<1e-7
            native_drift[method]={'allocated_mass_difference':mass-previous['methods'][method]['allocated_mass'],
                                  'actual_drop_difference':actual-previous['methods'][method]['actual_drop']}
        totals={}; per_family={}; signed_layers=[]
        for layer in c['layers']:
            for method,d in layer['methods'].items():
                assert abs(sum(d['errors'].values())-(d['input_prediction']-d['output_prediction']))<1e-8
                p=d['mixer_boundary_predictions']; parts=d['mixer_parts']
                assert parts['output_projection']==p['product']-p['output']
                assert parts['output_gate_or_norm_gate']==p['core_raw']+p['gate']-p['product']
                assert parts['GDN_upstream_cast']==p['core_native']-p['core_raw']
                assert abs(sum(parts.values())-d['errors']['mixer'])<1e-8
                if layer['mixer']=='full_attention':assert parts['GDN_upstream_cast']==0
            d,f=[layer['methods'][m] for m in ('DT','FT_K1')]
            signed={k:d['mixer_parts'][k]-f['mixer_parts'][k] for k in d['mixer_parts']}
            signed_layers.append({'layer':layer['layer'],'mixer':layer['mixer'],'parts':signed})
            family=per_family.setdefault(layer['mixer'],{})
            for k,v in signed.items():totals[k]=totals.get(k,0.)+v;family[k]=family.get(k,0.)+v
        mixer_total=sum(c['methods']['DT']['signed_family_errors'][k]-c['methods']['FT_K1']['signed_family_errors'][k] for k in ('linear_attention','full_attention'))
        assert abs(sum(totals.values())-mixer_total)<1e-7
        rows.append({'case':c['case'],'dataset':c['case'].rsplit('_',1)[0],'target_tokens':len(c['full_target_logprobs']),
                     'total_mixer_excess_preference':mixer_total,'parts':totals,'by_mixer':per_family,'layers':signed_layers,
                     'prior_process_scalar_differences':native_drift})
    means=[]
    for dataset in ('niah_mq_q2','morehopqa'):
        selected=[c for c in rows if c['dataset']==dataset];assert len(selected)==2
        means.append({'dataset':dataset,'cases':2,
                      'total_mixer_excess_preference':statistics.mean(c['total_mixer_excess_preference'] for c in selected),
                      'parts':{k:statistics.mean(c['parts'][k] for c in selected) for k in selected[0]['parts']},
                      'by_mixer':{m:{k:statistics.mean(c['by_mixer'][m][k] for c in selected) for k in selected[0]['parts']} for m in ('linear_attention','full_attention')}})
    report={'status':'verified','raw_sha256':sha((folder/'results.json').read_bytes()),'vectors_sha256':r['vectors_sha256'],
            'prior_raw_sha256':sha((HERE/'qwen35_partial_trace/results.json').read_bytes()),
            'same_process_plain_observed_vectors_exact':True,'cross_process_drift':drift,
            'complete_DT_calls':8,'partial_paired_root_calls':8,'native_initialization_calls':1,
            'extra_finite_linear_transpose_calls':4*8,'new_FA_FLA_or_backward_implementations':0,
            'charged_seconds_excluding_load':sum(c['seconds'] for c in r['calls'] if c['name']!='load'),
            'cases':rows,'summaries':means,
            'decision':'Stop the output-gate-only repair hypothesis: NI gate/norm-gate mean is opposing, small relative to the remaining mixer, and smaller than MH. No all32 symmetric-gate candidate or additional metric run is warranted by this pilot.',
            'limits':'Four reused development indices; no changed rule, intervention on hidden states, new quality metric, causal-mediation claim or production speed estimate. GDN norm-gate remains the actual composite boundary; its internal norm and product are not separated. Prior-process vector/scalar values drift; source is not localized, and none of that drift is treated as a candidate effect. Conclusions use same-process controls and original fixed sets.'}
    (HERE/'qwen35_output_gate_summary.json').write_text(json.dumps(report,indent=2)+'\n',newline='\n')
    print(json.dumps(means,indent=2))


if __name__=='__main__':main()
