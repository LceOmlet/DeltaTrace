"""Apply the frozen local gate without selecting layers or tuning an order."""
import argparse,hashlib,json,statistics,subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshots',type=Path,required=True);a=p.parse_args()
    check=subprocess.run([sys.executable,'-B',str(HERE/'verify_internal_traces.py'),'--family','qwen35','--core-order','--snapshots',str(a.snapshots)],capture_output=True,text=True)
    assert check.returncode==0,check.stdout+check.stderr
    raw=HERE/'qwen35_core_order_trace/results.json';r=json.loads(raw.read_bytes())
    checked=json.loads((HERE/'qwen35_core_order_internal_summary.json').read_bytes());assert checked['raw_sha256']==sha(raw.read_bytes())
    cases=[]
    for c in r['cases']:
        data={};families={}
        for method in ('DT','FT_K1'):
            records=[]
            for layer in c['layers']:
                q=layer['methods'][method]['core_order_contrast'];assert len(q)==5
                assert q['current_error']==layer['methods'][method]['internal_parts']['finite_core']
                assert q['current_prediction']==q['current_error']+q['actual_effect']
                assert q['reversed_error']==q['reversed_prediction']-q['actual_effect']
                records.append(q)
            data[method]={'current_signed_error_sum':sum(q['current_error'] for q in records),
                          'reversed_signed_error_sum':sum(q['reversed_error'] for q in records),
                          'current_absolute_error_mean':statistics.mean(abs(q['current_error']) for q in records),
                          'reversed_absolute_error_mean':statistics.mean(abs(q['reversed_error']) for q in records)}
        for family in ('linear_attention','full_attention'):
            layers=[l for l in c['layers'] if l['mixer']==family]
            records=[l['methods'][m]['core_order_contrast'] for l in layers for m in ('DT','FT_K1')]
            families[family]={'current_absolute_error_mean':statistics.mean(abs(q['current_error']) for q in records),
                              'reversed_absolute_error_mean':statistics.mean(abs(q['reversed_error']) for q in records),
                              'current_excess_preference':sum(l['methods']['DT']['core_order_contrast']['current_error']-l['methods']['FT_K1']['core_order_contrast']['current_error'] for l in layers),
                              'reversed_excess_preference':sum(l['methods']['DT']['core_order_contrast']['reversed_error']-l['methods']['FT_K1']['core_order_contrast']['reversed_error'] for l in layers)}
        current_excess=data['DT']['current_signed_error_sum']-data['FT_K1']['current_signed_error_sum']
        reversed_excess=data['DT']['reversed_signed_error_sum']-data['FT_K1']['reversed_signed_error_sum']
        current_abs=statistics.mean(v['current_absolute_error_mean'] for v in data.values())
        reversed_abs=statistics.mean(v['reversed_absolute_error_mean'] for v in data.values())
        cases.append({'case':c['case'],'dataset':c['case'].rsplit('_',1)[0],'methods':data,'by_mixer':families,
                      'current_excess_preference':current_excess,'reversed_excess_preference':reversed_excess,
                      'current_absolute_error_mean':current_abs,'reversed_absolute_error_mean':reversed_abs,
                      'excess_preference_reduced':reversed_excess<current_excess,'absolute_error_reduced':reversed_abs<current_abs})
    ni=[c for c in cases if c['dataset']=='niah_mq_q2'];mh=[c for c in cases if c['dataset']=='morehopqa']
    assert len(ni)==len(mh)==2
    ni_pass=all(c['excess_preference_reduced'] and c['absolute_error_reduced'] for c in ni)
    mh_pass=statistics.mean(c['reversed_absolute_error_mean'] for c in mh)<=statistics.mean(c['current_absolute_error_mean'] for c in mh)
    passed=ni_pass and mh_pass
    report={'status':'verified','raw_sha256':sha(raw.read_bytes()),'instrument_verification_sha256':sha((HERE/'qwen35_core_order_internal_summary.json').read_bytes()),
            'protocol_sha256':sha((HERE/'CORE_ORDER_LOCAL_CONTRAST.md').read_bytes()),'cases':cases,
            'local_gate':{'NI_both_conditions_pass':ni_pass,'MH_absolute_error_not_worse':mh_pass,'passed':passed},
            'decision':'Proceed to a separately frozen complete-method pilot; no quality advantage yet established.' if passed else 'Stop this uniform reversed-core explanation/alternative; no whole-method or metric expansion.',
            'cost':{'extra_finite_FA_calls':32,'extra_finite_FLA_calls':96,**checked['cost']},
            'limits':'Alternative local coefficients at current fixed upstreams are not a complete alternative DT vector. This is local causal rule substitution at actual stored states, not model intervention or proof about all possible core rules. No per-layer rescue or score tuning.'}
    (HERE/'qwen35_core_order_contrast_summary.json').write_text(json.dumps(report,indent=2)+'\n',newline='\n')
    print(json.dumps({'cases':[{k:v for k,v in c.items() if k not in ('methods','by_mixer')} for c in cases],'gate':report['local_gate'],'decision':report['decision']},indent=2))


if __name__=='__main__':main()
