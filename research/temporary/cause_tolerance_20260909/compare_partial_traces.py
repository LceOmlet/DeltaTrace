"""Compare within-model task patterns; do not equate raw cross-model credits."""
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()


def main():
    sources = {}; summaries = {}; rows = []; contrasts = []
    for model in ('qwen3','qwen35'):
        path = HERE/(model+'_partial_trace_summary.json')
        data = json.loads(path.read_bytes()); assert data['status']=='verified'
        sources[path.name] = sha(path.read_bytes()); summaries[model]=data
        for s in data['summaries']:
            c = [r for r in data['cases'] if r['dataset']==s['dataset']]
            assert len(c)==8
            categories = {'mixers':['full_attention','linear_attention'],
                          'outer_norms':['input_norm','post_norm','final_norm'],
                          'head_seed':['head_seed'],'MLP':['MLP'],
                          'residual_rounding':['residual1_rounding','residual2_rounding'],
                          'native_replay_boundary':['native_replay_boundary']}
            groups = {k:sum(s['families'].get(n,{}).get('mean_signed',0.) for n in fields) for k,fields in categories.items()}
            per_token = {k:sum(s['families'].get(n,{}).get('mean_per_target_token',0.) for n in fields) for k,fields in categories.items()}
            assert abs(sum(groups.values())-s['mean_excess_preference'])<1e-8
            rows.append({'model':model,'dataset':s['dataset'],'n':8,
                         'mean_target_tokens':statistics.mean(r['target_tokens'] for r in c),
                         'DT_set_less_destructive':s['DT_set_less_destructive'],
                         'original_metric_order_agreement':s['drop_order_agreement_to_frozen_record'],
                         'excess_preference':s['mean_excess_preference'],
                         'family_means':groups,'family_means_per_target_token':per_token})
        ni,mh=[next(r for r in rows if r['model']==model and r['dataset']==d) for d in ('niah_mq_q2','morehopqa')]
        contrasts.append({'model':model,'contrast':'NI minus MH within this model',
                          'excess_preference_difference':ni['excess_preference']-mh['excess_preference'],
                          'family_differences':{k:ni['family_means'][k]-mh['family_means'][k] for k in ni['family_means']},
                          'per_target_token_family_differences':{k:ni['family_means_per_target_token'][k]-mh['family_means_per_target_token'][k] for k in ni['family_means']}})
    path=HERE/'initial_evidence.json'; metrics=json.loads(path.read_bytes()); sources[path.name]=sha(path.read_bytes())
    path=HERE/'ni_curve_structure.json'; curves=json.loads(path.read_bytes()); sources[path.name]=sha(path.read_bytes())
    rise=[]
    for r in metrics['means']:
        if r['metric']!='rise':continue
        record=dict(r)
        if r['dataset']=='niah_mq_q2':
            c=[x for x in curves['cases'] if x['family']==r['family']]
            assert len(c)==8
            assert abs(statistics.mean(x['gap'] for x in c)-r['DT_minus_FT_mean'])<1e-12
            record['DT_metric_zero_within10pct']=sum(x['DT_zero_step']<=2 for x in c)
            record['FT_metric_zero_within10pct']=sum(x['FT_zero_step']<=2 for x in c)
        rise.append(record)
    report={'status':'verified_summary_comparison','source_sha256':sources,'rows':rows,'within_model_task_contrasts':contrasts,
            'original_signed_RISE':rise,'new_model_calls':0,'new_metric_calls':0,
            'limits':['Every family comparison is an additive accounting of fixed original10% set errors. It is not an operator intervention or explained RISE fraction.',
                      'The frozen models have different tokenizer/wrapping, native precision and original metric backends. Same original sample indices/full cached targets are retained, but raw total errors are not controlled architectural effects.',
                      'The DT set maximizes its own fixed scores by construction. MH positive controls demonstrate why a positive mixer term alone is not evidence of quality failure.',
                      'Q3 has36FA blocks; Q35 has24GDN+8FA. GDN is neither necessary for the observed NI ranking reversal nor uniquely identified as its cause.',
                      'No quality candidate has been adopted. Default native implementations and all clean method sources are unchanged.']}
    (HERE/'partial_trace_comparison.json').write_text(json.dumps(report,indent=2)+'\n',newline='\n')
    print(json.dumps({'rows':rows,'contrasts':contrasts},indent=2))


if __name__=='__main__':main()
