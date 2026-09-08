"""Aggregate only audited, complete original fixed-case results; no rescoring."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
A=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes())
parser=argparse.ArgumentParser();parser.add_argument('--require-eight',action='store_true');args=parser.parse_args()
fixed=[f'{dataset}_{i}' for dataset in ['niah_mq_q2','morehopqa'] for i in range(4)]
names=['dt_GDN1_K_whole_pilot_summary_20260909.json']+[f'dt_GDN1_K_remaining_summary_20260909_s{i}.json' for i in range(3)]
cases={};jobs=[];frozen=None
for name in names:
    if not (A/name).exists():continue
    s=read(A/name);assert s['status']=='independent_GDN1_K_whole_pilot_audit_passed'
    directory='codex_dt_GDN1_K_whole_pilot_20260909_v1' if 'whole_pilot' in name else f'codex_dt_GDN1_K_remaining_20260909_s{name.split("_s")[-1].split(".")[0]}_v1'
    D=A/'snapshot/tmp'/directory;r,p,rec=[read(D/f) for f in ['results.json','protocol.json','terminal_receipt.json']]
    assert r['protocol']==p and rec['proc_exists'] is False and sha(D/'results.json')==s['results_sha256']
    sources={k:v for k,v in p['files_sha256'].items() if k!='study.py'}
    if frozen is None:frozen=sources
    else:assert frozen==sources,'Candidate/source changed across fixed cases'
    jobs.append({'name':directory.removeprefix('codex_'),'summary':name,'summary_sha256':sha(A/name),'results_sha256':s['results_sha256'],'wall_seconds':s['wall_seconds'],'counts':s['counts']})
    for key,c in s['cases'].items():
        assert key in fixed and key not in cases
        methods={m:{k:q[k] for k in ['RISE','MAS','needle']} for m,q in c['methods'].items()}
        costs={x['method']:{'seconds':x['outer_seconds'],'peak_allocated':x['memory_cost']['peak_allocated_full_model_resident']} for x in s['runs'] if x['case']==key}
        cases[key]={'methods':methods,'candidate_minus_control':c['candidate_minus_control'],'MAS_parts_delta':c['candidate_minus_control_MAS_decomposition'],
            'two_actual_backgrounds':{bg:{m:{k:q[k] for k in ['MAE_AUC','signed_error_AUC']} for m,q in rows.items()} for bg,rows in c['both_actual_mask_families'].items()},
            'native_root_drift':c['same_case_native_root_drift'],'score_endpoint_deltas':c['candidate_minus_control_native_score_endpoints'],
            'historical_FT_reference_only':c['historical_FT_reference_only'],'single_observation_cost':costs,'summary_sha256':sha(A/name)}
missing=[k for k in fixed if k not in cases]
if args.require_eight:assert not missing,missing
out={'status':'GDN1_K_fixed_eight_original_audit_passed' if not missing else 'GDN1_K_fixed_case_partial_audit','analyzer_sha256':sha(Path(__file__)),'fixed_cases':fixed,'completed':list(cases),'unexecuted_or_unaudited':missing,'jobs':jobs,'cases':cases,'groups':{},'wall_seconds':sum(x['wall_seconds'] for x in jobs)}
for group,keys in [('NI',[k for k in fixed if k.startswith('niah')]),('MH',[k for k in fixed if k.startswith('more')]),('all',fixed)]:
    keys=[k for k in keys if k in cases]
    if not keys:continue
    row={'n':len(keys),'means':{m:{metric:float(np.mean([cases[k]['methods'][m][metric] for k in keys])) for metric in ['RISE','MAS']} for m in ['control','candidate']},'counts':{}}
    for metric in ['RISE','MAS']:
        d=[cases[k]['candidate_minus_control'][metric] for k in keys];row['counts'][metric]={'better':sum(x<0 for x in d),'equal':sum(x==0 for x in d),'worse':sum(x>0 for x in d),'max_increase':max(d),'mean_change':float(np.mean(d))}
    gold=[k for k in keys if cases[k]['methods']['control']['needle'] is not None]
    row['needle']={'n':len(gold),'eligible_gold_denominator':sum(cases[k]['methods']['control']['needle']['eligible_gold_denominator'] for k in gold) if gold else None,
        'hits':{m:sum(cases[k]['methods'][m]['needle']['hits'] for k in gold) for m in ['control','candidate']} if gold else None,
        'per_case_regressions':[k for k in gold if cases[k]['candidate_minus_control']['needle']<0]}
    row['fixed_background_MAE_improvements']={bg:sum(cases[k]['two_actual_backgrounds'][bg]['candidate']['MAE_AUC']<cases[k]['two_actual_backgrounds'][bg]['control']['MAE_AUC'] for k in keys) for bg in ['control','candidate']}
    out['groups'][group]=row
out['cost_scope']='All full quality job wall time and per-run cold/order-confounded timings retained. Unchanged FA/FLA call counts do not prove runtime speed. No new warmed or true batch acceptance in these jobs.'
out['quality_scope']='Previously used fixed development set only. Original author metrics and gold, no model/generation/rescoring here. Historical FT has matching inputs but a different process and original internal target, not contemporaneous control. Local K and whole metrics do not prove every MAS mismatch fixed.'
path=A/'dt_GDN1_K_fixed_eight_summary_20260909.json';path.write_text(json.dumps(out,indent=2,allow_nan=False))
print(json.dumps({'status':out['status'],'sha256':sha(path),'completed':len(cases),'missing':missing,'groups':out['groups']}))
