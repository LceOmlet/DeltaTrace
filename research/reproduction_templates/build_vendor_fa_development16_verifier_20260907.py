"""Apply the full integration checks to all sixteen fixed development cases."""
from pathlib import Path
A=Path(__file__).resolve().parent
source=(A/'verify_vendor_fa_end_to_end_20260907.py').read_text()
for old,new in [
 ('codex_vendor_fa_end_to_end_20260907_v1','codex_vendor_fa_development16_20260907_v1'),
 ('vendor_fa_end_to_end_protocol_20260907.json','vendor_fa_development16_protocol_20260907.json'),
 ('vendor_fa_end_to_end_probe_20260907.py','vendor_fa_development16_20260907.py'),
 ('vendor_fa_end_to_end_summary_20260907.json','vendor_fa_development16_summary_20260907.json'),
 ('vendor_fa_end_to_end_numeric_20260907.json','vendor_fa_development16_numeric_20260907.json'),
 ("==26 and all", "==130 and all"),
 ("==468\n", "==2340\n"),
 ("out['curves_verified']==6", "out['curves_verified']==32"),
 ("actual_FA_forwards=26*108+3*36,finite_kernel_calls=468*3", "actual_FA_forwards=130*108+16*36,finite_kernel_calls=2340*3"),
 ('Full signed vectors, original six curves','Full signed vectors, original thirty-two curves'),
 ('Three original development cases only. No full16newquality, multi-example, long-input, independent-quality or causal-token-sign confirmation.',
  'All sixteen original development cases. No multi-example, long-input, independent-quality or causal-token-sign confirmation.'),
]:
    assert old in source,old
    source=source.replace(old,new)
extra='''
out['means']={}
for dataset in ['niah_mq_q2','morehopqa']:
    rows=[r for r in d['records'] if r['dataset']==dataset];assert len(rows)==8
    methods=['dense','finite']+list(rows[0]['historical_FT_control']['metrics'])
    out['means'][dataset]={}
    for method in methods:
        values=[r['metrics'][method] if method in ['dense','finite'] else r['historical_FT_control']['metrics'][method] for r in rows]
        out['means'][dataset][method]={key:statistics.mean(v[key] for v in values) if all(v[key] is not None for v in values) else None for key in ['rise','mas','recovery']}
ni=out['means']['niah_mq_q2'];mh=out['means']['morehopqa'];ft=[m for m in ni if m.startswith('flashtrace_')]
out['unchanged_development_joint_gate']={mode:{'NI_recovery_at_least_best_FT':ni[mode]['recovery']>=max(ni[m]['recovery'] for m in ft),
    'MH_RISE_below_best_FT':mh[mode]['rise']<min(mh[m]['rise'] for m in ft),
    'MH_MAS_below_best_FT':mh[mode]['mas']<min(mh[m]['mas'] for m in ft)} for mode in ['dense','finite']}
out['cost_overview']={'median_finite_to_same_job_dense_ratio':statistics.median(r['finite_to_dense_time_ratio'] for r in out['cases']),
    'worst_finite_to_same_job_dense_ratio':max(r['finite_to_dense_time_ratio'] for r in out['cases']),
    'cases_faster':sum(r['finite_to_dense_time_ratio']<1 for r in out['cases']),
    'min_peak_saving_bytes':min(r['full_peak_saved_bytes'] for r in out['cases']),
    'max_peak_saving_bytes':max(r['full_peak_saved_bytes'] for r in out['cases']),
    'max_peak_excess_above_ordinary_backward':max(r['finite_peak_above_ordinary_backward'] for r in out['cases'])}
'''
source=source.replace("(A/'vendor_fa_development16_summary_20260907.json')",extra+"\n(A/'vendor_fa_development16_summary_20260907.json')")
(A/'verify_vendor_fa_development16_20260907.py').write_text(source,encoding='utf-8')
print('Prepared full16 independent numerical/original-metric/cost review.')
