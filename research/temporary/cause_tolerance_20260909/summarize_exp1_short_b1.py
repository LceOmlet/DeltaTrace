"""Derive short-input timing tables from retained raw author-exp1 records."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean,median,pstdev
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    selection_path=a.raw/'selection.json'
    selection=json.loads(selection_path.read_bytes()) if selection_path.exists() else {'excluded_attempts':{}}
    reports=[];sources={};identities={};groups=defaultdict(list);audits={};excluded=[];startup=[]
    cost={'successful_timed_call_seconds':0.0,'separate_audit_call_seconds':0.0,
      'model_load_seconds':0.0,'separate_native_initialization_seconds':0.0,'timed_runner_construction_seconds':0.0,
      'successful_timed_calls':0,'failed_timed_calls':0,'separate_audit_calls':0}
    for file in sorted(a.raw.glob('*/results.json')):
        r=json.loads(file.read_bytes())
        if 'rows' not in r:continue
        sources[file.relative_to(a.raw).as_posix()]=sha(file.read_bytes())
        if file.parent.name in selection['excluded_attempts']:
            excluded.append({'path':file.relative_to(a.raw).as_posix(),'sha256':sha(file.read_bytes()),
              'reason':selection['excluded_attempts'][file.parent.name],'status':r['status'],
              'rows':len(r['rows']),'failed_rows':sum(x['status']!='ok' for x in r['rows'])})
            continue
        reports.append({'family':r['family'],'method':r['method'],'status':r['status'],
          'model_load_seconds':r.get('model_load_seconds'),'error':r.get('error'),'rows':len(r['rows']),
          'base_text':r.get('base_text'),'environment':r.get('environment'),'initialization':r.get('initialization')})
        cost['model_load_seconds']+=r.get('model_load_seconds') or 0
        cost['separate_native_initialization_seconds']+=sum(x['seconds'] for x in r.get('initialization',[]))
        startup.append({'family':r['family'],'method':r['method'],
          'model_load_seconds':r.get('model_load_seconds'),'separate_initialization':r.get('initialization'),
          'first_call_seconds':r['rows'][0]['time_sec'] if r['rows'] else None,
          'first_call_status':r['rows'][0]['status'] if r['rows'] else None,
          'cache_scope':'First call in the method process; on-disk native/Inductor caches are shared within each family. Not an independent empty-cache cold-start comparison.'})
        arrays=np.load(file.parent/'vectors.npz') if (file.parent/'vectors.npz').exists() else None
        for case in r['cases']:
            key=(r['family'],case['input_length'])
            identity={k:case[k] for k in ['input_sha256','lengths','generation_length_includes_eos','input_ids']}
            if key in identities:assert identities[key]==identity,key
            identities[key]=identity
            audit=case.get('audit')
            if audit is not None:
                audit=dict(audit)
                cost['separate_audit_calls']+=1;cost['separate_audit_call_seconds']+=audit['separately_charged_seconds']
                vector=arrays[str(case['input_length'])]
                assert sha(vector.tobytes())==audit['vector_sha256']
                audit['raw_nan_count']=int(np.isnan(vector).sum())
                audit['raw_inf_count']=int(np.isinf(vector).sum())
                # Author llm_attr.py:309 explicitly uses NaN for visualization.
                # FT Both leaves omitted stop-word sink rows as NaN; IG and LRP
                # also pad unavailable matrix entries. These are not failures.
                # The public normalizer masks NaN, clamps to >=0, and row-normalizes.
                # Derive only its finiteness property here, without changing timing.
                if r['method']!='deltatrace_retained' and vector.ndim==2:
                    public=np.maximum(np.nan_to_num(vector,nan=0.0),0)
                    public=public/(public.sum(axis=1,keepdims=True)+1e-8)
                    audit['public_view_finite']=bool(np.isfinite(public).all()) and audit['raw_inf_count']==0
                    audit['nan_handling']='Original LLMAttributionResult.normalize_sum_to_one semantics; intentional NaN placeholders retained in raw npz.'
                else:audit['public_view_finite']=audit['finite']
            audits[(r['family'],r['method'],case['input_length'])]=audit
        for row in r['rows']:
            cost['timed_runner_construction_seconds']+=row['runner_init_seconds']
            cost['successful_timed_calls' if row['status']=='ok' else 'failed_timed_calls']+=1
            cost['successful_timed_call_seconds']+=row['time_sec'] or 0
            key=(r['family'],r['method'],row['target_input_tokens'],row['phase'])
            groups[key].append(row)
    cells=[]
    for (family,method,length,phase),rows in sorted(groups.items()):
        ok=[r for r in rows if r['status']=='ok']
        times=[r['time_sec'] for r in ok]
        audit=audits.get((family,method,length))
        cell={'family':family,'method':method,'target_input_tokens':length,'phase':phase,
          'actual_formatted_prompt_tokens':identities[(family,length)]['lengths']['formatted_prompt_tokens'],
          'actual_generation_tokens':identities[(family,length)]['lengths']['generation_tokens'],
          'actual_total_tokens':identities[(family,length)]['lengths']['total_tokens'],
          'n_runs':len(rows),'n_ok':len(ok),'statuses':dict(Counter(r['status'] for r in rows)),
          'seconds_mean':mean(times) if times else None,'seconds_median':median(times) if times else None,
          'seconds_std':pstdev(times) if times else None,'times_sec':times,
          'peak_allocated_gb':max(r['peak_allocated_gb'] for r in ok) if ok else None,
          'peak_reserved_gb':max(r['peak_mem_reserved_gb'] for r in ok) if ok else None,
          'audit_raw_finite':audit['finite'] if audit else None,
          'audit_finite':audit['public_view_finite'] if audit else None,
          'raw_nan_count':audit['raw_nan_count'] if audit else None,
          'raw_inf_count':audit['raw_inf_count'] if audit else None,
          'audit_observed_root_calls':len(audit['actual_calls']) if audit else None,
          'audit_max_observed_sequence_tokens':max((x['shape'][-1] for x in audit['actual_calls']),default=None) if audit else None,
          'valid_timing':len(ok)==3 and (audit is not None and audit['public_view_finite'])}
        cells.append(cell)
    lookup={(x['family'],x['method'],x['target_input_tokens'],x['phase']):x for x in cells}
    paired=[]
    for family in ['qwen3','qwen35']:
        for length in [128,256,512,1024]:
            d=lookup.get((family,'deltatrace_retained',length,'warm3'))
            for method in ['ifr_multi_hop','ifr_multi_hop_both']:
                f=lookup.get((family,method,length,'warm3'))
                if not d or not f or not d['valid_timing'] or not f['valid_timing']:continue
                ratio=d['seconds_mean']/f['seconds_mean']
                paired.append({'family':family,'target_input_tokens':length,'actual_total_tokens':d['actual_total_tokens'],
                  'FT_method':method,'DT_seconds':d['seconds_mean'],'FT_seconds':f['seconds_mean'],
                  'DT_over_FT':ratio,'DT_latency_change_percent':100*(ratio-1),'FT_over_DT_speedup':1/ratio,
                  'DT_faster':ratio<1,'DT_peak_allocated_gb':d['peak_allocated_gb'],'FT_peak_allocated_gb':f['peak_allocated_gb']})
    result={'version':'exp1-short-b1-derived-v1','raw_result_sha256':sources,'runs':reports,'cells':cells,
      'excluded_attempts':excluded,'selection':selection,
      'first_calls_and_loading':startup,
      'recorded_selected_execution_cost':cost,
      'execution_cost_scope':'Selected attempts only. Separate audit duration includes its runner construction. Failed call wall time, garbage collection, imports and cache cleanup are not included in these sums; whole queued process durations are retained separately in raw queue files.',
      'input_identity_matches_across_methods':True,
      'actual_lengths':[{'family':f,'target_input_tokens':n,**v['lengths'],'input_sha256':v['input_sha256']} for (f,n),v in sorted(identities.items())],
      'warm_comparisons':paired,
      'limits':['Synthetic author exp1 input/output; no attribution-quality measurement.',
        'No new optimization; tests the previously retained backend.',
        'Target input sizes are the original builder truncation sizes; retokenized actual sizes differ.',
        'Original3 includes first-call compilation and lazy initialization; warm3 is separate.',
        'Method process isolation and Qwen3.5 adapter are explicit benchmark extensions.',
        'Valid timing requires three successful calls and a finite public output view in the separate audit. Author visualization NaN placeholders are retained in raw matrices and handled by the author public normalizer; infinities are rejected.']}
    (a.output/'exp1_short_b1_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fields=['family','method','target_input_tokens','actual_formatted_prompt_tokens','actual_generation_tokens','actual_total_tokens','phase','n_runs','n_ok','seconds_mean','seconds_std','peak_allocated_gb','peak_reserved_gb','audit_finite','valid_timing','audit_observed_root_calls','audit_max_observed_sequence_tokens']
    with (a.output/'exp1_short_b1_cells.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(cells)
    print(json.dumps({'source_reports':len(reports),'cells':len(cells),'warm_comparisons':paired},indent=2))


if __name__=='__main__':main()
