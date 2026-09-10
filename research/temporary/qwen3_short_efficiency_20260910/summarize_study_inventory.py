"""Inventory every retained timed row, including failed candidates and cold calls."""
import argparse,csv,hashlib,io,json
from pathlib import Path
from collections import Counter


def summarize(raw):
    reports=[];rows=[];cells=[];sha=lambda b:hashlib.sha256(b).hexdigest()
    for path in sorted(raw.rglob('results.json')):
        data=json.loads(path.read_bytes());run=path.parent.relative_to(raw).as_posix()
        timed=data.get('rows',[])
        reports.append({'run':run,'status':data['status'],'raw_sha256':sha(path.read_bytes()),
            'recorded_timing_rows':len(timed),'successful_timing_rows':sum(r['status']=='ok' for r in timed),
            'failed_timing_rows':sum(r['status']!='ok' for r in timed),'recorded_case_count':len(data.get('cases',[])),
            'model_load_seconds':data.get('model_load_seconds'),'error':data.get('error')})
        groups={}
        for row in timed:
            record=dict(row,run=run);rows.append(record)
            key=(row.get('candidate_mode',row.get('attr_func')),row['target_input_tokens'])
            groups.setdefault(key,[]).append(row)
        for (mode,length),values in groups.items():
            warm=[r for r in values if r['phase']=='warm'];measured=[r for r in values if r['phase']=='measured']
            good=[r['time_sec'] for r in measured if r['status']=='ok']
            case=next((c for c in data.get('cases',[]) if c.get('input_length')==length),{})
            cells.append({'run':run,'mode':mode,'requested_input_tokens':length,
                'actual_complete_tokens':values[0]['actual_total_tokens'],
                'warm_seconds':[r['time_sec'] for r in warm],'measured_seconds':[r['time_sec'] for r in measured],
                'successful_measured_mean_seconds':sum(good)/len(good) if good else None,
                'measured_success_count':len(good),'measured_failure_count':len(measured)-len(good),
                'vector_equality_to_same_process_reference':case.get('candidate_vectors_equal',{}).get(mode),
                'math_equality_to_same_process_reference':case.get('math_diagnostics_equal',{}).get(mode),
                'peak_allocated_gb':max((r['peak_allocated_gb'] for r in values if r.get('peak_allocated_gb') is not None),default=None)})
    fields=['run','family','attr_func','candidate_mode','phase','repeat','target_input_tokens','actual_total_tokens','status','time_sec','peak_allocated_gb','peak_mem_reserved_gb','runner_init_seconds','input_sha256','error']
    output=io.StringIO(newline='');writer=csv.DictWriter(output,fieldnames=fields,lineterminator='\n');writer.writeheader()
    writer.writerows({k:r.get(k) for k in fields} for r in rows)
    result={'reports':reports,'cells':cells,'report_count':len(reports),'all_recorded_timing_rows':len(rows),
        'row_phase_counts':dict(Counter(r['phase'] for r in rows)),
        'successful_timing_rows':sum(r['status']=='ok' for r in rows),'failed_timing_rows':sum(r['status']!='ok' for r in rows),
        'successful_recorded_timing_seconds':sum(r['time_sec'] for r in rows if r['status']=='ok'),
        'excluded_timing_rows':0,
        'scope':'Recorded author-timer rows across all retained attempts. Separate audit, profiler, compatibility and graph-build subprogram details remain in their raw reports; missing failure durations are null, never imputed as zero. Pilot means are exploratory and are not the final FT acceptance.'}
    return result,output.getvalue()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result,csv_text=summarize(args.raw)
    (args.output/'study_inventory.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
    (args.output/'all_timing_rows.csv').write_text(csv_text,newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['reports','cells']},indent=2))
