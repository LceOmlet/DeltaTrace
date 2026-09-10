"""Verify archived bytes and inventory every saved attempt without discarding failures."""
from pathlib import Path,PurePosixPath
from collections import Counter
import argparse,csv,hashlib,json,zipfile
HERE=Path(__file__).resolve().parent;sha=lambda b:hashlib.sha256(b).hexdigest()

def build(extract=None):
    archives=[];members={};origins={}
    for receipt_path in sorted(HERE.glob('*_archive_verification.json')):
        receipt=json.loads(receipt_path.read_bytes());path=HERE/receipt['archive']
        assert sha(path.read_bytes())==receipt['sha256'] and path.stat().st_size==receipt['bytes']
        with zipfile.ZipFile(path) as z:
            manifest=json.loads(z.read('archive_manifest.json'));assert len(manifest['files'])==receipt['files_verified']
            assert set(z.namelist())==set(manifest['files'])|{'archive_manifest.json'}
            for n,r in manifest['files'].items():
                pure=PurePosixPath(n);assert not pure.is_absolute() and '..' not in pure.parts
                raw=z.read(n);assert len(raw)==r['bytes'] and sha(raw)==r['sha256'],n
                if n in members:assert raw==members[n],n
                else:members[n]=raw
                origins.setdefault(n,[]).append(receipt['archive'])
                if extract:
                    target=extract.joinpath(*pure.parts);target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():assert target.read_bytes()==raw,n
                    else:target.write_bytes(raw)
        archives.append(receipt)
    reports=[];timed=[];operators=[]
    for n,raw in sorted(members.items()):
        if not n.endswith('.json'):continue
        d=json.loads(raw)
        if not isinstance(d,dict) or 'status' not in d:continue
        if not any(k in d for k in ['rows','cases','jobs','files','error','operator_calls','compile_returncode']):continue
        r={'member':n,'sha256':sha(raw),'archives':origins[n],'status':d['status'],
           'method':d.get('method'),'error':d.get('error'),'cases':len(d.get('cases',[])),
           'rows':len(d.get('rows',[])),'jobs':len(d.get('jobs',[]))}
        for k in ['model_calls','attribution_calls','full_attribution_calls','operator_calls','generation_calls','metric_calls',
                  'elapsed_seconds','wall_seconds','model_load_seconds','initialization','native_model_forwards','native_model_backwards']:
            if k in d:r[k]=d[k]
        reports.append(r)
        for index,row in enumerate(d.get('rows',[])):
            if isinstance(row,dict) and 'time_sec' in row:
                timed.append(dict(row,source_member=n,source_sha256=sha(raw),source_row_index=index))
        if 'operator_calls' in d:
            operators.append({'member':n,'calls':d['operator_calls'],'status':d['status']})
    fields=sorted(set().union(*(r.keys() for r in timed)))
    with (HERE/'all_study_timing_rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in timed:w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    result={'status':'verified','archive_count':len(archives),'archives':archives,'archive_total_bytes':sum(r['bytes'] for r in archives),
        'unique_archived_files':len(members),'reports':reports,'report_count':len(reports),
        'timing_rows_count':len(timed),'timing_phase_counts':dict(Counter(str(r.get('phase')) for r in timed)),
        'timing_status_counts':dict(Counter(str(r.get('status')) for r in timed)),
        'timed_rows_seconds':sum(r['time_sec'] for r in timed if isinstance(r.get('time_sec'),(int,float))),
        'operator_only_reports':operators,'operator_only_calls':sum(r['calls'] for r in operators),
        'excluded_archived_reports':0,'excluded_timing_rows':0,'model_calls_for_this_verifier':0,
        'scope':'Inventory of all collected snapshots and flat author-timer rows, including warm, cold, measured and failed rows. Audit, diagnostic, build, operator-only and model-load costs remain in the linked reports. Timed-row sum is not total research compute or the count of all native graph executions. Duplicate identical archive members counted once. Missing or never-run jobs are not inferred.'}
    (HERE/'study_inventory.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',newline='\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--extract',type=Path);a=p.parse_args();d=build(a.extract)
    print(json.dumps({k:d[k] for k in ['status','archive_count','unique_archived_files','report_count','timing_rows_count','timing_phase_counts','timing_status_counts','timed_rows_seconds','operator_only_calls']}))
