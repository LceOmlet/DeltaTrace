"""Create a consistent transfer archive of completed cases and audit records."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--indices',type=int,nargs='+')
    a=p.parse_args();assert not a.output.exists()
    root=a.base/'all_baselines_v2';files=[];method_cases=[]
    with zipfile.ZipFile(a.output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        def add(path,name):
            payload=path.read_bytes();archive.writestr(name,payload)
            files.append(dict(path=name,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
        for path in sorted(root.glob('*/*/*/results.json')):
            receipt=json.loads(path.read_bytes())
            if a.indices is not None and receipt['index'] not in a.indices:continue
            assert receipt['status']=='complete'
            vector=path.with_name('vectors.npz');assert hashlib.sha256(vector.read_bytes()).hexdigest()==receipt['vectors_sha256']
            method_cases.append({k:receipt[k] for k in ('dataset','index','method','seconds')})
            for source in (path,vector):add(source,'raw/'+source.relative_to(root).as_posix())
        for path in sorted(root.glob('*/*/*/failed_attempt_*.json')):add(path,'raw/'+path.relative_to(root).as_posix())
        for path in sorted((a.base/'all_baselines_v1').glob('*/*/*/*')):
            if path.is_file():add(path,'initial_pilot/'+path.relative_to(a.base/'all_baselines_v1').as_posix())
        for path in sorted(a.base.glob('all_baselines*.log')):add(path,'audit/'+path.name)
        for folder in ('all_baselines_logs_v1','all_baselines_logs_v2','all_baselines_logs_v3'):
            for path in sorted((a.base/folder).glob('*')):
                if path.is_file():add(path,'audit/'+folder+'/'+path.name)
        add(a.base/'all_baselines_preflight.json','inputs_preflight.json')
        if (a.base/'all_baselines_storage_control.json').exists():
            add(a.base/'all_baselines_storage_control.json','storage_control/verification.json')
        for version in ('all_baselines_cpu_control','all_baselines_cpu_control_v2'):
            for path in sorted((a.base/version).glob('*/*/*/*')):
                if path.is_file():add(path,'storage_control/'+version+'/'+path.relative_to(a.base/version).as_posix())
        manifest=dict(created=time.time(),status='completed_case_snapshot',method_cases=method_cases,files=files)
        archive.writestr('manifest.json',json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(archive=str(a.output),bytes=a.output.stat().st_size,method_cases=len(method_cases),
        sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())),flush=True)

if __name__=='__main__':main()
