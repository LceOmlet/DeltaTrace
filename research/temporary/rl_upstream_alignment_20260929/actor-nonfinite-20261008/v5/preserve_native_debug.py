"""Preserve completed incident artifacts locally with their original SHA256.

Read-only with respect to the job; snapshots append in its diagnostic folder.
Only sidecar-published tensor files are copied, never an in-progress torch.save.
This is an incident archive, not a model checkpoint restore or formal backup.
"""
import hashlib
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
OUT = transport.ROOT+'/receipts/textcraft-native-first-iteration-nonfinite-20261008-v5'

CODE = r'''
import hashlib,json,psutil,time
from pathlib import Path
out=Path(OUT);launch=json.loads((out/'launch.json').read_bytes())
stamp=str(int(time.time())); folder=out/'debug-snapshots'/stamp;folder.mkdir(parents=True,exist_ok=False)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
 return h.hexdigest()
files=[]
for path in out.iterdir():
 if not path.is_file() or path.suffix not in ('.py','.json','.yaml','.log','.txt','.stdout','.stderr'):continue
 if path.name=='launch.json':continue # Retain existing launch receipt; no env secret replication.
 raw=path.read_bytes();target=folder/path.name;target.write_bytes(raw)
 files.append(dict(path=str(target),name=path.name,sha256=sha(target),bytes=len(raw),kind='metadata_snapshot'))
for rank in (0,1):
 initial=out/f'rank{rank}-initial.json'
 input_=out/f'rank{rank}-input.json'
 rows=[]
 for sidecar in (initial,input_):
  if sidecar.exists():rows.append(json.loads(sidecar.read_bytes()))
 report=out/f'rank{rank}.json'
 if report.exists():
  for step in json.loads(report.read_bytes()).get('optimizer_steps',[]):
   rows.append(step)
   if step.get('post_step_path'):rows.append(dict(path=step['post_step_path'],sha256=step['post_step_sha256']))
 for row in rows:
  path=Path(row['path']);digest=sha(path)
  assert digest==row['sha256'],str(path)
  files.append(dict(path=str(path),name=path.name,sha256=digest,bytes=path.stat().st_size,kind='published_tensor'))
original=Path(launch['source_path']);assert sha(original)==launch['source_sha256']
target=folder/'original-source.json';target.write_bytes(original.read_bytes())
files.append(dict(path=str(target),name=target.name,sha256=sha(target),bytes=target.stat().st_size,kind='original_source'))
alive=False
try:alive=psutil.Process(launch['pid']).create_time()==launch['birth']
except psutil.NoSuchProcess:pass
value=dict(observed_unix=time.time(),remote_snapshot=str(folder),driver_pid=launch['pid'],driver_birth=launch['birth'],driver_alive=alive,
 diagnostic_worker_sha256=launch['worker_sha256'],base_commit=launch['base_commit'],
 files=files,total_bytes=sum(f['bytes'] for f in files),
 initial_state_complete=all((out/f'rank{r}-initial.json').exists() for r in (0,1)),
 actual_update_input_complete=all((out/f'rank{r}-input.json').exists() for r in (0,1)),
 update_records_complete=all((out/f'rank{r}.json').exists() and json.loads((out/f'rank{r}.json').read_bytes()).get('phase')=='complete' for r in (0,1)),
 original_NaN_repaired=False,formal_training_started=False,checkpoint_restore=False,
 note='Initial state is captured before the first update via native generic worker RPC during original rollout. Update artifacts not yet published remain missing, never reconstructed.')
(folder/'manifest.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
'''


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):
            h.update(part)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-snapshot', type=int)
    parser.add_argument('--compressed', action='store_true',
                        help='Use stdlib lossless tar/gzip for large tensor files after SCP timeout')
    args = parser.parse_args()
    code = CODE
    if args.resume_snapshot is not None:
        code = ('import json\nfrom pathlib import Path\n'
                +'matches=[(p,json.loads(p.read_bytes())) for p in (Path(OUT)/"debug-snapshots").glob("*/manifest.json")]\n'
                +'matches=[(p,v) for p,v in matches if int(v["observed_unix"])==' +repr(args.resume_snapshot)+' ]\n'
                +'assert len(matches)==1, "Snapshot identity is not unique"\np,v=matches[0]\n'
                +'v["remote_snapshot"]=str(p.parent)\nprint(json.dumps(v))\n')
    script = ('source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'
              +'OUT='+repr(OUT)+'\n'+code+'\nPY\n')
    result = subprocess.run(transport.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=90)
    if result.returncode:
        print(result.stderr.decode(errors='replace'))
    result.check_returncode()
    value = json.loads(result.stdout)
    folder = HERE/'preserved-debug'/str(int(value['observed_unix']))
    folder.mkdir(parents=True,exist_ok=args.resume_snapshot is not None)
    assert shutil.disk_usage(folder).free > value['total_bytes']+1024**3
    # Persist the inventory before any transfer, so a failed transport can
    # resume this immutable snapshot rather than recapture or rerun the job.
    (folder/'manifest-pending.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    if args.compressed:
        remote_code = r'''
import hashlib,json,os,tarfile
from pathlib import Path
folder=Path(SNAPSHOT)
value=json.loads((folder/'manifest.json').read_bytes())
archive=folder/'incident.tar.gz'; receipt=folder/'archive.json'
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
 return h.hexdigest()
if not receipt.exists():
 temporary=folder/'incident.tar.gz.partial'
 with tarfile.open(temporary,'w:gz',compresslevel=1) as tar:
  for item in value['files']:
   source=Path(item['path'])
   assert digest(source)==item['sha256'],str(source)
   tar.add(source,arcname=item['name'],recursive=False)
 os.replace(temporary,archive)
 result=dict(path=str(archive),sha256=digest(archive),bytes=archive.stat().st_size,
             files=len(value['files']),compression='stdlib tarfile gzip level1; lossless')
 receipt.write_text(json.dumps(result,indent=2)+'\n')
print(receipt.read_text())
'''
        command=('source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'
                 +'SNAPSHOT='+repr(value['remote_snapshot'])+'\n'+remote_code+'\nPY\n')
        packed=subprocess.run(transport.SSH+['bash','-s'],input=command.encode(),
                              capture_output=True,check=True,timeout=900)
        binding=json.loads(packed.stdout)
        target=folder/'incident.tar.gz'
        subprocess.run(transport.SCP+[transport.SSH[-1]+':'+binding['path'],str(target)],
                       capture_output=True,check=True,timeout=1800)
        assert sha(target)==binding['sha256']
        expected={item['name'] for item in value['files']}
        with tarfile.open(target,'r:gz') as tar:
            assert {member.name for member in tar.getmembers()}==expected
            tar.extractall(folder,filter='data')
        value['lossless_archive']=dict(binding,local_path=str(target),local_verified=True)
    for entry in value['files']:
        target = folder/entry['name']
        assert target.parent == folder
        if not target.exists() or sha(target)!=entry['sha256']:
            subprocess.run(transport.SCP+[transport.SSH[-1]+':'+entry['path'],str(target)],
                           capture_output=True,check=True,timeout=1800)
        assert sha(target)==entry['sha256'],str(target)
        entry['local_path'] = str(target)
        entry['local_verified'] = True
    value['local_source'] = dict(path=str(Path(__file__)),sha256=sha(Path(__file__)))
    value['local_complete'] = True
    (folder/'manifest.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    (HERE/'preserved-debug-latest.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(path=str(folder/'manifest.json'),bytes=value['total_bytes'],
        files=len(value['files']),initial_state_complete=value['initial_state_complete'],
        actual_update_input_complete=value['actual_update_input_complete'],
        update_records_complete=value['update_records_complete'],local_verified=True)))


if __name__ == '__main__':
    main()
