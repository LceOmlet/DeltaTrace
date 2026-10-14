"""Copy selected native jobs to a local disk using rclone's official SFTP backend.

Only orchestration and source selection live here. Rclone owns incremental copy,
previous-file retention and checksum verification. Training is never imported.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import time

from experiments.rl.backup_metax_to_restic import recorded_metadata_sources, source_ssh_owner


PREPARE = r'''
import hashlib,json,os,pathlib,shutil,sys,time,psutil
request=json.loads(pathlib.Path(sys.argv[1]).read_bytes())
root=pathlib.Path(request['root']); outputs=request['outputs']
recorded=request['recorded']; stamp=request['stamp']
manifest=json.loads((root/'formal-training.json').read_bytes())
jobs=[j for j in manifest['jobs'] if j.get('output',j.get('run_dir')) in outputs]
assert {j.get('output',j.get('run_dir')) for j in jobs}==set(outputs)
stage=root/'backup-access'/'local-snapshots'/stamp
stage.mkdir(parents=True,exist_ok=False)
files={}; checkpoints={}; missing=[]; writing=set()
for job in jobs:
 try:
  proc=psutil.Process(job['pid'])
  if proc.create_time()!=job.get('observed_process_created_unix',job.get('birth')): continue
  for process in [proc,*proc.children(recursive=True)]:
   for handle in process.open_files():
    if any(mode in handle.mode for mode in ('w','a','+')): writing.add(handle.path)
 except psutil.NoSuchProcess: pass
def add(path, immutable=False):
 p=pathlib.Path(path)
 if p.is_file() and not (immutable and str(p.absolute()) in writing): files[str(p.absolute())]=immutable
 elif not p.exists(): missing.append(str(p))
def tree(path):
 p=pathlib.Path(path)
 if p.is_file(): add(p); return
 if not p.is_dir(): missing.append(str(p)); return
 for f in p.rglob('*'):
  if any(x in f.relative_to(p).parts for x in ('checkpoints','__pycache__','.git')): continue
  if f.is_file(): add(f, f.suffix=='.pt')
for name in ('formal-training.json','active-training.json','active-source.json'):
 add(root/name)
for p in recorded: tree(p)
for j in jobs:
 source_path=pathlib.Path(j['source_receipt'])
 source=json.loads(source_path.read_bytes())
 tree(source_path.parent)
 # New native jobs record the entry in source.json and argv in launch.json.
 # Read those authoritative records, preserving the older manifest schema.
 tree(j.get('entry') or source['entry'])
 argv=j.get('argv')
 if argv is None:
  launch_path=source_path.parent/'launch.json'
  launch=json.loads(launch_path.read_bytes())
  assert pathlib.Path(launch['options']['trainer.default_local_dir'])==pathlib.Path(j.get('checkpoint_dir') or j['checkpoints'])
  argv=launch['argv']; add(launch_path)
 for name in ('passive_credit_recording_runtime_override','initial_worker_state_verification'):
  row=j.get(name) or {}
  if isinstance(row,str): row={'path':row}
  receipt={}
  if row.get('path'):
   add(row['path']); receipt=json.loads(pathlib.Path(row['path']).read_bytes())
  for worker in [*row.get('workers',[]),*receipt.get('results',[]),*receipt.get('workers',[])]:
   if isinstance(worker,dict) and worker.get('active_method_source'): add(worker['active_method_source'])
 for arg in argv:
  if arg.startswith(('data.train_files=','data.val_files=')):
   value=json.loads(arg.split('=',1)[1]); add(value)
 ck=pathlib.Path(j.get('checkpoint_dir') or j['checkpoints'])
 marker=ck/'latest_checkpointed_iteration.txt'
 checkpoints[j['method']]=None
 if marker.exists():
  latest=int(marker.read_text()); checkpoints[j['method']]=latest; add(marker)
  for directory in ck.glob('global_step_*'):
   if int(directory.name.removeprefix('global_step_'))<=latest:
    for f in directory.rglob('*'):
     if f.is_file(): add(f,True)
plots=root/'receipts'/'training-plots'
if plots.exists():
 for output in outputs:
  run=pathlib.Path(output).parent.name
  for directory in plots.glob(run+'-*'): tree(directory)
hashes={}; total=0
for name,immutable in sorted(files.items()):
 src=pathlib.Path(name); relative=src.relative_to('/'); dst=stage/'files'/relative
 dst.parent.mkdir(parents=True,exist_ok=True)
 if immutable: os.link(src,dst,follow_symlinks=True)
 else: shutil.copy2(src,dst)
 digest=hashlib.sha256()
 with dst.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
 hashes[str(relative)]=digest.hexdigest(); total+=dst.stat().st_size
result=dict(unix=time.time(),stage=str(stage),source_root=str(root),outputs=outputs,
 files=len(hashes),bytes=total,completed_checkpoints=checkpoints,missing_recorded_paths=missing,
 hashes=hashes)
(stage/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--connector', type=Path, required=True)
    p.add_argument('--source-root', required=True)
    p.add_argument('--job-output', action='append', required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--rclone', type=Path, required=True)
    p.add_argument('--shared-root', help='Additional directory on the recorded remote shared mount')
    args = p.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    stamp = str(time.time_ns())
    receipts = args.destination/'receipts'/stamp
    receipts.mkdir(parents=True)
    report = dict(started=time.time(), status='preparing', destination=str(args.destination),
                  source_root=args.source_root, outputs=args.job_output)
    report_path = receipts/'backup.json'

    def record():
        report_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        (args.destination/'latest-backup.json').write_bytes(report_path.read_bytes())

    record()
    spec = importlib.util.spec_from_file_location('existing_metax_credentials', args.connector)
    provider = importlib.util.module_from_spec(spec); spec.loader.exec_module(provider)
    client = provider.connect(); client.get_transport().set_keepalive(15)
    _, ssh = source_ssh_owner()
    try:
        with client.open_sftp() as sftp:
            with sftp.open(args.source_root+'/formal-training.json') as f:
                training = json.load(f)
        jobs = [j for j in training['jobs'] if j.get('output', j.get('run_dir')) in args.job_output]
        if {j.get('output', j.get('run_dir')) for j in jobs} != set(args.job_output):
            raise ValueError('Requested job is absent from original training manifest')
        paths = recorded_metadata_sources(client, jobs)
        request = args.source_root+'/backup-access/local-selection-'+stamp+'.json'
        with client.open_sftp() as sftp:
            with sftp.open(request, 'wb') as f:
                f.write(json.dumps(dict(root=args.source_root, outputs=args.job_output,
                                        recorded=paths, stamp=stamp)).encode())
        command = ['/opt/conda/bin/python', '-c', PREPARE, request]
        prepared = subprocess.run(ssh+[shlex.join(command)], capture_output=True, check=True, timeout=900)
        selection = json.loads(prepared.stdout)
        (receipts/'selection.json').write_text(json.dumps(selection, indent=2)+'\n', encoding='utf-8')
        checksum = receipts/'sha256.txt'
        checksum.write_text(''.join(f'{digest}  {name}\n' for name, digest in selection['hashes'].items()), encoding='utf-8')
        report.update(status='copying', selection=str(receipts/'selection.json'),
                      files=selection['files'], bytes=selection['bytes'],
                      completed_checkpoints=selection['completed_checkpoints'], remote_stage=selection['stage'])
        record()
        destination = args.destination/'runtime'
        destination.mkdir(exist_ok=True)
        cmd = [str(args.rclone), 'copy', ':sftp:'+selection['stage']+'/files', str(destination),
               '--sftp-ssh', subprocess.list2cmdline(ssh), '--sftp-disable-hashcheck',
               '--backup-dir', str(args.destination/'previous'/stamp),
               '--transfers', '2', '--checkers', '4', '--buffer-size', '8M',
               '--multi-thread-streams', '0', '--stats', '30s', '--log-level', 'INFO',
               '--log-file', str(receipts/'copy.log')]
        report['copy_command'] = cmd; record()
        subprocess.run(cmd, check=True)
        report['status'] = 'verifying'; record()
        subprocess.run([str(args.rclone), 'checksum', 'SHA256', str(checksum), str(destination),
                        '--one-way', '--checkers', '4', '--log-level', 'INFO',
                        '--log-file', str(receipts/'verify.log')], check=True)
        report['local_full_sha256_verified'] = True
        if args.shared_root:
            report['status'] = 'shared_copy'; record()
            # GNU cp owns recursive/update/previous-file behavior. The saved,
            # immutable selection is also the verification input for this copy.
            shared_copy = r'''
import hashlib,json,pathlib,subprocess,sys,time
stage=pathlib.Path(sys.argv[1]); destination=pathlib.Path(sys.argv[2]).resolve()
selection=json.loads((stage/'selection.json').read_bytes())
root=pathlib.Path(selection['source_root']).resolve()
assert not destination.is_relative_to(root)
target=destination/'runtime'; target.mkdir(parents=True,exist_ok=True)
subprocess.run(['cp','-au','--backup=numbered','--reflink=auto',str(stage/'files')+'/.',str(target)+'/'],check=True)
for name,expected in selection['hashes'].items():
 digest=hashlib.sha256()
 with (target/name).open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''): digest.update(chunk)
 if digest.hexdigest()!=expected: raise ValueError('Shared checksum mismatch: '+name)
receipts=destination/'receipts';receipts.mkdir(exist_ok=True)
receipt=dict(unix=time.time(),stage=str(stage),destination=str(destination),files=selection['files'],
 bytes=selection['bytes'],completed_checkpoints=selection['completed_checkpoints'],full_sha256_verified=True)
(receipts/(stage.name+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''
            completed = subprocess.run(ssh+[shlex.join(['/opt/conda/bin/python', '-c', shared_copy,
                                        selection['stage'], args.shared_root])],
                                       capture_output=True, check=True, timeout=900)
            report['shared_backup'] = json.loads(completed.stdout)
        report.update(status='passed', completed=time.time(), full_sha256_verified=True)
        record()
        cleanup = "import pathlib,shutil,sys;root=pathlib.Path(sys.argv[1]).resolve();p=pathlib.Path(sys.argv[2]).resolve();assert p.parent==root/'backup-access'/'local-snapshots';shutil.rmtree(p)"
        subprocess.run(ssh+[shlex.join(['/opt/conda/bin/python', '-c', cleanup,
                                       args.source_root, selection['stage']])], check=True, timeout=60)
        print(json.dumps({k:v for k,v in report.items() if k!='copy_command'}, ensure_ascii=False))
    except Exception as exc:
        report.update(status='failed', error=str(exc), finished=time.time())
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            report['stderr'] = exc.stderr.decode(errors='replace') if isinstance(exc.stderr, bytes) else exc.stderr
        record()
        raise
    finally:
        client.close()


if __name__ == '__main__':
    main()
