"""Preserve completed experiment byte artifacts locally, without model calls."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--single-background-task',choices=('textcraft','appworld'))
parser.add_argument('--terminal-failure',action='store_true',help='Preserve a stopped, failed single-background diagnostic, including partial results.')
parser.add_argument('--nonfinite-replay',action='store_true')
parser.add_argument('--range-owner', choices=('replay', 'resume'),
                    help='Preserve the separately isolated numerical range diagnostic.')
parser.add_argument('--revision',choices=('v1','v2','v3'),default='v1')
parser.add_argument('--native-context', choices=('v1','v2','v3','v4','v5','profile','queries','memory-limit','memory-finite','conv-windows'),
                    help='Preserve terminal native context operator/API diagnostics.')
parser.add_argument('--metadata-only',action='store_true',
                    help='Preserve remote hashes/archive identity without claiming a completed local archive copy.')
args=parser.parse_args()
assert not args.terminal_failure or args.single_background_task
assert not args.nonfinite_replay or args.single_background_task
assert not args.range_owner or args.single_background_task == 'appworld'
assert not args.native_context or not (args.single_background_task or args.nonfinite_replay or args.range_owner)
remote=(transport.ROOT+'/receipts/credit-single-background-'+args.single_background_task+'-20261009-v1'
        if args.single_background_task else transport.ROOT+'/receipts/endpoint-head-collection-textcraft-20261009-v1')
folder=(HERE/('single-background-'+args.single_background_task)/'preserved-artifacts'
        if args.single_background_task else HERE/'endpoint-head-owner-v1/preserved-artifacts')
if args.nonfinite_replay:
 remote=transport.ROOT+'/receipts/credit-single-background-nonfinite-'+args.single_background_task+'-20261009-'+args.revision
 folder=HERE/('single-background-nonfinite-'+args.single_background_task+'-'+args.revision)/'preserved-artifacts'
if args.range_owner:
 remote=transport.ROOT+('/receipts/credit-range-owner-replay-appworld-20261009-v1' if args.range_owner=='replay'
                       else '/receipts/credit-single-background-range-resume-appworld-20261009-v1')
 folder=HERE/('range-owner-'+args.range_owner)/'preserved-artifacts'
if args.native_context:
 names={'profile':'/receipts/native-context-profile-20261009-v1',
        'queries':'/receipts/native-conditional-queries-20261009-v1',
        'memory-limit':'/receipts/conditional-memory-limit-20261009-v1',
        'memory-finite':'/receipts/conditional-memory-finite-20261009-v1',
        'conv-windows':'/receipts/conditional-conv-windows-20261009-v1'}
 remote=transport.ROOT+names.get(args.native_context,'/receipts/native-context-readouts-20261009-'+args.native_context)
 folder=HERE/('native-context-'+args.native_context)/'preserved-artifacts'
folder.mkdir(parents=True,exist_ok=True)
body=r'''
import hashlib,json,tarfile,psutil
from pathlib import Path
out=Path(OUT)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if TERMINAL_FAILURE:
 launch=json.loads((out/'launch.json').read_bytes())
 assert not psutil.pid_exists(launch['pid']) or psutil.Process(launch['pid']).create_time()!=launch['birth']
 assert any(json.loads((out/f'results/rank{rank}.json').read_bytes())['phase']=='failed' for rank in (0,1))
else:assert (out/'results/completed.json').exists()
files={}
for rank in (0,1):
 record=json.loads((out/f'results/rank{rank}.json').read_bytes())
 if not TERMINAL_FAILURE:assert record['phase']=='complete' or (NONFINITE_REPLAY and record['phase']=='diagnostic_capture_complete')
 if SINGLE_BACKGROUND:
  for batch in record['batches']:
   for point in batch['points']:
    item=point['artifact'];assert sha(item['path'])==item['sha256']
  for p in out.glob(f'results/rank{rank}-batch*-round*.pt'):files[p.name]=p
 else:
  baseline=json.loads((out/f'textcraft-v3-rank{rank}.json').read_bytes())
  for batch in record['batches']:
   item=batch['variants']['endpoint_head'];assert sha(item['artifact'])==item['sha256']
   files['candidate-'+Path(item['artifact']).name]=Path(item['artifact'])
  for batch in baseline['batches']:
   item=batch['variants']['original'];assert sha(item['artifact'])==item['sha256']
   files['original-'+Path(item['artifact']).name]=Path(item['artifact'])
for name in ('results/rank0.json','results/rank1.json','results/rank0-phases.jsonl','results/rank1-phases.jsonl',
             'results/actor-initialization.json','results/completed.json','comparison-inputs.json',
             'launch.json','driver.log','endpoint_head_seed.py','inspect_conditional_collection.py',
             'inspect_single_background_collection.py','layer-collection-inputs.json','reference-adapter-cpu.json',
             'results/effective-config.yaml','inspect_action_curve.py','inspect_extreme_endpoint.py'):
 p=out/name
 if p.exists():files[name.replace('/','-')]=p
if SINGLE_BACKGROUND:
 source_path=Path(json.loads((out/'launch.json').read_bytes())['source_path'])
 files['original-source.json']=source_path
 for p in out.glob('*.py'):files.setdefault(p.name,p)
 for rank in (0,1):
  record=json.loads((out/f'results/rank{rank}.json').read_bytes())
  for key,item in dict(record.get('owners',{}),finite_runner=record['finite_runner']).items():
   p=Path(item['path']);assert sha(p)==item['sha256']
   files['owner-'+key.replace('.','_')+'.py']=p
remote_only=[]
if NONFINITE_REPLAY:
 for p in out.glob('results/rank*-*.json'):files[p.name]=p
 for p in out.glob('results/rank*-exact-input.pt'):files[p.name]=p
 for p in list(out.glob('results/rank*-first-nonfinite.pt'))+list(out.glob('results/rank*-precast-seed.pt')):
  remote_only.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),
                         scope='Exact large operands retained on original host; not copied into local metadata archive.'))
manifest=dict(files=[dict(name=n,source=str(p),bytes=p.stat().st_size,sha256=sha(p)) for n,p in files.items()],
              remote_only_files=remote_only,
              model_calls=0,DT_calls=0,optimizer=0,status='terminal_failed_partial' if TERMINAL_FAILURE else 'complete',
              scope='Exact attribution matrices and source/provenance logs; incomplete failed results remain partial; no restoration')
path=out/'preserved-artifacts.tar.gz'
if not path.exists():
 with tarfile.open(path,'w:gz',compresslevel=1) as stream:
  for name,p in files.items():stream.add(p,arcname=name)
manifest['archive']=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
print(json.dumps(manifest))
'''
if args.native_context:
 body=r'''
import hashlib,json,tarfile,psutil
from pathlib import Path
out=Path(OUT)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
launch=json.loads((out/'launch.json').read_bytes())
assert not psutil.pid_exists(launch['pid']) or psutil.Process(launch['pid']).create_time()!=launch['birth']
files={p.name:p for p in out.iterdir() if p.is_file() and p.suffix in ('.py','.json','.log','.pt')}
for p in list(files.values()):
 if p.suffix=='.json' and p.name=='result.json':
  result=json.loads(p.read_bytes())
  for case in result.get('cases',[]):
   item=case.get('exact_artifact') or case.get('artifact')
   if item:assert sha(item['path'])==item['sha256']
manifest=dict(files=[dict(name=n,source=str(p),bytes=p.stat().st_size,sha256=sha(p)) for n,p in files.items()],
 remote_only_files=[],status='terminal_native_operator_diagnostic',
 scope='Exact real operator inputs/results, original API layout counterexamples and source/launch/log; not a training estimator.',
 model_calls=0,DT_calls=0,optimizer=0)
path=out/'preserved-artifacts.tar.gz'
if not path.exists():
 with tarfile.open(path,'w:gz',compresslevel=1) as stream:
  for name,p in files.items():stream.add(p,arcname=name)
manifest['archive']=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
print(json.dumps(manifest))
'''
shell='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\nOUT='+repr(remote)+'\nSINGLE_BACKGROUND='+repr(bool(args.single_background_task))+'\nTERMINAL_FAILURE='+repr(args.terminal_failure)+'\nNONFINITE_REPLAY='+repr(args.nonfinite_replay)+'\n'+body+'\nPY\n'
result=subprocess.run(transport.SSH+['bash','-s'],input=shell.encode(),capture_output=True,timeout=60)
(folder/'remote.stderr.txt').write_bytes(result.stderr);result.check_returncode()
manifest=json.loads(result.stdout)
archive=folder/'preserved-artifacts.tar.gz'
manifest.update(local_archive=str(archive),all_file_SHA256_verified=False,
                local_archive_copy_verified=False,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(folder/'manifest.remote.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
if args.metadata_only:
 print(json.dumps(dict(files=len(manifest['files']),remote_archive=manifest['archive'],
                      local_archive_copy_verified=False,metadata=str(folder/'manifest.remote.json'))))
 raise SystemExit(0)
subprocess.run(transport.SCP+[transport.SSH[-1]+':'+manifest['archive']['path'],str(archive)],check=True,timeout=120)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['archive']['sha256']
expected={f['name']:f for f in manifest['files']}
with tarfile.open(archive,'r:gz') as stream:
    assert {m.name for m in stream.getmembers()}==set(expected)
    for member in stream:
        raw=stream.extractfile(member).read();entry=expected[member.name]
        assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256']
manifest.update(local_archive=str(archive),all_file_SHA256_verified=True,local_archive_copy_verified=True,extracted=False,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(files=len(expected),bytes=sum(f['bytes'] for f in manifest['files']),archive_bytes=archive.stat().st_size,
                     local_manifest=str(folder/'manifest.json'),all_file_SHA256_verified=True,model_calls=0)))
