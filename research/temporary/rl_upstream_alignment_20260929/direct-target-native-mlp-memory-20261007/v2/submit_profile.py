"""One isolated memory diagnostic, with original source and held-job guard."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('transport', AUDIT/'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
REMOTE = transport.ROOT+'/receipts/direct-target-native-mlp-memory-20261007-v3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--consumed-cache-candidate', action='store_true')
    args = parser.parse_args()
    global REMOTE
    if args.consumed_cache_candidate:
        REMOTE=transport.ROOT+'/receipts/direct-target-native-mlp-memory-20261007-v6'
    files = [HERE/'profile_existing_offload.py', HERE/'prepare_failed_batch.py',
             AUDIT/'direct-target-extreme-token-endpoint-20261007/v1/inspect_extreme_endpoint.py']
    for path in files:
        compile(path.read_bytes(), str(path), 'exec')
    remote = subprocess.run(transport.SSH+['bash', '-s'], input=(
        'set -eu\nmkdir -p '+REMOTE+'\n').encode(), capture_output=True)
    remote.check_returncode()
    for path in files:
        result = subprocess.run(transport.SCP+[str(path), transport.SSH[-1]+':'+REMOTE+'/'+path.name], capture_output=True)
        result.check_returncode()
    command = '''set -eu
source {environment}
"$VENV_PYTHON" - <<'PY'
import hashlib,json,os,pathlib,subprocess,time
import psutil
P=pathlib.Path
root=P({root!r});out=P({remote!r})
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
source=json.loads(source_path.read_bytes())
assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
for name,expected in {hashes!r}.items():
 assert hashlib.sha256((out/name).read_bytes()).hexdigest()==expected
assert hashlib.sha256(P(source['candidate_environment']['path']).read_bytes()).hexdigest()==source['candidate_environment']['sha256']
env=dict(os.environ,**source['environment'])
env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env['DT_TASK']=source['startup_options']['env.env_name']
env['DT_MAX_STEPS']=str(source['startup_options']['env.max_steps'])
if {cache_candidate!r}:
 preparation=json.loads((root/'candidates/direct-target-consumed-cache-release-20261007-v1/preparation.json').read_bytes())
 assert preparation['status']=='prepared_only_unaccepted'
 assert hashlib.sha256(P(preparation['changed_file']).read_bytes()).hexdigest()==preparation['changed_sha256']
 assert hashlib.sha256(P(preparation['candidate_environment']).read_bytes()).hexdigest()==preparation['candidate_environment_sha256']
 env['DT_ROOT']=preparation['candidate_dt_root'];env['DT_ENVIRONMENT_JSON']=preparation['candidate_environment']
dt=P(env['DT_ROOT']);qwen=json.loads(P(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
env['PYTHONPATH']=':'.join([str(out),str(dt),env.get('DT_OFFICIAL_ROOT') or qwen['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],qwen['ft_extension_root']])
saved=root/'receipts/direct-target-prefix-runtime-20261007-v1/appworld-first-dt'
if not {launch!r}:
 env['CUDA_VISIBLE_DEVICES']=''
 argv=[env['VENV_PYTHON'],str(out/'prepare_failed_batch.py'),'--source',str(source_path),'--artifacts',str(saved),'--output',str(out/'failed-inputs')]
 result=subprocess.run(argv,env=env,cwd=str(out),capture_output=True)
 (out/'prepare.stdout.txt').write_bytes(result.stdout);(out/'prepare.stderr.txt').write_bytes(result.stderr)
 print(result.stdout.decode(errors='replace'))
 result.check_returncode()
else:
 assert not (out/'launch.json').exists(),'Never submit this diagnostic twice'
 failed_inputs=(root/'receipts/direct-target-native-mlp-memory-20261007-v3/failed-inputs') if {cache_candidate!r} else out/'failed-inputs'
 assert (failed_inputs/'selection.json').exists()
 text=psutil.Process(2833207)
 assert text.create_time()==1791370325.16
 for rank in (0,1):
  artifacts=root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
  assert not (artifacts/('rank'+str(rank)+'-release-update')).exists()
  assert (artifacts/('rank'+str(rank)+'-pre-update.pt')).exists()
 for pid in (2786671,2792547,2794018,3238620,3463033,3521392,3589964,3645085):
  assert not psutil.pid_exists(pid),('Old process still present',pid)
 physical=subprocess.run(['mx-smi'],capture_output=True,check=True)
 (out/'before-physical.txt').write_bytes(physical.stdout)
 import re
 text_smi=physical.stdout.decode(errors='replace')
 # Require the two selected cards to have no compute process; do not stop
 # unrelated cards or infer availability from historical device indices.
 for line in text_smi.splitlines():
  if re.match(r'^\|\s*[45]\s+\d+\s+\S',line):
   raise AssertionError('Selected card has an existing process: '+line)
 env['CUDA_VISIBLE_DEVICES']='4,5'
 argv=[env['VENV_PYTHON'],str(out/'profile_existing_offload.py'),'--source',str(source_path),'--native',str(saved/'rank1-readout-native-batch-6.pt'),'--output',str(out/'results'),'--failed-batch',str(failed_inputs)]
 if {cache_candidate!r}:argv+=['--with-vllm']
 log=out/'driver.log'
 with log.open('xb') as stream:
  process=subprocess.Popen(argv,env=env,cwd=str(out),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 record=dict(pid=process.pid,process_created_unix=psutil.Process(process.pid).create_time(),launched_unix=time.time(),argv=argv,devices=[4,5],source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),scripts={hashes!r},actual_dt_root=env['DT_ROOT'],actual_environment=env['DT_ENVIRONMENT_JSON'],candidate_preparation=preparation if {cache_candidate!r} else None,formal_restart=False,optimizer_steps_requested=0,text_update_released=False)
 (out/'launch.json').write_text(json.dumps(record,indent=2)+'\\n')
 print(json.dumps(record))
PY
'''.format(environment=transport.ENTRY+'/metax-entry.env.sh', root=transport.ROOT,
           remote=REMOTE, hashes={path.name:sha(path) for path in files}, launch=args.launch,
           cache_candidate=args.consumed_cache_candidate)
    phase = 'launch' if args.launch else 'prepare'
    (HERE/(phase+'-command.sh')).write_text(command, encoding='utf-8', newline='\n')
    result = subprocess.run(transport.SSH+['bash','-s'],input=command.encode(),capture_output=True)
    (HERE/(phase+'.stdout.txt')).write_bytes(result.stdout)
    (HERE/(phase+'.stderr.txt')).write_bytes(result.stderr)
    print(result.stdout.decode(errors='replace'))
    if result.returncode:
        print(result.stderr.decode(errors='replace'))
    result.check_returncode()


if __name__ == '__main__':
    main()
