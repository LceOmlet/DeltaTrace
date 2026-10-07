"""Stage and launch one fixed saved endpoint diagnostic; never a formal job."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('existing_transport', AUDIT/'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
REMOTE = transport.ROOT+'/receipts/direct-target-extreme-token-endpoint-20261007-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['textcraft', 'appworld'], required=True)
    args = parser.parse_args()
    candidate = HERE/'inspect_extreme_endpoint.py'
    preparation = json.loads((HERE/'preparation.json').read_bytes())
    assert sha(candidate) == preparation['candidate']['sha256']
    files = [candidate]
    steps = AUDIT/'direct-target-prefix-runtime-20261007/v1/textcraft-taskrunner-resolved-training-steps.json'
    if args.case == 'textcraft':
        files.append(steps)
    setup = subprocess.run(transport.SSH+['bash', '-s'], input=(
        'set -eu\nmkdir -p '+REMOTE+'\n').encode(), capture_output=True)
    setup.check_returncode()
    for path in files:
        transfer = subprocess.run(transport.SCP+[str(path), transport.SSH[-1]+':'+REMOTE+'/'+path.name],
                                  capture_output=True)
        transfer.check_returncode()
    # The existing environment owns MACA/runtime/cache configuration.  Only the
    # diagnostic's device assignment and source import path are selected here.
    launch = '''set -eu
source {environment}
"$VENV_PYTHON" - <<'PY'
import hashlib,json,os,pathlib,subprocess,time
import psutil
P=pathlib.Path
root=P({root!r});out=P({remote!r});case={case!r}
source_path=root/('runs/direct-target-prefix-runtime-20261007-v1/'+case+'/'+case+'-dt/source.json')
source=json.loads(source_path.read_bytes())
script=out/'inspect_extreme_endpoint.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()=={script_sha!r}
environment_source=source['environment_source'] if case=='textcraft' else source['candidate_environment']
assert hashlib.sha256(P(environment_source['path']).read_bytes()).hexdigest()==environment_source['sha256']
resultdir=out/case
assert not resultdir.exists(),'Do not overwrite a diagnostic or launch it twice'
resultdir.mkdir()
env=dict(os.environ,**source['environment'])
env['DT_TASK']=source['startup_options']['env.env_name']
env['DT_MAX_STEPS']=str(source['startup_options']['env.max_steps'])
env['CUDA_VISIBLE_DEVICES']='4,5'
env.pop('RAY_ADDRESS',None)
env.pop('MACA_VISIBLE_DEVICES',None)
import re
physical=subprocess.run(['mx-smi'],capture_output=True,check=True)
assert not re.search(r'^\|\s*[45]\s+\d+\s+\S',physical.stdout.decode(errors='replace'),re.M),'Selected cards still have a process'
(resultdir/'before-physical.txt').write_bytes(physical.stdout)
dt=P(source['dt_root']);qwen=json.loads(P(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
# Same import roots installed by the original producer, needed for the early
# owner SHA check before constructing any actor or producer instance.
env['PYTHONPATH']=':'.join([str(dt),env.get('DT_OFFICIAL_ROOT') or qwen['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],qwen['ft_extension_root']])
argv=[env['VENV_PYTHON'],str(script),'--case',case,'--source',str(source_path),'--output',str(resultdir)]
if case=='textcraft':
 evidence=out/'textcraft-taskrunner-resolved-training-steps.json'
 assert hashlib.sha256(evidence.read_bytes()).hexdigest()=={steps_sha!r}
 argv+=['--owner-total-training-steps','330','--owner-total-steps-evidence',str(evidence)]
log=resultdir/'driver.log'
with log.open('xb') as stream:
 process=subprocess.Popen(argv,cwd=str(out),env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(case=case,pid=process.pid,process_created_unix=psutil.Process(process.pid).create_time(),launched_unix=time.time(),argv=argv,log=str(log),source_path=str(source_path),source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),script_sha256={script_sha!r},devices=[4,5],diagnostic_only=True,optimizer_steps_requested=0,formal_job_restarted=False)
(resultdir/'launch.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
PY
'''.format(environment=transport.ENTRY+'/metax-entry.env.sh', root=transport.ROOT,
           remote=REMOTE, case=args.case, script_sha=sha(candidate), steps_sha=sha(steps))
    (HERE/(args.case+'-launch-command.sh')).write_text(launch, encoding='utf-8', newline='\n')
    result = subprocess.run(transport.SSH+['bash', '-s'], input=launch.encode(), capture_output=True)
    (HERE/(args.case+'-launch.stdout.txt')).write_bytes(result.stdout)
    (HERE/(args.case+'-launch.stderr.txt')).write_bytes(result.stderr)
    result.check_returncode()
    record = json.loads(result.stdout)
    (HERE/(args.case+'-launch.json')).write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
