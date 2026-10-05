"""Prepare/launch an isolated native-reader observation, never formal training."""
import argparse
import hashlib
import json
import subprocess

from stage_environment_entry import AUDIT, ROOT, SSH, SCP


OUT = ROOT + '/receipts/textcraft-native-readout-20261006-v2'
BASE = ROOT + '/receipts/textcraft-native-minibatch-20261006-v4'
NAME = 'verify_textcraft_native_readout.py'

SCRIPT = r'''/opt/conda/bin/python - <<'PY'
import ast,hashlib,json,pathlib,psutil,re,subprocess,time
out=pathlib.Path('__OUT__'); base=pathlib.Path('__BASE__'); mode='__MODE__'; name='__NAME__'
assert not (out/'job.json').exists(), 'Inspect the existing job; never submit twice'
out.mkdir(exist_ok=True)
source=out/name; assert hashlib.sha256(source.read_bytes()).hexdigest()=='__SHA__'
ast.parse(source.read_bytes())
prior=json.loads((base/'job.json').read_bytes()); prepared=json.loads((base/'prepared-diagnostic.json').read_bytes())
assert (base/'native-minibatch-completed.json').is_file()
roots={}
for basename,kind in [('textcraft_owner_rollout.py','entry'),('dp_actor.py','verl'),('qwen35_dense_finite_runner.py','dt')]:
 paths=[pathlib.Path(p) for p in prepared['sources'] if pathlib.Path(p).name==basename]
 assert len(paths)==1
 p=paths[0]; roots[kind]=p.parent if kind=='entry' else p.parents[3] if kind=='verl' else p.parents[2]
checks={}
for f,h in prepared['sources'].items():
 p=pathlib.Path(f); actual=hashlib.sha256(p.read_bytes()).hexdigest(); assert actual==h,(str(p),actual)
 checks[f]=h
cases=base/'readout-first-response-cases.json'
assert cases.is_file(), 'CPU mapping must prepare and verify the exact original prefixes first'
data=json.loads(cases.read_bytes()); assert [len(rows) for rows in data['rank_cases']]==[32,32]
assert len({row['traj_uid'] for rows in data['rank_cases'] for row in rows})==64
parent=psutil.Process(prior['reused_environment_pid'])
assert abs(parent.create_time()-prior['reused_environment_pid_birth'])<.05
env=parent.environ(); env.pop('RAY_ADDRESS',None)
tail=env['PYTHONPATH']
root=base.parents[1]
tail=tail.replace(str(root/'candidates/appworld-eval-client-routing-resume-20261005-v1/entry'),str(roots['entry']))
tail=tail.replace(str(root/'candidates/appworld-native-prefix-resume-20261005-v2/verl'),str(roots['verl']))
tail=tail.replace(str(root/'candidates/appworld-native-prefix-resume-20261005-v2/deltatrace'),str(roots['dt']))
env.update(PYTHONPATH=str(out)+':'+tail,VERL_ROOT=str(roots['verl']),DT_ROOT=str(roots['dt']),
 DT_TASK='TextCraft',DT_MAX_STEPS='30',DT_MAX_LENGTH='32768',
 DT_TEXTCRAFT_READOUT_ROOT=str(out),DT_TEXTCRAFT_READOUT_BASE=str(base),
 DT_TEXTCRAFT_READOUT_CASES=str(cases),DT_TEXTCRAFT_CHECKPOINT=prior['checkpoint'],CUDA_VISIBLE_DEVICES='4,5')
launch=json.loads((base/'launch.json').read_bytes())
env['DT_SAMPLING_JSON']=json.dumps(dict(temperature=launch['options']['actor_rollout_ref.rollout.temperature'],
 max_tokens=launch['options']['data.max_response_length']))
for key in ['DT_TEXTCRAFT_GRADIENT_DIAGNOSTIC','DT_TEXTCRAFT_MATCHED_DIAGNOSTIC','DT_TEXTCRAFT_PROBE_ROOT',
            'DT_TEXTCRAFT_MINIBATCH_ROOT','DT_TEXTCRAFT_NATIVE_OBSERVATION_DIR']:
 env.pop(key,None)
receipt=dict(role='Isolated 64 real first-response prefixes; original native outcome reader; no rollout/DT/backward/update',
 sources=checks,diagnostic_source=dict(path=str(source),sha256='__SHA__'),
 input=dict(path=str(cases),sha256=hashlib.sha256(cases.read_bytes()).hexdigest(),bytes=cases.stat().st_size),
 checkpoint=prior['checkpoint'],devices=[4,5],reused_environment_pid=parent.pid,
 reused_environment_pid_birth=parent.create_time(),observed_unix=time.time(),optimizer_steps=0,
 native_batch_per_call=4,finite_trace_calls=0,config_source=dict(path=str(base/'launch.json'),
 sha256=hashlib.sha256((base/'launch.json').read_bytes()).hexdigest()),
 runtime_environment={k:env.get(k) for k in ['VERL_ROOT','DT_ROOT','DT_ENVIRONMENT_JSON','CUDA_VISIBLE_DEVICES',
  'TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','PYTHONPATH','DT_SAMPLING_JSON']})
if mode=='prepare':
 with (out/'native-owner-inspection.stdout.txt').open('wb') as log:
  result=subprocess.run([env['VENV_PYTHON'],'-u',str(source),'--inspect-only'],cwd=out,
   env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0, 'Inspect CPU binding failure before any GPU submission'
 inspected=json.loads((out/'native-owner-inspection.json').read_bytes())
 assert not inspected['cuda_initialized']
 assert inspected['native_reader_inputs_sha256']==receipt['input']['sha256']
 prior_imports=json.loads((base/'native-minibatch-source-identity.json').read_bytes())
 assert inspected['worker_constructor']=={key:prior_imports['worker_constructor'][key] for key in ['path','sha256']}, 'Native owner path/hash mismatch'
 receipt['inspection']=dict(path=str(out/'native-owner-inspection.json'),
  sha256=hashlib.sha256((out/'native-owner-inspection.json').read_bytes()).hexdigest())
 (out/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(dict(status='prepared_cpu_owner_inspection',out=str(out),input_sha256=receipt['input']['sha256'])))
else:
 accepted=json.loads((out/'prepared.json').read_bytes())
 for key in ['sources','diagnostic_source','input','config_source','checkpoint']:
  assert receipt[key]==accepted[key], 'Prepared and launch sources differ: '+key
 physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
 (out/'physical-before-start.txt').write_text(physical)
 occupied=[line for line in physical.splitlines() if re.match(r'^\|\s+[45]\s+\d+\s+',line)]
 assert not occupied,occupied
 available=psutil.virtual_memory().available; assert available>150*1024**3
 argv=[env['VENV_PYTHON'],'-u',str(source)]
 with (out/'diagnostic.log').open('wb') as log:
  child=subprocess.Popen(argv,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 receipt.update(pid=child.pid,pid_birth=psutil.Process(child.pid).create_time(),started_unix=time.time(),
  argv=argv,log=str(out/'diagnostic.log'),host_available_before_bytes=available)
 (out/'job.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(dict(status='native_reader_submitted',out=str(out),pid=child.pid,pid_birth=receipt['pid_birth'],devices=[4,5])))
PY
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch'))
    args = parser.parse_args()
    sha = hashlib.sha256((AUDIT / NAME).read_bytes()).hexdigest()
    if args.mode == 'prepare':
        subprocess.run(SSH + ['mkdir', '-p', OUT], check=True)
        subprocess.run(SCP + [str(AUDIT / NAME), f'{SSH[-1]}:{OUT}/{NAME}'], check=True)
    script = (SCRIPT.replace('__OUT__', OUT).replace('__BASE__', BASE).replace('__MODE__', args.mode)
              .replace('__NAME__', NAME).replace('__SHA__', sha))
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
    local = AUDIT / 'textcraft-degradation-20261005/readout-quality-20261006'
    local.mkdir(exist_ok=True)
    (local / f'{args.mode}.stdout.txt').write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
