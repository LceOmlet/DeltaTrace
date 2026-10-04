"""Replay saved multi-turn requests with the completed native diagnostic owner.

No training restart, model/task download, optimizer, or formal import change.
Only GPUs 2/3 are used; physical GPUs 0/1 stay free as requested.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tarfile

from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, SSH, SCP, remote


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase-only',action='store_true',
        help='Two DT calls on the largest recorded residual B4, retaining the full factual capture bank.')
    args=parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    parent = ROOT+'/receipts/owner-b8-dispatch-20260930/native-prefix-dt-leases-20261003-v2'
    out = ROOT+'/receipts/owner-b8-dispatch-20260930/native-prefix-reuse-'+(
        'phase' if args.phase_only else 'workload')+'-20261004-'+commit[:7]
    files = {
        REPO/'experiments/rl/native_prefix_leases.py': 'native_prefix_leases.py',
        REPO/'experiments/rl/test_native_prefix_leases.py': 'test_native_prefix_leases.py',
        AUDIT/'diagnose_native_prefix_leases.py': 'diagnose_native_prefix_leases.py',
        Path(__file__): 'run_native_prefix_reuse_workload.py',
    }
    bundle = AUDIT/('native-prefix-reuse-workload-'+commit[:7]+'.tar')
    with tarfile.open(bundle, 'w') as archive:
        for path, name in files.items():
            archive.add(path, arcname=name)
    remote(f'test ! -e {out}/prepared.json && test ! -e {out}/job.json && mkdir -p {out}\n')
    subprocess.run(SCP+[str(bundle), f'{SSH[-1]}:{out}/overlay.tar'], check=True)
    remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import ast,hashlib,json,os,pathlib,psutil,re,shutil,subprocess,sys,time
root=pathlib.Path('@ROOT@');parent=pathlib.Path('@PARENT@');out=pathlib.Path('@OUT@')
previous=json.loads((parent/'prepared.json').read_bytes())
assert (parent/'result.json').exists(), 'Reuse only the completed original comparison'
assert not (out/'prepared.json').exists() and not (out/'job.json').exists()
physical=subprocess.check_output(['mx-smi'],text=True)
assert not re.search(r'^\|\s+[23]\s+\d+\s+',physical.split('| Process:')[-1],re.M), 'GPUs2/3 are occupied'
manifest=json.loads((root/'active-training.json').read_bytes())
text=next(j for j in manifest['jobs'] if j['task']=='TextCraft')
live=psutil.Process(text['pid'])
run_env={k.decode():v.decode() for k,v in (x.split(b'=',1) for x in
 pathlib.Path('/proc',str(live.pid),'environ').read_bytes().split(b'\0') if b'=' in x)}
for key in ('MACA_VISIBLE_DEVICES','RAY_ADDRESS','RAY_TMPDIR'):
 run_env.pop(key,None)
# Frozen original numerical owner, validated diagnostic and literal inputs.
for path in parent.iterdir():
 if path.is_dir() and path.name=='verl-root': shutil.copytree(path,out/path.name)
 elif path.is_file() and (path.suffix=='.py' or path.name=='native-launch-options.json'
                          or path.name.startswith('actual-requests-rank')):
  shutil.copy2(path,out/path.name)
for source,expected in previous['source_files'].items():
 if source.startswith(str(parent)):
  destination=out/pathlib.Path(source).relative_to(parent)
  assert hashlib.sha256(destination.read_bytes()).hexdigest()==expected, destination
subprocess.run(['tar','-xf',str(out/'overlay.tar'),'-C',str(out)],check=True)
options=json.loads((out/'native-launch-options.json').read_bytes())
run_env.update(CUDA_VISIBLE_DEVICES='2,3',DT_PREFIX_PROBE_ROOT=str(out),VERL_ROOT=str(out/'verl-root'),
 DT_PREFIX_DT_LEASE_DIAGNOSTIC='1',DT_PREFIX_DIAGNOSTIC_ROWS='88',
 DT_TASK='AppWorld',DT_MAX_STEPS=str(options['env.max_steps']),DT_MAX_LENGTH='32768',
 DT_SAMPLING_JSON=json.dumps(dict(temperature=options['actor_rollout_ref.rollout.temperature'],
  max_tokens=options['data.max_response_length'])),
 DT_PREFIX_OWNER_SOURCE=str(out/'qwen35_dense_finite_runner_candidate.py'),
 DT_PREFIX_ARTIFACT_SOURCE=str(out/'qwen35_native_prefix_artifacts.py'))
run_env['PYTHONPATH']=':'.join([str(out),previous['source_formal_entry'],str(out/'verl-root'),run_env.get('PYTHONPATH','')])
selected_observation=None
if @PHASE_ONLY@:
 import torch
 completed=root/'receipts/owner-b8-dispatch-20260930/native-prefix-reuse-workload-20261004-712795d'
 maxima=[]
 for rank in (0,1):
  p=completed/f'prefix-lease-vectors-rank{rank}.pt'
  values=torch.load(p,map_location='cpu',weights_only=False)
  original=values['original_warm']['dt_token_advantages']
  delta=(values['shared_warm']['dt_token_advantages']-original).abs()
  index=int(delta.argmax());row,column=divmod(index,delta.shape[1])
  maxima.append(dict(rank=rank,row=row,column=column,residual=float(delta[row,column]),
   vector_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 peak=max(maxima,key=lambda item:item['residual']);offset=peak['row']//4*4
 run_env.update(DT_PREFIX_PHASE_ONLY='1',DT_PREFIX_DIAGNOSTIC_OFFSET=str(offset),DT_PREFIX_DIAGNOSTIC_ROWS='4')
 selected_observation=dict(peak=peak,all_ranks=maxima,offset=offset,
  scope='Same original B4 stream and full 88-row capture bank; stage instrumentation only, not a repeat of the workload benchmark')
receipt=dict(role='Isolated original B4 DT replay; no formal deployment or acceptance of a new numerical core',
 diagnostic_commit='@COMMIT@',stager_sha256='@SHA@',devices=[2,3],rows_per_rank=int(run_env['DT_PREFIX_DIAGNOSTIC_ROWS']),
 selected_observation=selected_observation,
 parent_prepared=dict(path=str(parent/'prepared.json'),sha256=hashlib.sha256((parent/'prepared.json').read_bytes()).hexdigest()),
 live_environment_source=dict(pid=live.pid,pid_birth=live.create_time(),task='TextCraft',
  use='Recorded provisioning/cache/DT flags only; AppWorld task and sampling come from the original frozen launch'),
 baseline_dt_release='c9cd147',baseline_dt_reference='fc2e6c2',baseline_verl_upstream='20bd331',
 source_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.py')},
 literal_request_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('actual-requests-rank*.pt')},
 configuration=dict(rank=8,alpha=16,actor_microbatch=4,dt_minibatch=4,max_length=32768,
  official_launch_sha256=hashlib.sha256((out/'native-launch-options.json').read_bytes()).hexdigest(),
  event_sampling=json.loads(run_env['DT_SAMPLING_JSON']),
  numerical_environment=dict(path=run_env['DT_ENVIRONMENT_JSON'],sha256=hashlib.sha256(pathlib.Path(run_env['DT_ENVIRONMENT_JSON']).read_bytes()).hexdigest())),
 observed_before_start=time.time(),physical_before_start=physical,
 available_host_bytes=psutil.virtual_memory().available)
(out/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
test_env=dict(run_env,CUDA_VISIBLE_DEVICES='')
with (out/'cpu-tests.log').open('wb') as log:
 p=subprocess.run([run_env['VENV_PYTHON'],'-m','pytest',str(out/'test_native_prefix_leases.py'),
  '-k','not moved_artifact','-q','--junitxml='+str(out/'cpu-tests.xml')],env=test_env,cwd=out,
  stdout=log,stderr=subprocess.STDOUT)
receipt['cpu_tests_returncode']=p.returncode
(out/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
assert p.returncode==0, (out/'cpu-tests.log').read_text()
with (out/'probe.log').open('wb') as log:
 p=subprocess.Popen([run_env['VENV_PYTHON'],'-u',str(out/'verify_native_prefix_artifacts.py')],
  cwd=out,env=run_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt.update(pid=p.pid,pid_birth=psutil.Process(p.pid).create_time(),started_unix=time.time(),log=str(out/'probe.log'))
(out/'job.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(out=str(out),pid=p.pid,pid_birth=receipt['pid_birth'],cpu_tests_returncode=0,devices=[2,3])))
PY
'''.replace('@ENTRY@',ENTRY).replace('@ROOT@',ROOT).replace('@PARENT@',parent)
        .replace('@OUT@',out).replace('@COMMIT@',commit).replace('@PHASE_ONLY@',str(args.phase_only))
        .replace('@SHA@',hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
