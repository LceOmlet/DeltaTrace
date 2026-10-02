"""Stage an isolated artifact/API probe; do not alter formal job imports."""
import hashlib
import argparse
from pathlib import Path
import subprocess

from stage_environment_entry import ENTRY, ROOT, REPO, AUDIT, SSH, SCP, remote


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component-diagnostic', action='store_true',
                        help='Observe actual first-layer FLA operands and invoke the pinned owner forward assertions; no production deployment.')
    args = parser.parse_args()
    out = ROOT+'/receipts/owner-b8-dispatch-20260930/' + (
        'native-prefix-components-20261003' if args.component_diagnostic
        else 'native-prefix-artifacts-20261003-v2')
    # Preserve the actual sources of every completed/failed attempt before SCP.
    remote(f'test ! -e {out}/job.json && mkdir -p {out}\n')
    names = ('native_prefix_artifacts_candidate.py', 'verify_native_prefix_artifacts.py')
    if args.component_diagnostic:
        names += ('diagnose_native_prefix_components.py',)
    for name in names:
        subprocess.run(SCP+[str(AUDIT/name), f'{SSH[-1]}:{out}/{name}'], check=True)
    remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import json,os,pathlib,psutil,re,shutil,subprocess,time,hashlib
root=pathlib.Path('@ROOT@');out=pathlib.Path('@OUT@')
assert not (out/'job.json').exists(), 'Reuse the actual recorded probe process; do not duplicate it'
job=next(j for j in json.loads((root/'active-training.json').read_bytes())['jobs'] if j['task']=='AppWorld')
driver=psutil.Process(job['pid'])
physical=subprocess.check_output(['mx-smi'],text=True)
process_section=physical.split('| Process:')[-1]
assert not re.search(r'^\|\s+[67]\s+\d+\s+',process_section,re.M), 'Optional probe GPUs are occupied'
source=root/'candidates/native-host-cache-phase-20261002/e5eb4afc42f1/fsdp_workers.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='e5eb4afc42f10fb4608b3ac43046c906d6a2d21a5387ee02e176bc395f1c6f39'
shutil.copy2(out.parent/'native-prefix-artifacts-20261003/actual-minimum-inputs.json',out/'actual-minimum-inputs.json')
framework=out/'verl-root'
shutil.copytree(pathlib.Path(job['verl_root'])/'verl',framework/'verl')
shutil.copy2(source,framework/'verl/workers/fsdp_workers.py')
launch=json.loads(pathlib.Path(job['output'],'launch.json').read_bytes())
options=launch['options']
(out/'native-launch-options.json').write_text(json.dumps(options,indent=2)+'\n')
env=dict(x.split(b'=',1) for x in pathlib.Path('/proc',str(driver.pid),'environ').read_bytes().split(b'\0') if b'=' in x)
run_env={k.decode():v.decode() for k,v in env.items()}
run_env.pop('MACA_VISIBLE_DEVICES',None);run_env.pop('RAY_ADDRESS',None);run_env.pop('RAY_TMPDIR',None)
run_env.update(CUDA_VISIBLE_DEVICES='6,7',DT_PREFIX_PROBE_ROOT=str(out),VERL_ROOT=str(framework))
if @DIAGNOSTIC@:
 run_env['DT_PREFIX_COMPONENT_DIAGNOSTIC']='1'
run_env['PYTHONPATH']=':'.join([str(out),job['entry'],str(framework),run_env.get('PYTHONPATH','')])
receipt=dict(role='Prepared-only native artifact/API probe, not accepted or deployed training acceleration',
 baseline_dt_release='c9cd147',baseline_dt_reference='fc2e6c2',baseline_verl_upstream='20bd331',
 source_formal_driver_pid=driver.pid,source_formal_driver_created_unix=driver.create_time(),
 source_formal_entry=job['entry'],source_formal_verl_root=job['verl_root'],
 diagnostic_commit='@COMMIT@',stager_sha256='@SHA@',gpus=[6,7],
 source_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
  [source,framework/'verl/workers/actor/dp_actor.py',out/'native_prefix_artifacts_candidate.py',out/'verify_native_prefix_artifacts.py']})
if @DIAGNOSTIC@:
 p=out/'diagnose_native_prefix_components.py';receipt['source_files'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(out/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
with (out/'probe.log').open('wb') as log:
 p=subprocess.Popen([run_env['VENV_PYTHON'],'-u',str(out/'verify_native_prefix_artifacts.py')],
  cwd=out,env=run_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt.update(pid=p.pid,created_unix=psutil.Process(p.pid).create_time(),started_unix=time.time(),log=str(out/'probe.log'))
(out/'job.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
PY
'''.replace('@ROOT@', ROOT).replace('@ENTRY@', ENTRY).replace('@OUT@', out)
           .replace('@COMMIT@', subprocess.check_output(['git', 'rev-parse', 'HEAD'],cwd=REPO,text=True).strip())
           .replace('@DIAGNOSTIC@', str(args.component_diagnostic))
           .replace('@SHA@', hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
