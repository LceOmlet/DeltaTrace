"""CPU-only boundary tests in the recorded runtime; no GPU or deployment."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from stage_environment_entry import ROOT, ENTRY, REPO, AUDIT, SSH, SCP, remote


if __name__ == '__main__':
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    out = ROOT+'/receipts/owner-b8-dispatch-20260930/native-prefix-leases-cpu-'+commit[:7]
    files = {
        REPO/'deltatrace/clean/qwen35/qwen35_native_prefix_artifacts.py':'qwen35_native_prefix_artifacts.py',
        REPO/'deltatrace/clean/qwen35/qwen35_dense_finite_runner.py':'qwen35_dense_finite_runner_candidate.py',
        **{REPO/'experiments/rl'/name:name for name in (
            'reward_readout.py','deltatrace_credit.py','native_prefix_leases.py',
            'test_native_prefix_provider.py','test_native_prefix_leases.py','test_reward_readout.py')},
    }
    bundle = AUDIT/('native-prefix-lease-interfaces-'+commit[:7]+'.tar')
    with tarfile.open(bundle,'w') as archive:
        for path,name in files.items():archive.add(path,arcname=name)
    remote(f'test ! -e {out}/prepared.json && mkdir -p {out}\n')
    subprocess.run(SCP+[str(bundle),f'{SSH[-1]}:{out}/source.tar'],check=True)
    remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
tar -xf @OUT@/source.tar -C @OUT@
"$VENV_PYTHON" - <<'PY'
import hashlib,json,os,pathlib,psutil,subprocess,time
root=pathlib.Path('@ROOT@');out=pathlib.Path('@OUT@')
job=next(j for j in json.loads((root/'active-training.json').read_bytes())['jobs'] if j['task']=='AppWorld')
driver=psutil.Process(job['pid'])
record=dict(role='CPU interface contracts only, no GPU/model execution or production deployment',
 diagnostic_commit='@COMMIT@',stager_sha256='@SHA@',created_unix=time.time(),
 formal_driver=driver.pid,formal_driver_birth=driver.create_time(),
 source_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.py')},
 official_verl=job['verl_root'],original_entry=job['entry'])
(out/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
env=dict(os.environ)
env.pop('MACA_VISIBLE_DEVICES',None)
env.update(CUDA_VISIBLE_DEVICES='',DT_PREFIX_OWNER_SOURCE=str(out/'qwen35_dense_finite_runner_candidate.py'),
 DT_PREFIX_ARTIFACT_SOURCE=str(out/'qwen35_native_prefix_artifacts.py'),
 PYTHONPATH=':'.join([str(out),job['entry'],job['verl_root'],env.get('PYTHONPATH','')]))
argv=[env['VENV_PYTHON'],'-m','pytest',str(out/'test_native_prefix_provider.py'),
 str(out/'test_native_prefix_leases.py'),str(out/'test_reward_readout.py'),
 '-k','not default_body and not moved_artifact','-q','--junitxml='+str(out/'result.xml')]
with (out/'tests.log').open('wb') as log:
 p=subprocess.Popen(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 record.update(pid=p.pid,pid_birth=psutil.Process(p.pid).create_time(),argv=argv)
 (out/'job.json').write_text(json.dumps(record,indent=2)+'\n')
 code=p.wait()
record.update(finished_unix=time.time(),returncode=code)
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(receipt=str(out),returncode=code)),flush=True)
print((out/'tests.log').read_text())
raise SystemExit(code)
PY
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY).replace('@OUT@',out)
           .replace('@COMMIT@',commit).replace('@SHA@',hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
