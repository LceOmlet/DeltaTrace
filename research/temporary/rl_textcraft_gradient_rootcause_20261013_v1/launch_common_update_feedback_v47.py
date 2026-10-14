"""Launch prepared v47 only after two selected physical GPUs are actually free."""
import argparse
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--devices',required=True)
args=parser.parse_args()
devices=[int(v) for v in args.devices.split(',')]
assert len(devices)==2 and len(set(devices))==2 and all(0<=v<8 for v in devices)
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('strict_entry',HERE.parent/'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
staged=json.loads((HERE/'common-update-feedback-source-v47.json').read_text(encoding='utf-8'))
script=rf'''source {shlex.quote(entry.ENTRY+'/metax-entry.env.sh')}
"$VENV_PYTHON" - <<'PY'
import hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
root=Path({entry.ROOT!r});out=root/'receipts/textcraft-gradient-rootcause-20261013-v1'
dest=out/'native-common-update-feedback-v47';source=dest/'source'
assert not (dest/'job.json').exists() and not (dest/'result.json').exists(), 'No duplicate or overwrite'
manifest=dest/'case-manifest.json';spec=json.loads(manifest.read_bytes())
assert spec['status']=='prepared_only_no_GPU_submission'
for name,sha in {staged['source_SHA256']!r}.items():
 assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha,name
for path,sha in spec['source_bindings'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
physical=subprocess.check_output(['mx-smi'],text=True,timeout=30)
(dest/'before-launch-mx-smi.txt').write_text(physical)
occupied={{int(m[1]) for m in re.finditer(r'\|\s*(\d+)\s+\d+\s+',physical.split('| Process:')[-1])}}
assert not occupied.intersection({set(devices)!r}), 'Selected GPUs still occupied; no stop, takeover or fallback'
protected=[]
for pid,birth in [(1419849,1791977138.31),(3174448,1791793629.73)]:
 p=psutil.Process(pid);assert p.create_time()==birth and p.status()!=psutil.STATUS_ZOMBIE
 protected.append(dict(pid=pid,birth=birth))
launch_path=Path(spec['original_launch']);launch=json.loads(launch_path.read_bytes())
env=dict(os.environ);env.update(launch['effective_environment']);env['CUDA_VISIBLE_DEVICES']={args.devices!r}
for key in ('RAY_ADDRESS','RAY_TMPDIR','VERL_GLOBAL_TOKEN_MEAN'):env.pop(key,None)
env['PYTHONPATH']=str(source)+':'+str(out)+':'+env['PYTHONPATH']
env['DT_NATIVE_DIAGNOSTIC_MANIFEST']=str(manifest)
env['DT_FSDP_EVENT_WAIT_RECEIPTS']=str(dest/'original-eventwait')
with (dest/'stdout.log').open('wb') as stdout,(dest/'stderr.log').open('wb') as stderr:
 p=subprocess.Popen([env['VENV_PYTHON'],'-u',str(source/'observe_common_update_feedback_v47.py')],
                    env=env,cwd=env['VERL_ROOT'],stdout=stdout,stderr=stderr,start_new_session=True)
record=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),started_unix=time.time(),physical_gpus={devices!r},
            diagnostic_commit={commit!r},status='submitted_native_diagnostic_initialization_pending',
            manifest=str(manifest),manifest_SHA256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            source_SHA256={staged['source_SHA256']!r},source_bindings=spec['source_bindings'],
            native_optimizer_steps_planned=8,native_optimizer_steps_executed='read result.json',
            formal_training_changed=False,old_formal_DTPO_resumed=False,GRPO_reruns=0,
            protected_formal_jobs=protected,accepted_repair=None,deployed_repair=None,
            scope=spec['scope'],question=spec['question'],limits=spec['limits'])
(dest/'job.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
PY
'''
r=subprocess.run(entry.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=90)
(HERE/'launch-common-update-feedback-v47.stdout.txt').write_bytes(r.stdout)
(HERE/'launch-common-update-feedback-v47.stderr.txt').write_bytes(r.stderr)
if r.returncode:
    print(r.stderr.decode(errors='replace'));raise SystemExit(r.returncode)
record=json.loads(r.stdout)
(HERE/'common-update-feedback-v47-job.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:record[k] for k in ('pid','birth','physical_gpus','status','diagnostic_commit')}))
