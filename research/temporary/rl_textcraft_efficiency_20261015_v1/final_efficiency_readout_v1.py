"""Read current accepted bindings and resources after bounded head verification."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('strict_entry',HERE.parent/'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry=importlib.util.module_from_spec(s);s.loader.exec_module(entry)
code=r'''
import hashlib,json,shutil,subprocess,time
from pathlib import Path
import psutil
r=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
d=r/'receipts/textcraft-efficiency-20261015-v1'
records={}
for label,name in [('DTPO','textcraft-dt-grpo-multiplier-variance-only-20261014-v3'),('PPO','textcraft-ppo-control-fixed-verl-fresh-20261014-v2')]:
 run=r/'runs'/name
 deploy=json.loads((run/'runtime-overrides/native-frozen-head-efficiency-20261015-v1.json').read_bytes())
 p=psutil.Process(deploy['driver_pid']);assert p.create_time()==deploy['driver_birth']
 env=json.loads((run/'launch.json').read_bytes())['effective_environment']
 owner=Path(env['VERL_ROOT'])
 files={}
 for rel in ['verl/utils/experimental/torch_functional.py','verl/models/transformers/qwen3_vl.py','verl/workers/actor/dp_actor.py','verl/trainer/ppo/core_algos.py']:
  path=owner/rel;files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
 assert files[str(owner/'verl/utils/experimental/torch_functional.py')]=='bb46ebe9cb496a10e13539f8605544c550c892007345a8001bb0ff95f2c425b2'
 assert files[str(owner/'verl/models/transformers/qwen3_vl.py')]=='ebc52fb35812a9b0a5d9251f4e42e6fdbc15a076308390f2da81e47eea56a9d7'
 assert files[str(owner/'verl/workers/actor/dp_actor.py')]=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
 assert files[str(owner/'verl/trainer/ppo/core_algos.py')]=='fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'
 workers=[]
 for row in deploy['workers']:
  p=psutil.Process(row['pid']);assert p.create_time()==row['birth']
  workers.append(dict(pid=p.pid,birth=p.create_time(),phase=p.cmdline()[0],PSS_bytes=p.memory_full_info().pss))
 records[label]=dict(driver_pid=deploy['driver_pid'],driver_birth=deploy['driver_birth'],workers=workers,
  current_bound_file_SHA256=files,head_selection_candidate_deployed=False,
  accepted_deployment_receipt=str(run/'runtime-overrides/native-frozen-head-efficiency-20261015-v1.json'))
dt=Path(json.loads((r/'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3/launch.json').read_bytes())['effective_environment']['DT_ROOT'])
dt_files={str(dt/p):hashlib.sha256((dt/p).read_bytes()).hexdigest() for p in [
 'clean/qwen35/qwen35_dense_finite_runner.py','clean/qwen35/qwen35_decoder_finite.py',
 'clean/qwen35/qwen35_gdn_finite.py','accelerated/qwen35/qwen35_retained_capture.py']}
memory={}
for rel in ['memory.current','memory.max','memory.events','memory.stat','memory/memory.usage_in_bytes','memory/memory.limit_in_bytes','memory/memory.failcnt','memory/memory.oom_control','memory/memory.stat']:
 path=Path('/sys/fs/cgroup')/rel
 if path.is_file():memory[str(path)]=path.read_text()
gpu=subprocess.run(['mx-smi'],capture_output=True,text=True,timeout=20)
v=dict(unix=time.time(),current_jobs=records,DT_actual_file_SHA256=dt_files,memory=memory,
 disk=dict(zip(['total','used','free'],shutil.disk_usage(r))),physical_mx_smi=gpu.stdout,
 completed_head_verification=json.loads((d/'selected-head-worker-verification.json').read_bytes()),
 no_candidate_in_current_launch_path=True)
print(json.dumps(v,ensure_ascii=False))
'''
p=subprocess.run(entry.SSH+['bash','-s'],input=('set -e\nsource '+entry.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n').encode(),capture_output=True)
if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace'))
v=json.loads(p.stdout)
(HERE/'final-efficiency-readout.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v[k] for k in ['unix','current_jobs','DT_actual_file_SHA256','disk','no_candidate_in_current_launch_path']},ensure_ascii=False))
