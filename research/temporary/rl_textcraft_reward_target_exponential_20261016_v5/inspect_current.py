"""Read recorded native jobs and actual device occupancy through the strict entry."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location(
    "recorded_entry", REPO / "research/temporary/rl_upstream_alignment_20260929/stage_environment_entry.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)

script = r'''
import hashlib,json,os,psutil,shutil,subprocess,time
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
formal=json.loads((root/'formal-training.json').read_bytes())
jobs=[]
for job in formal['jobs']:
    if job.get('task')!='TextCraft':continue
    item={k:job.get(k) for k in ('task','method','pid','observed_process_created_unix','output','status','version')}
    try:
        p=psutil.Process(job['pid'])
        item['actual_process']=dict(birth=p.create_time(),status=p.status(),name=p.name(),cmdline=p.cmdline())
    except psutil.NoSuchProcess:item['actual_process']=None
    run=Path(job['output']).parent
    launch=json.loads((run/'launch.json').read_bytes())
    item['launch']=dict(path=str(run/'launch.json'),upstream_commit=launch.get('upstream_commit'),
                        local_patch_commit=launch.get('local_patch_commit'),budget=launch.get('budget'))
    item['source_roots']={k:launch['effective_environment'].get(k) for k in ('DT_ROOT','VERL_ROOT','DT_ENTRY_ROOT','DT_CREDIT_MODE')}
    jobs.append(item)
native=json.loads((root/'receipts/textcraft-gradient-rootcause-20261013-v1/native-PPO-control-v9-job.json').read_bytes())
ppo={k:native.get(k) for k in ('pid','birth','run','output','physical_gpus','status')}
try:
    p=psutil.Process(native['pid']);ppo['actual_process']=dict(birth=p.create_time(),status=p.status(),cmdline=p.cmdline())
except psutil.NoSuchProcess:ppo['actual_process']=None
base=json.loads((root/'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3/launch.json').read_bytes())
env=base['effective_environment'];verl=Path(env['VERL_ROOT']);dt=Path(env['DT_ROOT'])
files=[verl/'verl/utils/experimental/torch_functional.py',verl/'verl/models/transformers/qwen3_vl.py',
       verl/'verl/workers/actor/dp_actor.py',verl/'verl/trainer/ppo/core_algos.py',
       dt/'deltatrace/clean/qwen35/qwen35_answer_finite.py',dt/'clean/qwen35/qwen35_answer_finite.py',
       dt/'clean/qwen35/qwen35_dense_finite_runner.py',dt/'clean/qwen35/qwen35_gdn_finite.py',
       dt/'accelerated/qwen35/qwen35_retained_capture.py']
bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
device=shutil.which('mx-smi') or '/opt/maca/bin/mx-smi'
gpu=subprocess.run([device],capture_output=True,text=True,timeout=20)
print(json.dumps(dict(unix=time.time(),jobs=jobs,PPO=ppo,bindings=bindings,
                     GPU=gpu.stdout,GPU_stderr=gpu.stderr,GPU_returncode=gpu.returncode),indent=2))
'''
entry.remote("source " + entry.ENTRY + "/metax-entry.env.sh\n\"$VENV_PYTHON\" - <<'PY'\n" + script + "\nPY\n")
