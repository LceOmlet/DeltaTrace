source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import os,json,subprocess,time,hashlib
from pathlib import Path
r=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');out=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/appworld-memory-stable-composition-20261009-v1');source_path=r/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json';source=json.loads(source_path.read_bytes());memory=r/'candidates/direct-target-consumed-cache-release-20261007-v1';qwen=json.loads((memory/'environment.json').read_bytes())['qwen35']
env=dict(os.environ,**source['environment']);env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env.update(DT_TASK='AppWorld',DT_MAX_STEPS=str(source['startup_options']['env.max_steps']),DT_ROOT=str(memory/'deltatrace'),DT_ENVIRONMENT_JSON=str(memory/'environment.json'),CUDA_VISIBLE_DEVICES='-1')
env['PYTHONPATH']=':'.join([str(memory/'deltatrace'),env.get('DT_OFFICIAL_ROOT') or qwen['official_root'],str(memory/'deltatrace/clean/qwen35'),source['pythonpath'],qwen['ft_extension_root']])
result=subprocess.run([env['VENV_PYTHON'],str(out/'probe_appworld_memory_stable_imports.py')],env=env,capture_output=True,timeout=40)
(out/'probe.stdout').write_bytes(result.stdout);(out/'probe.stderr').write_bytes(result.stderr)
assert result.returncode==0,(result.stdout.decode(errors='replace'),result.stderr.decode(errors='replace'))
record=dict(unix=time.time(),status='prepared_appworld_composed_memory_and_stable_numeric_CPU_imports_verified',source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),result=json.loads(result.stdout),formal_deployment=False,default_entry_changed=False,diagnostic_GPU_process_started=False,original_training_configuration=dict(lora_rank=source['lora_rank'],lora_alpha=source['lora_alpha'],actor_per_card_microbatch=source['startup_options'].get('actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu'),max_steps=source['startup_options']['env.max_steps']),scope='Existing memory-owner root and accepted numerical leaf only; no source/math/config implementation changes')
(out/'prepared-imports.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
PY
