"""Original Hydra/owner CPU checks of the isolated trial configuration."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/appworld-sql-native-speed-20261016-v1'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
index=read(AUDIT/'prepared-index.json');results=[]
failed=ROOT/'candidates/appworld-sql-native-speed-20261016-v1/AppWorld/CPU-native-config.stderr.log'
if failed.exists() and 'ModuleNotFoundError' in failed.read_text():
    saved=AUDIT/'failed-CPU-probe-missing-existing-extras-r1.stderr.log'
    if not saved.exists():shutil.copyfile(failed,saved)
for item in index['tasks']:
    prepared=Path(item['path']);record=read(prepared)
    env=dict(record['effective_environment']);env['CUDA_VISIBLE_DEVICES']=''
    env.pop('MACA_VISIBLE_DEVICES',None)
    check=subprocess.run(record['argv']+['--cfg','job','--resolve'],
        cwd=env['VERL_ROOT'],env=env,capture_output=True)
    (prepared.parent/'effective-config.yaml').write_bytes(check.stdout)
    (prepared.parent/'CPU-config.stderr.log').write_bytes(check.stderr)
    check.check_returncode()
    from omegaconf import OmegaConf
    cfg=OmegaConf.create(check.stdout.decode())
    assert cfg.trainer.n_gpus_per_node==2
    assert cfg.actor_rollout_ref.model.lora_rank==8 and cfg.actor_rollout_ref.model.lora_alpha==16
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu==4
    assert cfg.actor_rollout_ref.rollout.max_model_len==32768
    assert cfg.actor_rollout_ref.rollout.enforce_eager is False
    assert cfg.trainer.total_training_steps==2 and cfg.trainer.save_freq==-1
    assert cfg.algorithm.adv_estimator=='deltatrace'
    assert all(sha(path)==expected for path,expected in record['source_bindings'].items())
    if record['task']=='AppWorld':
        source='''import sys,json,os,site
site.addsitedir(os.environ['LOOP_EXTRAS'])
sys.path.insert(0,sys.argv[1])
from loop_owner_recipe import compose
from pathlib import Path
c=compose(Path(os.environ['LOOP_ROOT']),[])
assert c.rl.params.scenarios_per_iteration==4
assert c.rl.params.rollouts_per_scenario==6
print(json.dumps(dict(groups=c.rl.params.scenarios_per_iteration,samples=c.rl.params.rollouts_per_scenario,
    max_turns=c.rl.scenario_runner.appworld_config.env.max_interactions,
    max_output=c.llm.vllm_class.max_new_tokens,context=c.llm.vllm_server.max_model_len)))
'''
        native=subprocess.run([env['VENV_PYTHON'],'-c',source,record['entry']],
            env=env,capture_output=True,text=True)
        (prepared.parent/'CPU-native-config.stdout.json').write_text(native.stdout)
        (prepared.parent/'CPU-native-config.stderr.log').write_text(native.stderr)
        native.check_returncode()
    results.append(dict(task=record['task'],status='original_Hydra_CPU_configuration_verified',
        config=str(prepared.parent/'effective-config.yaml'),config_SHA256=sha(prepared.parent/'effective-config.yaml'),
        source_bindings_verified=len(record['source_bindings']),model_DT_optimizer_calls=0))
assert index['formal_authorities_unchanged']=={p:sha(p) for p in index['formal_authorities_unchanged']}
(AUDIT/'CPU-verification.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
