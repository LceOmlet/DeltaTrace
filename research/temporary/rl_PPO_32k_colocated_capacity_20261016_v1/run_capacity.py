"""Exact32768 B4/rank capacity through the pinned TaskRunner initialization.

Replace only the test driver's fit callback; all workers, model placement,
logprobs, values, GAE, checkpoint restore and updates remain native VERL.
No environment rollout, validation, formal iteration or checkpoint creation.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION = 'PPO-32k-colocated-capacity-20261016-v1'
AUDIT = ROOT/'receipts'/VERSION/'execution3'
CANDIDATE = ROOT/'candidates/PPO-32k-native-efficiency-20261016-v4/verl-owner'
PREPARED = ROOT/'candidates/appworld-native-PPO-32k-speed-20261016-v3/prepared.json'
FIXTURE = ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/same-session-native-data-32768.pt'
CHECKPOINT = ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/performance-exec2/numerical-warm-state-original-VERL'

if '--native' not in sys.argv:
    import re
    smi = subprocess.check_output(['mx-smi'], text=True)
    assert not any(re.match(r'\|\s*(?:6|7)\s+\d+\s+\S+', line) for line in smi.splitlines()), 'Physical6/7 occupied'
    prepared = json.loads(PREPARED.read_bytes())
    env = dict(os.environ); env.update(prepared['effective_environment'])
    env['VERL_ROOT'] = str(CANDIDATE)
    env['PYTHONPATH'] = str(CANDIDATE)+':'+env['PYTHONPATH']
    env['CUDA_VISIBLE_DEVICES'] = '6,7'
    env['FLASH_ATTENTION_DETERMINISTIC'] = '0'
    env['DT_FSDP_EVENT_WAIT_RECEIPTS'] = str(AUDIT/'worker-runtime')
    env.pop('MACA_VISIBLE_DEVICES', None)
    assert env['VENV_PYTHON'] == '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python'
    extra = ['--preflight'] if '--preflight' in sys.argv else []
    raise SystemExit(subprocess.call([env['VENV_PYTHON'], __file__, '--native']+extra, env=env))

import inspect
import psutil
import ray
import torch
from omegaconf import OmegaConf
from unittest.mock import patch
from verl import DataProto
from verl.trainer import main_ppo
from verl.trainer.ppo.ray_trainer import compute_advantage, compute_response_mask, AdvantageEstimator
from verl.utils.fs import copy_to_local

AUDIT.mkdir(parents=True, exist_ok=True)
status = dict(status='initializing_native_TaskRunner', started=time.time(),
              pid=os.getpid(), birth=psutil.Process().create_time(), physical_gpus=[6,7],
              source_commit=os.environ['PPO_32K_SOURCE_COMMIT'], context=32768,
              batch=8, microbatch_per_rank=4, formal_training_mutations=0,
              formal_iterations=0, rollout_calls=0, DT_calls=0, checkpoint_create_calls=0,
              FA_deterministic=False, fixture=str(FIXTURE), checkpoint=str(CHECKPOINT),
              scope=__doc__, stages=[])

def write_status():
    p = AUDIT/'status.json'
    t = p.with_suffix('.tmp')
    t.write_text(json.dumps(status, indent=2, allow_nan=False)+'\n')
    t.replace(p)

def worker_identity(worker):
    import hashlib, os, psutil, sys
    from pathlib import Path
    names = ('verl.workers.fsdp_workers', 'verl.workers.actor.dp_actor',
             'verl.workers.critic.dp_critic', 'verl.trainer.ppo.core_algos',
             'verl.utils.activation_offload', 'verl.utils.experimental.torch_functional')
    roles = {}
    for role, item in worker.worker_dict.items():
        cfg = item.config
        if getattr(item, 'critic', None) is not None:
            roles[role] = dict(kind='critic', microbatch=cfg.ppo_micro_batch_size_per_gpu,
                               activation_offload=cfg.model.enable_activation_offload,
                               native_bound=cfg.model.get('activation_offload_native_max_seq_len',0))
        if getattr(item, 'actor', None) is not None:
            roles[role] = dict(kind='actor_rollout', microbatch=cfg.actor.ppo_micro_batch_size_per_gpu,
                               enforce_eager=cfg.rollout.enforce_eager,
                               free_cache_engine=cfg.rollout.free_cache_engine,
                               max_model_len=cfg.rollout.max_model_len,
                               has_vllm=getattr(item,'rollout',None) is not None)
    return dict(pid=os.getpid(), birth=psutil.Process().create_time(),
                physical_CVD=os.environ.get('CUDA_VISIBLE_DEVICES'),
                FA_deterministic=os.environ.get('FLASH_ATTENTION_DETERMINISTIC'), roles=roles,
                modules={name:dict(path=sys.modules[name].__file__,
                                   SHA256=hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest())
                         for name in names if name in sys.modules})

def identities():
    actors = [v for v in ray.util.list_named_actors(all_namespaces=True)
              if 'WorkerDict' in v['name'] and 'register_center' not in v['name']]
    handles = [ray.get_actor(v['name'], namespace=v['namespace']) for v in actors]
    assert len(handles) == 2, actors
    return ray.get([v.execute_with_func_generator.remote(func=worker_identity) for v in handles])

def capacity_data():
    original = torch.load(FIXTURE, weights_only=False, map_location='cpu')
    # Same IDs and existing policy actions: expose the observed history on the
    # response axis, as the original LOOP batch does (one prompt token).
    # Added history keeps zero policy mask and zero newly assigned reward.
    length = original.batch['input_ids'].shape[1]-1
    extra = length-original.batch['responses'].shape[1]
    data = DataProto.from_dict(tensors=dict(
        input_ids=original.batch['input_ids'], responses=original.batch['input_ids'][:,1:],
        attention_mask=original.batch['attention_mask'], position_ids=original.batch['position_ids'],
        token_level_rewards=torch.nn.functional.pad(original.batch['token_level_rewards'],(extra,0)),
        loss_mask=torch.nn.functional.pad(original.batch['response_mask'],(extra,0))),
        meta_info=dict(original.meta_info))
    data.batch['response_mask'] = compute_response_mask(data)
    data.meta_info['multi_turn'] = True
    assert torch.equal(data.batch['loss_mask'][:,-original.batch['responses'].shape[1]:], original.batch['response_mask'])
    assert not bool(data.batch['loss_mask'][:,:extra].any())
    return data

def capacity_probe(trainer):
    status['workers_initial'] = identities(); write_status()
    data = capacity_data()
    assert tuple(data.batch['input_ids'].shape) == (8,32768)
    assert bool(data.batch['attention_mask'].all())
    assert tuple(data.batch['responses'].shape) == (8,32767)
    status['input'] = dict(shape=list(data.batch['input_ids'].shape),
                           response_shape=list(data.batch['responses'].shape),
                           SHA256=hashlib.sha256(data.batch['input_ids'].contiguous().numpy().tobytes()).hexdigest(),
                           global_token_num=data.batch['attention_mask'].sum(-1).tolist(),
                           policy_tokens=data.batch['loss_mask'].sum(-1).tolist(),
                           response_axis='32767 positions; same original policy mask, added history masked out',
                           natural_task_quality_claimed=False)
    status['status'] = 'native_warm_critic_checkpoint_restore'; write_status()
    cached = copy_to_local(str(CHECKPOINT), use_shm=True)
    trainer.critic_wg.load_checkpoint(cached, del_local_after_load=False)
    status['checkpoint_loaded_from'] = cached
    for label, call in (
        ('old_log_prob', lambda: trainer.actor_rollout_wg.compute_log_prob(data)),
        ('values', lambda: trainer.critic_wg.compute_values(data)),
        ('adv', lambda: compute_advantage(data, AdvantageEstimator.GAE,
                                         gamma=trainer.config.algorithm.gamma, lam=trainer.config.algorithm.lam)),
        ('update_critic', lambda: trainer.critic_wg.update_critic(data)),
        ('update_actor', lambda: trainer.actor_rollout_wg.update_actor(data)),
    ):
        status['status'] = label; write_status()
        start = time.perf_counter(); result = call(); seconds = time.perf_counter()-start
        if label in ('old_log_prob','values'):
            data.batch.update(result.batch)
            data.meta_info.update(result.meta_info)
            assert all(bool(torch.isfinite(v).all()) for v in result.batch.values())
        elif label == 'adv':
            data = result
            assert bool(torch.isfinite(data.batch['advantages']).all())
        metrics = result.meta_info.get('metrics',{})
        status['stages'].append(dict(stage=label, seconds=seconds, native_metrics=metrics))
        write_status()
    status['workers_final'] = identities()
    expected = json.loads(PREPARED.read_bytes())['source_bindings']
    for worker in status['workers_final']:
        assert worker['FA_deterministic'] == '0'
        assert {v['kind'] for v in worker['roles'].values()} == {'actor_rollout','critic'}
        assert all(v['microbatch'] == 4 for v in worker['roles'].values())
        for module in worker['modules'].values():
            if module['path'] in expected:
                assert module['SHA256'] == expected[module['path']]
    status['status'] = 'completed_native_colocated_capacity'
    status['finished'] = time.time(); write_status()

cfg_path = ROOT/'candidates/appworld-native-PPO-32k-speed-20261016-v3/effective-config.yaml'
cfg = OmegaConf.load(cfg_path)
cfg.trainer.experiment_name = VERSION
cfg.trainer.default_local_dir = str(AUDIT/'unused-checkpoints')
cfg.ray_init.runtime_env.env_vars.DT_FSDP_EVENT_WAIT_RECEIPTS = str(AUDIT/'worker-runtime')
assert cfg.algorithm.adv_estimator == 'gae'
assert cfg.actor_rollout_ref.rollout.max_model_len == 32768
assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 4
assert cfg.critic.ppo_micro_batch_size_per_gpu == 4
assert cfg.critic.model.enable_activation_offload
assert cfg.critic.model.activation_offload_native_max_seq_len == 8192
status['effective_configuration'] = OmegaConf.to_container(cfg, resolve=True)
native_run = main_ppo.TaskRunner.__ray_metadata__.modified_class.run
status['original_TaskRunner'] = dict(path=inspect.getsourcefile(native_run),
                                   SHA256=hashlib.sha256(Path(inspect.getsourcefile(native_run)).read_bytes()).hexdigest(),
                                   test_seam='Temporary fit callback only; original TaskRunner initialization invoked unchanged')
if '--preflight' in sys.argv:
    data = capacity_data()
    assert tuple(data.batch['input_ids'].shape) == (8,32768)
    assert bool(data.batch['attention_mask'].all())
    assert tuple(data.batch['response_mask'].shape) == tuple(data.batch['responses'].shape)
    assert tuple(data.batch['responses'].shape) == (8,32767)
    assert not cfg.actor_rollout_ref.actor.use_kl_loss
    receipt = dict(status='CPU_preflight_passed', source_commit=status['source_commit'],
                   context=32768, response_length=32767, batch=8, microbatch_per_rank=4,
                   original_TaskRunner=status['original_TaskRunner'],
                   effective_configuration=OmegaConf.to_container(cfg, resolve=True),
                   model_calls=0, optimizer_updates=0, source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (AUDIT/'CPU-preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','context','batch','microbatch_per_rank','model_calls','optimizer_updates')}))
    raise SystemExit(0)
write_status()
env = dict(os.environ); env.pop('CUDA_VISIBLE_DEVICES',None)
ray.init(address='local', num_cpus=16, num_gpus=2, include_dashboard=False,
         _temp_dir='/tmp/ppo32cap', runtime_env=dict(env_vars=env,
         worker_process_setup_hook='metax_fsdp_event_wait.worker_setup'))
try:
    with patch.object(main_ppo.RayPPOTrainer, 'fit', capacity_probe):
        native_run(None,cfg)
except BaseException as exc:
    status.update(status='failed', error=repr(exc), finished=time.time()); write_status()
    raise
finally:
    ray.shutdown()
print(json.dumps(dict(status=status['status'], stages=status['stages']),indent=2))
