"""Corresponding VERL output replay at the new 8192 native-dispatch boundary.

Reuse the existing native PerformanceCritic observer and original checkpoint.
The unchanged long branch retains its completed exact32768 numerical receipt.
FA deterministic backward is test-only; production flags and speed stay intact.
"""
import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT/'receipts/PPO-short-numeric-reference-20261016-v1/execution2'
CANDIDATE = ROOT/'candidates/PPO-32k-native-efficiency-20261016-v4/verl-owner'
if '--native' not in sys.argv:
    import re
    smi = subprocess.check_output(['mx-smi'],text=True)
    assert not any(re.match(r'\|\s*(?:2|3)\s+\d+\s+\S+',v) for v in smi.splitlines()), 'Physical2/3 occupied'
    prepared = json.loads((ROOT/'candidates/appworld-native-PPO-32k-speed-20261016-v3/prepared.json').read_bytes())
    env = dict(os.environ); env.update(prepared['effective_environment'])
    env['VERL_ROOT'] = str(CANDIDATE); env['PYTHONPATH'] = str(CANDIDATE)+':'+env['PYTHONPATH']
    env['CUDA_VISIBLE_DEVICES'] = '2,3'; env.pop('MACA_VISIBLE_DEVICES',None)
    env['FLASH_ATTENTION_DETERMINISTIC'] = '1'
    env['DT_FSDP_EVENT_WAIT_RECEIPTS'] = str(AUDIT/'worker-runtime')
    extra = ['--preflight'] if '--preflight' in sys.argv else []
    raise SystemExit(subprocess.call([env['VENV_PYTHON'],__file__,'--native']+extra,env=env))

import numpy as np
import pickle
import psutil
import random
import ray
import torch
from omegaconf import OmegaConf
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import CriticWorker
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage, compute_response_mask, AdvantageEstimator

owner = ROOT/'receipts/PPO-32k-native-efficiency-20261016-v4/source/run_performance.py'
expected = '5e0ef17663b583e502b27b69846458967fe39f2c94a946d7e82d67c8a0f81112'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == expected
tree = ast.parse(owner.read_text())
nodes = [v for v in tree.body if isinstance(v,ast.ClassDef) and v.name == 'PerformanceCritic']
assert len(nodes) == 1
helper = ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/source/run_same_session.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest() == 'f4a22d506ba958a0259ab39b6407852269d42438bc5614eb5fa2b71352f8bd09'
helpers = [v for v in ast.parse(helper.read_text()).body
           if isinstance(v,ast.FunctionDef) and v.name in ('write','local','fingerprint','state')]
assert len(helpers) == 4
exec(compile(ast.Module(body=helpers,type_ignores=[]),str(helper),'exec'),globals())
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(owner),'exec'),globals())
AUDIT.mkdir(parents=True,exist_ok=True)
prepared = json.loads((ROOT/'candidates/appworld-native-ppo-speed-20261016-v1/prepared.json').read_bytes())
cfg = OmegaConf.load(CANDIDATE/'verl/trainer/config/ppo_trainer.yaml')
for key,value in prepared['options'].items():
    OmegaConf.update(cfg,key.lstrip('+'),value,force_add=True)
cfg.critic.model.enable_activation_offload = False
cfg.critic.optim.total_training_steps = 2
data = torch.load(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v1/baseline-fit/native-input-with-GAE.pt',weights_only=False,map_location='cpu')
assert tuple(data.batch['input_ids'].shape) == (8,8192)
original = data
extra = 8191-original.batch['responses'].shape[1]
data = DataProto.from_dict(tensors=dict(input_ids=original.batch['input_ids'],
    responses=original.batch['input_ids'][:,1:], attention_mask=original.batch['attention_mask'],
    position_ids=original.batch['position_ids'],
    token_level_rewards=torch.nn.functional.pad(original.batch['token_level_rewards'],(extra,0)),
    loss_mask=torch.nn.functional.pad(original.batch['response_mask'],(extra,0))),meta_info=dict(original.meta_info))
data.batch['response_mask'] = compute_response_mask(data)
data.meta_info['multi_turn'] = True
assert tuple(data.batch['responses'].shape) == (8,8191)
assert not bool(data.batch['loss_mask'][:,:extra].any())
assert cfg.critic.ppo_micro_batch_size_per_gpu == 4
status = dict(status='initializing',started=time.time(),pid=os.getpid(),birth=psutil.Process().create_time(),
              source_commit=os.environ['PPO_32K_SOURCE_COMMIT'],physical_gpus=[2,3],
              context=8192,response_length=8191,batch=8,microbatch_per_rank=4,test_only_FA_deterministic=True,
              production_FA_deterministic=False,formal_training_mutations=0,checkpoint_create_calls=0,
              criterion=dict(owner='Pinned VERL tests/utils/gpu_tests/test_activation_offload.py:123',atol=0.0,rtol=0.0),
              reuse_original_test_driver=dict(path=str(owner),SHA256=expected),
              long_32768_unchanged_receipt=str(ROOT/'receipts/PPO-32k-numeric-reference-20261016-v3/same-session-status.json'))
if '--preflight' in sys.argv:
    write(AUDIT/'CPU-preflight.json',dict(status='CPU_preflight_passed',context=8192,response_length=8191,batch=8,microbatch_per_rank=4,
                                        model_calls=0,optimizer_updates=0,criterion=status['criterion'],
                                        source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('8192 B4/rank native replay CPU preflight passed')
    raise SystemExit(0)
write(AUDIT/'status.json',status)
checkpoint = copy_to_local(str(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/performance-exec2/numerical-warm-state-original-VERL'),use_shm=True)
env = dict(os.environ); env.pop('CUDA_VISIBLE_DEVICES',None)
ray.init(address='local',num_cpus=8,num_gpus=2,include_dashboard=False,_temp_dir='/tmp/ppo8num2',
         runtime_env=dict(env_vars=env,worker_process_setup_hook='metax_fsdp_event_wait.worker_setup'))
try:
    group = RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),RayClassWithInitArgs(PerformanceCritic,cfg.critic))
    status['workers'] = group.initialize(); group.memory_mode('adaptive')
    group.memory_mode('plain'); group.load_checkpoint(checkpoint,del_local_after_load=False)
    data.batch.update(group.compute_values(data).batch)
    data = compute_advantage(data,AdvantageEstimator.GAE,gamma=cfg.algorithm.gamma,lam=cfg.algorithm.lam)
    outputs = {}; probe = None; digest = None
    for label,mode in [('reference','plain'),('repeat','plain'),('adaptive','adaptive')]:
        status['status'] = label; write(AUDIT/'status.json',status)
        group.memory_mode(mode); group.load_checkpoint(checkpoint,del_local_after_load=False)
        current = group.restore_probe()
        if probe is None: probe=current
        assert current == probe
        if label in ('reference','adaptive'):
            current_digest = group.restore_probe(full=True)
            if digest is None: digest=current_digest
            assert current_digest == digest
        if mode == 'adaptive': result=group.verify_native_dispatch(data)
        else: result=group.update_critic(data)
        outputs[label] = group.compute_values(data).batch['values']
        torch.save(outputs[label],AUDIT/(label+'-post-values.pt'))
        status.setdefault('completed',[]).append(dict(label=label,native_metrics=result.meta_info['metrics'],
                                                    checkpoint_RNG_restore_probe_exact=True))
        write(AUDIT/'status.json',status)
    status['comparisons'] = {}
    for label in ('repeat','adaptive'):
        status['comparisons'][label] = dict(max_abs=float((outputs['reference']-outputs[label]).abs().max()),
                                           exact=torch.equal(outputs['reference'],outputs[label]))
        torch.testing.assert_close(outputs['reference'],outputs[label],atol=0.0,rtol=0.0)
    status.update(status='completed_official_output_criterion_passed',finished=time.time())
    write(AUDIT/'status.json',status)
except BaseException as exc:
    status.update(status='failed',error=repr(exc),finished=time.time()); write(AUDIT/'status.json',status)
    raise
finally:
    ray.shutdown()
print(json.dumps(dict(status=status['status'],comparisons=status.get('comparisons')),indent=2))
