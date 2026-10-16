"""Reuse native replay observations to localize the failed full response update.

The unchanged v2 ReplayCritic owns the removable gradient/state observations;
native CriticWorker/GAE/PPO/optimizer still own all calculations. Only supply
v4's actual8192 dispatch bound to the original offload API. No new checkpoint.
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

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/PPO-full-response-update-replay-20261017-v1'
CANDIDATE=ROOT/'candidates/PPO-32k-native-efficiency-20261016-v4/verl-owner'
if '--native' not in sys.argv:
    import re
    smi=subprocess.check_output(['mx-smi'],text=True)
    assert not any(re.match(r'\|\s*(?:2|3)\s+\d+\s+\S+',v) for v in smi.splitlines()),'Physical2/3 occupied'
    p=json.loads((ROOT/'candidates/appworld-native-PPO-32k-speed-20261016-v3/prepared.json').read_bytes())
    env=dict(os.environ);env.update(p['effective_environment'])
    env['VERL_ROOT']=str(CANDIDATE);env['PYTHONPATH']=str(CANDIDATE)+':'+env['PYTHONPATH']
    env['CUDA_VISIBLE_DEVICES']='2,3';env.pop('MACA_VISIBLE_DEVICES',None)
    env['FLASH_ATTENTION_DETERMINISTIC']='1'
    env['DT_FSDP_EVENT_WAIT_RECEIPTS']=str(AUDIT/'worker-runtime')
    extra=['--preflight'] if '--preflight' in sys.argv else []
    raise SystemExit(subprocess.call([env['VENV_PYTHON'],__file__,'--native']+extra,env=env))

import numpy as np
import pickle
import psutil
import random
import ray
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch,register
from verl.single_controller.ray import RayClassWithInitArgs,RayResourcePool,RayWorkerGroup
from verl.trainer.ppo.ray_trainer import compute_advantage,compute_response_mask,AdvantageEstimator
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import CriticWorker

owner=ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/source/run_same_session.py'
expected='f4a22d506ba958a0259ab39b6407852269d42438bc5614eb5fa2b71352f8bd09'
assert hashlib.sha256(owner.read_bytes()).hexdigest()==expected
tree=ast.parse(owner.read_text())
nodes=[v for v in tree.body if (isinstance(v,ast.FunctionDef) and v.name in ('write','local','fingerprint','state'))
       or (isinstance(v,ast.ClassDef) and v.name=='ReplayCritic')]
assert len(nodes)==5
calls=[v for node in nodes for v in ast.walk(node) if isinstance(v,ast.Call)
       and isinstance(v.func,ast.Name) and v.func.id=='enable_activation_offloading']
assert len(calls)==1
calls[0].keywords.append(ast.keyword(arg='native_checkpoint_max_seq_len',value=ast.Constant(value=8192)))
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(owner),'exec'),globals())
AUDIT.mkdir(parents=True,exist_ok=True)
prepared=json.loads((ROOT/'candidates/appworld-native-ppo-speed-20261016-v1/prepared.json').read_bytes())
cfg=OmegaConf.load(CANDIDATE/'verl/trainer/config/ppo_trainer.yaml')
for key,value in prepared['options'].items():OmegaConf.update(cfg,key.lstrip('+'),value,force_add=True)
cfg.critic.model.enable_activation_offload=False;cfg.critic.optim.total_training_steps=2
original=torch.load(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v1/baseline-fit/native-input-with-GAE.pt',weights_only=False,map_location='cpu')
extra=8191-original.batch['responses'].shape[1]
data=DataProto.from_dict(tensors=dict(input_ids=original.batch['input_ids'],responses=original.batch['input_ids'][:,1:],
    attention_mask=original.batch['attention_mask'],position_ids=original.batch['position_ids'],
    token_level_rewards=torch.nn.functional.pad(original.batch['token_level_rewards'],(extra,0)),
    loss_mask=torch.nn.functional.pad(original.batch['response_mask'],(extra,0))),meta_info=dict(original.meta_info))
data.batch['response_mask']=compute_response_mask(data);data.meta_info['multi_turn']=True
assert tuple(data.batch['input_ids'].shape)==(8,8192) and tuple(data.batch['responses'].shape)==(8,8191)
status=dict(status='initializing',started=time.time(),pid=os.getpid(),birth=psutil.Process().create_time(),
            source_commit=os.environ['PPO_32K_SOURCE_COMMIT'],physical_gpus=[2,3],context=8192,response_length=8191,
            microbatch_per_rank=4,formal_training_mutations=0,checkpoint_create_calls=0,
            scope=__doc__,test_only_FA_deterministic=True,production_FA_deterministic=False,
            original_replay_owner=dict(path=str(owner),SHA256=expected,change='Only actual v4 native_checkpoint_max_seq_len=8192'),completed=[])
if '--preflight' in sys.argv:
    write(AUDIT/'CPU-preflight.json',dict(status='CPU_preflight_passed',context=8192,response_length=8191,
                                       model_calls=0,optimizer_updates=0,source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('Native full-response gradient/state observation CPU preflight passed')
    raise SystemExit(0)
write(AUDIT/'status.json',status)
checkpoint=copy_to_local(str(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/performance-exec2/numerical-warm-state-original-VERL'),use_shm=True)
env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
ray.init(address='local',num_cpus=8,num_gpus=2,include_dashboard=False,_temp_dir='/tmp/ppoupd1',
         runtime_env=dict(env_vars=env,worker_process_setup_hook='metax_fsdp_event_wait.worker_setup'))
try:
    group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),RayClassWithInitArgs(ReplayCritic,cfg.critic))
    status['workers']=group.initialize_replay();group.set_memory_mode('native_plain')
    group.load_checkpoint(checkpoint,del_local_after_load=False)
    data.batch.update(group.compute_values(data).batch)
    data=compute_advantage(data,AdvantageEstimator.GAE,gamma=cfg.algorithm.gamma,lam=cfg.algorithm.lam)
    torch.save(data,AUDIT/'native-full-response-data-with-GAE.pt')
    input_digest={k:fingerprint(v) for k,v in data.batch.items()}
    baseline=group.observe_state('original_saved')
    outputs={}
    for label,mode in [('native_reference','native_plain'),('native_repeat','native_plain'),('native_async','native_async')]:
        status['status']=label;write(AUDIT/'status.json',status)
        group.set_memory_mode(mode);group.load_checkpoint(checkpoint,del_local_after_load=False)
        restored=group.observe_state(label+'_restored');assert restored==baseline
        assert {k:fingerprint(v) for k,v in data.batch.items()}==input_digest
        data.meta_info['replay_label']=label
        result=group.replay_update(data)
        output=group.compute_values(data);outputs[label]=output.batch['values'].clone()
        torch.save(outputs[label],AUDIT/(label+'-post-values.pt'))
        status['completed'].append(dict(label=label,native_metrics=result.meta_info['metrics'],
                                         full_preupdate_state_restore_exact=True,input_fields_exact=True))
        write(AUDIT/'status.json',status)
    status['comparisons']={label:dict(max_abs=float((outputs['native_reference']-v).abs().max()),
                                      different=int(torch.count_nonzero(outputs['native_reference']-v))) for label,v in outputs.items()}
    status['instrumented_post_values_exact']=all(v['max_abs']==0 for v in status['comparisons'].values())
    status['acceptance_limit']='Diagnostic gradient/state reads can synchronize GPU work; this is localization, not an unobserved production equivalence or speed claim.'
    status.update(status='completed_instrumented_native_update_measurements',finished=time.time());write(AUDIT/'status.json',status)
except BaseException as exc:
    status.update(status='failed',error=repr(exc),finished=time.time());write(AUDIT/'status.json',status);raise
finally:ray.shutdown()
print(json.dumps(status,indent=2))
