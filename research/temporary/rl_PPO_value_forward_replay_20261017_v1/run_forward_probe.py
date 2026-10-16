"""Localize the failed native post-update value replay, with zero updates.

Use the original native compute_values; a removable observer copies its GPU
microbatch result for comparison to the original CPU/DP return. No new critic,
forward, optimizer, training objective or numerical tolerance implementation.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/PPO-value-forward-replay-20261017-v1'
CANDIDATE=ROOT/'candidates/PPO-32k-native-efficiency-20261016-v4/verl-owner'
if '--native' not in sys.argv:
    import re
    smi=subprocess.check_output(['mx-smi'],text=True)
    assert not any(re.match(r'\|\s*(?:2|3)\s+\d+\s+\S+',v) for v in smi.splitlines()),'Physical2/3 occupied'
    prepared=json.loads((ROOT/'candidates/appworld-native-PPO-32k-speed-20261016-v3/prepared.json').read_bytes())
    env=dict(os.environ);env.update(prepared['effective_environment'])
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
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import CriticWorker

helper=ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/source/run_same_session.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest()=='f4a22d506ba958a0259ab39b6407852269d42438bc5614eb5fa2b71352f8bd09'
nodes=[v for v in ast.parse(helper.read_text()).body if isinstance(v,ast.FunctionDef) and v.name in ('write','local','fingerprint','state')]
assert len(nodes)==4
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(helper),'exec'),globals())

@ray.remote
class ForwardProbe(CriticWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize_probe(self):
        random.seed(20261016);np.random.seed(20261016);torch.manual_seed(20261016)
        super().init_model()
        return dict(rank=self.rank,pid=os.getpid(),birth=psutil.Process().create_time(),
                    physical_CVD=os.environ['CUDA_VISIBLE_DEVICES'],FA_deterministic=os.environ['FLASH_ATTENTION_DETERMINISTIC'])

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def state_digest(self):
        value=state(self)
        value['buffers']={k:fingerprint(v) for k,v in self.critic_module.named_buffers()}
        return dict(rank=self.rank,SHA256=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest(),
                    components={k:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for k,v in value.items()})

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def observed_values(self,data):
        label=data.meta_info['probe_label'];original=self.critic._forward_micro_batch;copies=[]
        def observe(micro_batch):
            result=original(micro_batch)
            torch.cuda.synchronize()
            copies.append(result.detach().to('cpu',non_blocking=False).clone())
            return result
        self.critic._forward_micro_batch=observe
        try:
            result=super().compute_values(data)
            torch.cuda.synchronize()
        finally:
            self.critic._forward_micro_batch=original
        expected=torch.cat(copies,dim=0)
        mask=data.batch['attention_mask'][:,-expected.shape[-1]-1:-1].to('cpu',non_blocking=False)
        expected=expected*mask
        actual=result.batch['values']
        torch.save(expected,AUDIT/f'{label}-GPU-captured-rank{self.rank}.pt')
        write(AUDIT/f'{label}-handoff-rank{self.rank}.json',dict(
            GPU_to_native_CPU_return_exact=torch.equal(expected,actual),max_abs=float((expected-actual).abs().max()),
            shape=list(expected.shape),observer_removed=True,optimizer_updates=0))
        return result

AUDIT.mkdir(parents=True,exist_ok=True)
prepared=json.loads((ROOT/'candidates/appworld-native-ppo-speed-20261016-v1/prepared.json').read_bytes())
cfg=OmegaConf.load(CANDIDATE/'verl/trainer/config/ppo_trainer.yaml')
for key,value in prepared['options'].items():OmegaConf.update(cfg,key.lstrip('+'),value,force_add=True)
cfg.critic.model.enable_activation_offload=False
cfg.critic.optim.total_training_steps=2
assert not cfg.critic.use_dynamic_bsz
original=torch.load(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v1/baseline-fit/native-input-with-GAE.pt',weights_only=False,map_location='cpu')
data=DataProto.from_dict(tensors=dict(input_ids=original.batch['input_ids'],responses=original.batch['input_ids'][:,1:],
    attention_mask=original.batch['attention_mask'],position_ids=original.batch['position_ids']),meta_info=dict(original.meta_info))
assert tuple(data.batch['input_ids'].shape)==(8,8192) and tuple(data.batch['responses'].shape)==(8,8191)
status=dict(status='initializing',started=time.time(),pid=os.getpid(),birth=psutil.Process().create_time(),
            source_commit=os.environ['PPO_32K_SOURCE_COMMIT'],physical_gpus=[2,3],context=8192,response_length=8191,
            microbatch_per_rank=4,optimizer_updates=0,checkpoint_create_calls=0,formal_training_mutations=0,
            native_activation_offload=False,completed=[])
if '--preflight' in sys.argv:
    write(AUDIT/'CPU-preflight.json',dict(status='CPU_preflight_passed',source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                       context=8192,response_length=8191,optimizer_updates=0,model_calls=0))
    print('Native forward-only replay CPU preflight passed')
    raise SystemExit(0)
write(AUDIT/'status.json',status)
checkpoint=copy_to_local(str(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/performance-exec2/numerical-warm-state-original-VERL'),use_shm=True)
env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
ray.init(address='local',num_cpus=8,num_gpus=2,include_dashboard=False,_temp_dir='/tmp/ppofwd1',
         runtime_env=dict(env_vars=env,worker_process_setup_hook='metax_fsdp_event_wait.worker_setup'))
try:
    group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),RayClassWithInitArgs(ForwardProbe,cfg.critic))
    status['workers']=group.initialize_probe();group.load_checkpoint(checkpoint,del_local_after_load=False)
    status['before']=group.state_digest();outputs={}
    for label,observed in [('native1',False),('native2',False),('observed1',True),('observed2',True),('native3',False)]:
        status['status']=label;write(AUDIT/'status.json',status);data.meta_info['probe_label']=label
        result=group.observed_values(data) if observed else group.compute_values(data)
        outputs[label]=result.batch['values'].clone()
        torch.save(outputs[label],AUDIT/(label+'-values.pt'))
        status['completed'].append(label);write(AUDIT/'status.json',status)
    status['after']=group.state_digest();status['full_state_and_buffers_unchanged']=status['before']==status['after']
    status['comparisons']={label:dict(max_abs=float((outputs['native1']-v).abs().max()),different=int(torch.count_nonzero(outputs['native1']-v))) for label,v in outputs.items()}
    status.update(status='completed_forward_probe',finished=time.time());write(AUDIT/'status.json',status)
except BaseException as exc:
    status.update(status='failed',error=repr(exc),finished=time.time());write(AUDIT/'status.json',status);raise
finally:ray.shutdown()
print(json.dumps(status,indent=2))
