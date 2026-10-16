"""Native fast-mode PPO cost, matched full state, and exact offload byte transport.

Uses only the original critic update/optimizer/checkpoint and native offloader.
Diagnostic observers are absent from every timed call. No formal training.
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
AUDIT=ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/performance'
CANDIDATE=ROOT/'candidates/PPO-32k-native-efficiency-20261016-v2/verl-owner'
if '--native' not in sys.argv:
    import re
    p=json.loads((ROOT/'candidates/appworld-native-ppo-speed-20261016-v1/prepared.json').read_bytes())
    smi=subprocess.check_output(['mx-smi'],text=True)
    assert not any(re.match(r'\|\s*(?:6|7)\s+\d+\s+\S+',line) for line in smi.splitlines()),'Physical6/7 occupied'
    env=dict(os.environ);env.update(p['effective_environment'])
    env['VERL_ROOT']=str(CANDIDATE);env['PYTHONPATH']=str(CANDIDATE)+':'+env['PYTHONPATH']
    env['CUDA_VISIBLE_DEVICES']='6,7';env['FLASH_ATTENTION_DETERMINISTIC']='0'
    env['DT_FSDP_EVENT_WAIT_RECEIPTS']=str(AUDIT/'worker-runtime')
    env.pop('MACA_VISIBLE_DEVICES',None)
    raise SystemExit(subprocess.call([env['VENV_PYTHON'],__file__,'--native'],env=env))

import numpy as np
import pickle
import psutil
import random
import ray
import statistics
import torch
from omegaconf import OmegaConf
from verl.single_controller.base.decorator import Dispatch,register
from verl.single_controller.ray import RayClassWithInitArgs,RayResourcePool,RayWorkerGroup
from verl.workers.fsdp_workers import CriticWorker

# Reuse the existing diagnostic state reader; no second model/optimizer loader.
helper_source=ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/source/run_same_session.py'
assert hashlib.sha256(helper_source.read_bytes()).hexdigest()=='f4a22d506ba958a0259ab39b6407852269d42438bc5614eb5fa2b71352f8bd09'
tree=ast.parse(helper_source.read_text())
helpers=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('write','local','fingerprint','state')]
assert len(helpers)==4
exec(compile(ast.Module(body=helpers,type_ignores=[]),str(helper_source),'exec'),globals())

@ray.remote
class PerformanceCritic(CriticWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self):
        random.seed(20261016);np.random.seed(20261016);torch.manual_seed(20261016)
        super().init_model()
        self._plain={id(layer):(layer,layer._gradient_checkpointing_func) for layer in self.critic_module.model.language_model.layers}
        self._async=None
        return dict(rank=self.rank,pid=os.getpid(),birth=psutil.Process().create_time(),
            physical_CVD=os.environ['CUDA_VISIBLE_DEVICES'],FA_deterministic=os.environ['FLASH_ATTENTION_DETERMINISTIC'],
            sources={name:dict(path=sys.modules[name].__file__,SHA256=hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest())
                for name in ('verl.workers.fsdp_workers','verl.workers.critic.dp_critic','verl.trainer.ppo.core_algos','verl.utils.activation_offload')})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def memory_mode(self,mode):
        if mode=='async' and self._async is None:
            from verl.utils.activation_offload import enable_activation_offloading
            enable_activation_offloading(self.critic_module.model.language_model,'fsdp2',True,preserve_hf_checkpoint=True)
            self._async={id(layer):layer._gradient_checkpointing_func for layer in self.critic_module.model.language_model.layers}
        for key,(layer,fn) in self._plain.items():layer._gradient_checkpointing_func=self._async[key] if mode=='async' else fn
        self._mode=mode

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def state_digest(self):
        return dict(rank=self.rank,SHA256=hashlib.sha256(json.dumps(state(self),sort_keys=True).encode()).hexdigest())

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def native_update(self,data):
        ctx=torch.autograd.graph.save_on_cpu(pin_memory=True) if self._mode=='CPU_reference' else contextlib.nullcontext()
        with ctx:return super().update_critic(data)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def verify_transport(self,data):
        assert self._mode=='async'
        fn=next(iter(self._async.values()))
        closure=dict(zip(fn.__code__.co_freevars,[c.cell_contents for c in fn.__closure__]))
        owner=closure['handler']._offload_ctx.offload_handler
        original_push,original_pop=owner.tensor_push,owner.tensor_pop
        expected={};records=[]
        def push(tensor,**kwargs):
            tag=original_push(tensor,**kwargs)
            if not isinstance(tag,torch.Tensor):
                expected[tag]=dict(value=fingerprint(tensor),stride=list(tensor.stride()),device=str(tensor.device))
            return tag
        def pop(tag,**kwargs):
            tensor=original_pop(tag,**kwargs)
            if not isinstance(tag,torch.Tensor):
                before=expected.pop(tag)
                after=dict(value=fingerprint(tensor),stride=list(tensor.stride()),device=str(tensor.device))
                assert before==after,('Native offload transport changed bytes or tensor metadata',tag,before,after)
                records.append(dict(tag=list(tag),elements=tensor.numel(),bytes=tensor.numel()*tensor.element_size(),
                    dtype=str(tensor.dtype),shape=list(tensor.shape),all_bytes_SHA256_equal=True,stride_and_device_equal=True))
            return tensor
        owner.tensor_push,owner.tensor_pop=push,pop
        try:result=super().update_critic(data)
        finally:owner.tensor_push,owner.tensor_pop=original_push,original_pop
        assert not expected
        assert any(item['shape']==[4,32768,4096] for item in records)
        write(AUDIT/f'exact-transport-rank{self.rank}.json',dict(status='passed',records=records,
            copied_elements=sum(v['elements'] for v in records),copied_bytes=sum(v['bytes'] for v in records),
            criterion='All native saved input bytes and tensor metadata equal. No dtype conversion or numerical tolerance.',
            observer_removed_before_timing=True,native_metrics=result.meta_info['metrics']))
        return result

AUDIT.mkdir(exist_ok=True)
prepared=json.loads((ROOT/'candidates/appworld-native-ppo-speed-20261016-v1/prepared.json').read_bytes())
cfg=OmegaConf.load(CANDIDATE/'verl/trainer/config/ppo_trainer.yaml')
for key,value in prepared['options'].items():OmegaConf.update(cfg,key.lstrip('+'),value,force_add=True)
cfg.critic.model.enable_activation_offload=False;cfg.critic.optim.total_training_steps=2
small=torch.load(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v1/baseline-fit/native-input-with-GAE.pt',weights_only=False,map_location='cpu')
large=torch.load(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/same-session-native-data-32768.pt',weights_only=False,map_location='cpu')
status=dict(status='initializing',started=time.time(),pid=os.getpid(),birth=psutil.Process().create_time(),
    source_commit=os.environ['PPO_32K_SOURCE_COMMIT'],physical_gpus=[6,7],FA_deterministic=False,
    formal_mutations=0,rollout_calls=0,DT_calls=0,scope=__doc__,critic_config=OmegaConf.to_container(cfg.critic,resolve=True),
    limitations=['B8 native capacity input; original global32/epochs2 preserved, not formal full-iteration throughput',
        '8192 has an original no-activation-offload baseline; original32768 no-offload already OOM, so32k uses official CPU saving reference',
        'Fast default native backward can vary numerically; no diagnostic hashing/observers in measured calls'])
write(AUDIT/'status.json',status)
env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
ray.init(address='local',num_cpus=8,num_gpus=2,include_dashboard=False,_temp_dir='/tmp/ray-PPO32k-fast-performance-v2',
    runtime_env=dict(env_vars=env,worker_process_setup_hook='metax_fsdp_event_wait.worker_setup'))
try:
    group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),RayClassWithInitArgs(PerformanceCritic,cfg.critic))
    status['workers']=group.initialize();group.memory_mode('plain')
    status['status']='original_warmup';write(AUDIT/'status.json',status)
    group.native_update(small)
    checkpoint=AUDIT/'numerical-warm-state-original-VERL'
    group.save_checkpoint(str(checkpoint),global_step=1,max_ckpt_to_keep=None)
    saved=group.state_digest()
    status['status']='exact_transport_32768';write(AUDIT/'status.json',status)
    group.memory_mode('async');group.load_checkpoint(str(checkpoint),del_local_after_load=False)
    assert group.state_digest()==saved
    group.verify_transport(large)
    status['exact_transport']='passed'
    measurements=[]
    for length,mode in [(8192,'plain'),(8192,'async'),(8192,'async'),(8192,'plain'),(8192,'plain'),(8192,'async'),
                        (32768,'CPU_reference'),(32768,'async')]:
        status['status']=f'timing_{length}_{mode}_{len(measurements)+1}';write(AUDIT/'status.json',status)
        group.memory_mode(mode);group.load_checkpoint(str(checkpoint),del_local_after_load=False)
        assert group.state_digest()==saved, 'Native full model/optimizer/scheduler/RNG must match before every timed call'
        start=time.perf_counter();result=group.native_update(small if length==8192 else large)
        measurements.append(dict(context=length,mode=mode,seconds=time.perf_counter()-start,
            native_metrics=result.meta_info['metrics'],initial_full_state_RNG_exact=True))
        status['measurements']=measurements;write(AUDIT/'status.json',status)
    for length,reference in [(8192,'plain'),(32768,'CPU_reference')]:
        a=[v['seconds'] for v in measurements if v['context']==length and v['mode']==reference]
        b=[v['seconds'] for v in measurements if v['context']==length and v['mode']=='async']
        status.setdefault('comparisons',[]).append(dict(context=length,reference=reference,reference_seconds=a,
            async_seconds=b,median_ratio=statistics.median(b)/statistics.median(a)))
    status.update(status='completed',finished=time.time());write(AUDIT/'status.json',status)
except BaseException as exc:
    status.update(status='failed',error=repr(exc),finished=time.time());write(AUDIT/'status.json',status);raise
finally:ray.shutdown()
print(json.dumps(status,indent=2))
