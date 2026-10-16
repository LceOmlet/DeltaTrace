"""Native same-session checkpoint replay at exact B4/rank x32768.

Use the pinned VERL checkpoint/PPO/GAE and original HF checkpoint callable.
The saved states are diagnostic artifacts, never formal progress/checkpoints.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION='PPO-32k-native-efficiency-20261016-v2'
AUDIT=ROOT/'receipts'/VERSION
CANDIDATE=ROOT/'candidates'/VERSION/'verl-owner'
if '--native' not in sys.argv:
    import re
    p=json.loads((ROOT/'candidates/appworld-native-ppo-speed-20261016-v1/prepared.json').read_bytes())
    smi=subprocess.check_output(['mx-smi'],text=True)
    assert not any(re.match(r'\|\s*(?:2|3)\s+\d+\s+\S+',line) for line in smi.splitlines()),'Physical2/3 occupied'
    env=dict(os.environ);env.update(p['effective_environment'])
    env['VERL_ROOT']=str(CANDIDATE);env['PYTHONPATH']=str(CANDIDATE)+':'+env['PYTHONPATH']
    env['CUDA_VISIBLE_DEVICES']='2,3';env.pop('MACA_VISIBLE_DEVICES',None)
    raise SystemExit(subprocess.call([env['VENV_PYTHON'],__file__,'--native'],env=env))

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
from verl.workers.fsdp_workers import CriticWorker

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def local(t):
    if hasattr(t,'to_local'):t=t.to_local()
    return t.detach().cpu().contiguous()

def fingerprint(t):
    value=local(t)
    return dict(shape=list(value.shape),dtype=str(value.dtype),
        SHA256=hashlib.sha256(memoryview(value.reshape(-1).view(torch.uint8).numpy())).hexdigest())

def state(w):
    s=w.critic_optimizer.state_dict()
    return dict(parameters={k:fingerprint(v) for k,v in w.critic_module.named_parameters()},
        optimizer={str(k):{n:fingerprint(v) if torch.is_tensor(v) else v for n,v in item.items()} for k,item in s['state'].items()},
        param_groups=s['param_groups'],scheduler=w.critic_lr_scheduler.state_dict(),
        RNG=dict(torch=fingerprint(torch.get_rng_state()),cuda=[fingerprint(v) for v in torch.cuda.get_rng_state_all()],
            python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
            numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest()))

@ray.remote
class ReplayCritic(CriticWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize_replay(self):
        random.seed(20261016);np.random.seed(20261016);torch.manual_seed(20261016)
        super().init_model()
        self._native_checkpoints={id(layer):(layer,layer._gradient_checkpointing_func)
            for layer in self.critic_module.model.language_model.layers}
        self._offloaded_checkpoints=None
        self._reference_gradients=[]
        return dict(rank=self.rank,pid=os.getpid(),birth=psutil.Process().create_time(),
            physical_CVD=os.environ.get('CUDA_VISIBLE_DEVICES'),classifier_dropout=self.critic_model_config.classifier_dropout,
            sources={name:dict(path=sys.modules[name].__file__,SHA256=hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest())
                for name in ('verl.workers.fsdp_workers','verl.workers.critic.dp_critic','verl.trainer.ppo.core_algos','verl.utils.activation_offload')})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_memory_mode(self,mode):
        if mode=='native_async' and self._offloaded_checkpoints is None:
            from verl.utils.activation_offload import enable_activation_offloading
            enable_activation_offloading(self.critic_module.model.language_model,'fsdp2',True,preserve_hf_checkpoint=True)
            self._offloaded_checkpoints={id(layer):layer._gradient_checkpointing_func
                for layer in self.critic_module.model.language_model.layers}
            for key,(layer,original) in self._native_checkpoints.items():
                assert any(cell.cell_contents is original for cell in layer._gradient_checkpointing_func.__closure__)
        for key,(layer,original) in self._native_checkpoints.items():
            layer._gradient_checkpointing_func=self._offloaded_checkpoints[key] if mode=='native_async' else original
        self._memory_mode=mode
        return dict(rank=self.rank,mode=mode,original_HF_callables_preserved=True,active_text_groups=32)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def observe_state(self,label):
        value=state(self)
        write(AUDIT/f'{label}-state-rank{self.rank}.json',value)
        return dict(rank=self.rank,whole_state_SHA256=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest())

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def replay_update(self,data):
        label=data.meta_info['replay_label'];original=self.critic._optimizer_step;captures=[]
        def measured_native_step():
            current={k:local(p.grad).clone() for k,p in self.critic_module.named_parameters() if p.grad is not None}
            hashes={k:fingerprint(v) for k,v in current.items()}
            delta=None
            if label=='native_reference':self._reference_gradients.append(current)
            else:
                ref=self._reference_gradients[len(captures)]
                squared=0.;denom=0.;maximum=0.;different=0;count=0
                for k,v in current.items():
                    d=v-ref[k]
                    if d.numel()==0:continue
                    squared+=float(torch.linalg.vector_norm(d,dtype=torch.float64))**2
                    denom+=float(torch.linalg.vector_norm(ref[k],dtype=torch.float64))**2
                    maximum=max(maximum,float(d.abs().max()))
                    different+=int(torch.count_nonzero(d));count+=d.numel()
                delta=dict(max_abs=maximum,relative_L2=(squared/denom)**.5 if denom else None,
                    different_elements=different,total_elements=count)
            norm=original()
            captures.append(dict(preclip_gradient=hashes,error_to_native_reference=delta,
                native_grad_norm=float(norm.detach().cpu()),after=state(self)))
            return norm
        self.critic._optimizer_step=measured_native_step
        try:
            ctx=torch.autograd.graph.save_on_cpu(pin_memory=True) if self._memory_mode=='native_CPU_reference' else contextlib.nullcontext()
            with ctx:result=super().update_critic(data)
        finally:self.critic._optimizer_step=original
        write(AUDIT/f'{label}-rank{self.rank}.json',dict(captures=captures,native_metrics=result.meta_info['metrics'],
            original_optimizer_calls=len(captures),memory_mode=self._memory_mode))
        return result

prepared=json.loads((ROOT/'candidates/appworld-native-ppo-speed-20261016-v1/prepared.json').read_bytes())
cfg=OmegaConf.load(CANDIDATE/'verl/trainer/config/ppo_trainer.yaml')
for k,v in prepared['options'].items():OmegaConf.update(cfg,k.lstrip('+'),v,force_add=True)
cfg.critic.optim.total_training_steps=2
cfg.critic.model.enable_activation_offload=False
original_data=torch.load(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v1/baseline-fit/native-input-with-GAE.pt',weights_only=False,map_location='cpu')
response=original_data.batch['responses'];n=response.shape[1]
prefix=original_data.batch['input_ids'][:,:-n]
needed=32768-n
prefix=prefix.repeat(1,(needed+prefix.shape[1]-1)//prefix.shape[1])[:,:needed]
ids=torch.cat((prefix,response),dim=1);mask=torch.ones_like(ids)
from verl.utils.model import compute_position_id_with_mask
data=DataProto.from_dict(tensors=dict(input_ids=ids,responses=response,attention_mask=mask,
    position_ids=compute_position_id_with_mask(mask),token_level_rewards=original_data.batch['token_level_rewards']))
data.batch['response_mask']=compute_response_mask(data);data.meta_info['global_token_num']=mask.sum(-1).tolist()
status=dict(status='initializing_same_session_native_replay',pid=os.getpid(),birth=psutil.Process().create_time(),
    source_commit=os.environ['PPO_32K_SOURCE_COMMIT'],physical_gpus=[2,3],exact_context=32768,batch=8,microbatch_per_rank=4,
    started=time.time(),scope=__doc__,formal_mutations=0,rollout_calls=0,DT_calls=0,
    original_input_source=str(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v1/baseline-fit/native-input-with-GAE.pt'),
    original_critic_config=OmegaConf.to_container(cfg.critic,resolve=True),
    numerical_criterion='Pinned VERL offload replay: after-update native outputs atol=0, rtol=0; original self-replay, RNG and gradient errors separately reported. No widened threshold.')
write(AUDIT/'same-session-status.json',status)
env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
ray.init(address='local',num_cpus=8,num_gpus=2,include_dashboard=False,_temp_dir='/tmp/ray-PPO32k-native-checkpoint-v2',
    runtime_env=dict(env_vars=env,worker_process_setup_hook='metax_fsdp_event_wait.worker_setup'))
try:
    group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),RayClassWithInitArgs(ReplayCritic,cfg.critic))
    status['workers']=group.initialize_replay()
    status['status']='original_native_values_32768';write(AUDIT/'same-session-status.json',status)
    values=group.compute_values(data);data=data.union(values)
    data=compute_advantage(data,AdvantageEstimator.GAE,gamma=cfg.algorithm.gamma,lam=cfg.algorithm.lam)
    torch.save(data,AUDIT/'same-session-native-data-32768.pt')
    checkpoint=AUDIT/'numerical-state-original-VERL'
    group.save_checkpoint(str(checkpoint),global_step=0,max_ckpt_to_keep=None)
    saved=group.observe_state('original_saved')
    for label,mode in [('native_reference','native_CPU_reference'),('native_repeat','native_CPU_reference'),('native_async','native_async')]:
        status['status']=label;write(AUDIT/'same-session-status.json',status)
        group.set_memory_mode(mode)
        group.load_checkpoint(str(checkpoint),del_local_after_load=False)
        restored=group.observe_state(label+'_restored')
        assert restored==saved,('Native checkpoint state/RNG restore mismatch',restored,saved)
        data.meta_info['replay_label']=label
        start=time.perf_counter();result=group.replay_update(data)
        status.setdefault('completed_stages',[]).append(dict(label=label,seconds_including_diagnostics=time.perf_counter()-start,
            metrics=result.meta_info['metrics'],native_state_and_RNG_restore_exact=True))
        output=group.compute_values(data);torch.save(output,AUDIT/(label+'-post-values.pt'))
        write(AUDIT/'same-session-status.json',status)
    a=torch.load(AUDIT/'native_reference-post-values.pt',weights_only=False,map_location='cpu').batch['values']
    for label in ('native_repeat','native_async'):
        b=torch.load(AUDIT/(label+'-post-values.pt'),weights_only=False,map_location='cpu').batch['values']
        status[label+'_post_values_exact']=torch.equal(a,b)
        status[label+'_post_values_max_abs']=float((a-b).abs().max())
    status['status']='completed_native_capacity_and_replay'
    status['numerical_accepted']=status['native_repeat_post_values_exact'] and status['native_async_post_values_exact']
    status['efficiency_accepted']=False
    status['performance_note']='No performance claim from replay times that include CPU gradient capture. Matched performance work requires numerical acceptance.'
    status['finished']=time.time();write(AUDIT/'same-session-status.json',status)
except BaseException as exc:
    status.update(status='failed',error=repr(exc),finished=time.time());write(AUDIT/'same-session-status.json',status)
    raise
finally:ray.shutdown()
print(json.dumps(status,indent=2))
