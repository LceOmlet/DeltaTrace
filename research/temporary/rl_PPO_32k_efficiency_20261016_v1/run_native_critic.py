"""Same-state native critic memory validation; no replacement PPO, GAE, loss or optimizer.

The fixed SQL first-turn token IDs are a capacity input, extended only in the
ordinary prefix. They are not a new rollout, reward test or quality result.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION='PPO-32k-native-efficiency-20261016-v1'
AUDIT=ROOT/'receipts'/VERSION
BASE=ROOT/'candidates/textcraft-dt-reward-target-exponential-20261016-v5/verl-owner'
CANDIDATE=ROOT/'candidates'/VERSION/'verl-owner'
NATIVE_INPUT=ROOT/'receipts/sql-native-format-observation-20261016-v1/native-turns/native-turns-1792140016959744098.json'

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def source(path):
    path=Path(path)
    return dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest())

def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--case',required=True,choices=['baseline-fit','baseline-repeat-fit','candidate-fit','candidate-32k','reference-32k'])
    p.add_argument('--native',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    return p.parse_args()

args=arguments()
if not args.native:
    import re
    import subprocess
    prepared=json.loads((ROOT/'candidates/appworld-native-ppo-speed-20261016-v1/prepared.json').read_bytes())
    env=dict(os.environ);env.update(prepared['effective_environment'])
    owner=CANDIDATE if args.case.startswith('candidate') else BASE
    env['VERL_ROOT']=str(owner);env['PYTHONPATH']=str(owner)+':'+env['PYTHONPATH']
    env['CUDA_VISIBLE_DEVICES']='' if args.prepare_only else '2,3'
    env.pop('MACA_VISIBLE_DEVICES',None)
    if not args.prepare_only:
        smi=subprocess.check_output(['mx-smi'],text=True)
        assert not any(re.match(r'\|\s*(?:2|3)\s+\d+\s+\S+',line) for line in smi.splitlines()),'Physical2/3 occupied'
    cmd=[env['VENV_PYTHON'],__file__,'--native','--case',args.case]
    if args.prepare_only:cmd.append('--prepare-only')
    raise SystemExit(subprocess.call(cmd,env=env))

import numpy as np
import psutil
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage,compute_response_mask,AdvantageEstimator

CASE=AUDIT/args.case
CASE.mkdir(exist_ok=True)
length=32768 if args.case.endswith('32k') else 8192
prepared=json.loads((ROOT/'candidates/appworld-native-ppo-speed-20261016-v1/prepared.json').read_bytes())
config=OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
for key,value in prepared['options'].items():OmegaConf.update(config,key.lstrip('+'),value,force_add=True)
config.critic.optim.total_training_steps=2  # Native trainer's diagnostic budget; LR/warmup unchanged.
config.critic.model.enable_activation_offload=args.case.startswith('candidate')

def fixture():
    native=json.loads(NATIVE_INPUT.read_bytes())
    turn=native['turns'][0][0]
    response=turn['output_ids'][:turn['end']-turn['start']]
    prompt=native['prompt_ids'][0]
    assert len(prompt)==turn['start'] and turn['reward']==0
    prefix_length=length-len(response)
    assert prefix_length>=len(prompt)
    # Keep exact native response IDs. Repeating the original ordinary prefix
    # explicitly creates storage load, not a natural SQL conversation.
    prefix=(prompt*((prefix_length+len(prompt)-1)//len(prompt)))[:prefix_length]
    ids=torch.tensor([prefix+response]*8,dtype=torch.long)
    responses=torch.tensor([response]*8,dtype=torch.long)
    mask=torch.ones_like(ids)
    from verl.utils.model import compute_position_id_with_mask
    data=DataProto.from_dict(tensors=dict(input_ids=ids,responses=responses,attention_mask=mask,
        position_ids=compute_position_id_with_mask(mask),token_level_rewards=torch.zeros_like(responses,dtype=torch.float32)))
    data.batch['response_mask']=compute_response_mask(data)
    data.meta_info['global_token_num']=mask.sum(-1).tolist()
    return data,dict(native_source=source(NATIVE_INPUT),trajectory=0,turn=0,native_reward=0,
        native_prompt_tokens=len(prompt),native_response_tokens=len(response),exact_context=length,
        batch=8,microbatch_per_rank=4,artificial_prefix_extension=True,
        claim='Capacity/numerical fixture only, not a task-quality or full-iteration result',
        input_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest())

data,fixture_record=fixture()
receipt=dict(case=args.case,scope=__doc__,status='CPU_prepared_only',fixture=fixture_record,
    critic_config=OmegaConf.to_container(config.critic,resolve=True),owner=str(Path(os.environ['VERL_ROOT'])),
    source_commit=os.environ.get('PPO_32K_SOURCE_COMMIT'),source_bindings={
        name:source(Path(os.environ['VERL_ROOT'])/name) for name in
        ('verl/workers/fsdp_workers.py','verl/workers/critic/dp_critic.py','verl/trainer/ppo/core_algos.py',
         'verl/utils/activation_offload.py','verl/utils/fsdp_utils.py')},
    numerical_criterion='Corresponding pinned VERL activation-offload replay test: exact comparison (atol=0, rtol=0). No tolerance relaxation on failure.',
    checkpoint_mode='Original non-reentrant checkpoint; original forward BF16 / FP32 masters and reduce',
    reference_context='torch.autograd.graph.save_on_cpu(pin_memory=True)' if args.case=='reference-32k' else None,
    formal_mutations=0,DT_calls=0,rollout_calls=0)
write(CASE/'prepared.json',receipt)
if args.prepare_only:
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(case=args.case,status=receipt['status'],fixture=fixture_record,critic_config=receipt['critic_config']),indent=2))
    raise SystemExit(0)

import ray
from verl.single_controller.base.decorator import Dispatch,register
from verl.single_controller.ray import RayClassWithInitArgs,RayResourcePool,RayWorkerGroup
from verl.workers.fsdp_workers import CriticWorker

def fingerprint(tensor):
    if hasattr(tensor,'to_local'):tensor=tensor.to_local()
    value=tensor.detach().cpu().contiguous()
    return dict(shape=list(value.shape),dtype=str(value.dtype),SHA256=hashlib.sha256(memoryview(value.reshape(-1).view(torch.uint8).numpy())).hexdigest())

def native_state(worker,with_optimizer=True):
    result=dict(parameters={name:fingerprint(p) for name,p in worker.critic_module.named_parameters()},
        scheduler=worker.critic_lr_scheduler.state_dict())
    if with_optimizer:
        state=worker.critic_optimizer.state_dict()
        result['optimizer']={str(k):{name:fingerprint(v) if torch.is_tensor(v) else v for name,v in item.items()}
                             for k,item in state['state'].items()}
        result['param_groups']=state['param_groups']
    return result

@ray.remote
class ObservedCritic(CriticWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize_observed(self):
        import random
        random.seed(20261016);np.random.seed(20261016);torch.manual_seed(20261016)
        super().init_model()
        record=dict(rank=self.rank,pid=os.getpid(),birth=psutil.Process().create_time(),
            CVD=os.environ.get('CUDA_VISIBLE_DEVICES'),initial=native_state(self),
            actual_sources={name:source(sys.modules[name].__file__) for name in
                ('verl.workers.fsdp_workers','verl.workers.critic.dp_critic','verl.trainer.ppo.core_algos','verl.utils.activation_offload')})
        write(CASE/f'initial-rank{self.rank}.json',record)
        return {k:v for k,v in record.items() if k!='initial'}

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def numerical_update(self,inputs):
        original=self.critic._optimizer_step
        captures=[]
        def observe_native_step():
            before={name:fingerprint(p.grad) for name,p in self.critic_module.named_parameters() if p.grad is not None}
            result=original()
            captures.append(dict(preclip_gradient=before,native_grad_norm=float(result.detach().cpu()),
                                 after_native_step=native_state(self)))
            return result
        self.critic._optimizer_step=observe_native_step
        try:
            ctx=torch.autograd.graph.save_on_cpu(pin_memory=True) if args.case=='reference-32k' else contextlib.nullcontext()
            with ctx:output=super().update_critic(inputs)
        finally:self.critic._optimizer_step=original
        write(CASE/f'numerical-rank{self.rank}.json',dict(captures=captures,final=native_state(self),
            native_metrics=output.meta_info['metrics'],optimizer_calls=len(captures)))
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def observed_final(self):
        value=native_state(self)
        write(CASE/f'final-rank{self.rank}.json',value)
        return dict(rank=self.rank,scheduler=value['scheduler'])

env=dict(os.environ);env.pop('CUDA_VISIBLE_DEVICES',None)
state=dict(case=args.case,status='initializing',pid=os.getpid(),birth=psutil.Process().create_time(),
           physical_gpus=[2,3],started=time.time(),scope=__doc__)
write(CASE/'status.json',state)
ray.init(address='local',num_cpus=8,num_gpus=2,include_dashboard=False,
    _temp_dir='/tmp/ray-PPO32k-'+args.case,
    runtime_env=dict(env_vars=env,worker_process_setup_hook='metax_fsdp_event_wait.worker_setup'))
try:
    group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
                        RayClassWithInitArgs(ObservedCritic,config.critic))
    state['workers']=group.initialize_observed()
    state['status']='native_values';write(CASE/'status.json',state)
    started=time.perf_counter();values=group.compute_values(data);state['values_seconds']=time.perf_counter()-started
    torch.save(values,CASE/'native-values.pt')
    reference=AUDIT/('reference-32k' if length==32768 else 'baseline-fit')/'native-input-with-GAE.pt'
    if reference.exists():
        native_input=torch.load(reference,map_location='cpu',weights_only=False)
        torch.testing.assert_close(data.batch['input_ids'],native_input.batch['input_ids'],rtol=0,atol=0)
        # Identical native old values and GAE targets for memory-path comparison.
        data=native_input
        state['fixed_GAE_source']=source(reference)
    else:
        data=data.union(values)
        data=compute_advantage(data,AdvantageEstimator.GAE,gamma=config.algorithm.gamma,lam=config.algorithm.lam)
        torch.save(data,CASE/'native-input-with-GAE.pt')
    state['status']='native_numerical_update';write(CASE/'status.json',state)
    result=group.numerical_update(data)
    state['metrics']=result.meta_info['metrics']
    if length<32768 and args.case!='baseline-repeat-fit':
        # Numerical hashing is disabled for these warmed native timings.
        timings=[]
        for i in range(3):
            state['status']=f'native_timing_{i+1}';write(CASE/'status.json',state)
            start=time.perf_counter();out=group.update_critic(data)
            timings.append(dict(seconds=time.perf_counter()-start,metrics=out.meta_info['metrics']))
        state['native_warm_update_timings']=timings
    state['final']=group.observed_final()
    state.update(status='completed',finished=time.time())
    write(CASE/'status.json',state)
except BaseException as exc:
    state.update(status='failed',error=repr(exc),finished=time.time())
    write(CASE/'status.json',state)
    raise
finally:
    ray.shutdown()
print(json.dumps(state,indent=2))
