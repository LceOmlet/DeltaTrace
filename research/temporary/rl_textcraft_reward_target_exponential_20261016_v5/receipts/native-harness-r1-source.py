"""Native B4/rank credit transport and exact-32768 capacity, zero optimizer.

This verifies actual DT vectors reach the original exponential Q/V primitive.
No single-update quality or deletion-prediction experiment is performed.
"""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time

import psutil
import ray
import torch
from omegaconf import OmegaConf
from verl.single_controller.base.decorator import Dispatch,register
from verl.single_controller.ray import RayClassWithInitArgs,RayResourcePool,RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5'
spec=importlib.util.spec_from_file_location('native_verified_harness_v4',ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4/source/run_native_behavior_v4.py')
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
base.AUDIT=AUDIT


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


@ray.remote
class BehaviorWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def verify_credit(self,data):
        from deltatrace_rollout import DeltaTraceRolloutProducer
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu,offload_fsdp_model_to_cpu
        from agent_system.multi_turn_rollout.utils import to_list_of_dict
        import reward_readout
        stage=data.meta_info['behavior_stage'];path=AUDIT/f'{stage}-rank{self.rank}.json'
        state=dict(stage=stage,status='starting',rank=self.rank,pid=os.getpid(),birth=psutil.Process().create_time(),
                   before=base.actor_state(self),optimizer_updates=0,generated_rollouts=0,quality_evaluations=0)
        write(path,state)
        if self._is_offload_param:load_fsdp_model_to_gpu(self.actor_module_fsdp)
        try:
            if not hasattr(self,'_deltatrace_producer'):
                self._deltatrace_producer=DeltaTraceRolloutProducer(self.actor_module_fsdp,
                    eos_token_id=data.meta_info['eos_token_id'],pad_token_id=data.meta_info['pad_token_id'],invalid_action_penalty_coef=0.)
            producer=self._deltatrace_producer;runner=producer.runner
            native_attribute=runner.attribute;native_credit=reward_readout.reward_event_token_credit
            def observe_attribute(pair,mask,selection,*args,**kwargs):
                assert not hasattr(selection,'credit_seed_weights')
                signed,detail=native_attribute(pair,mask,selection,*args,**kwargs)
                state.setdefault('finite_calls',[]).append(dict(shape=list(pair.shape),
                    target_offsets=selection.offsets,original_unscaled_seed=True,
                    signed_min=float(signed.min()),signed_max=float(signed.max()),
                    seconds=detail['complete_attribution_seconds_with_diagnostics']))
                write(path,state)
                return signed,detail
            def observe_credit(d,b,future,policy,**kwargs):
                assert not kwargs.get('linear_attribution',False)
                result=native_credit(d,b,future,policy,**kwargs)
                own=kwargs['self_target_mask'];estimated=policy[:,None,:]&~own
                # Native TextCraft has one settlement. These exact expression
                # checks verify representation/transport, not prediction accuracy.
                expected=b[:,:,None]*-torch.expm1(-torch.where(estimated,d,0.))
                expected=torch.where(own,b[:,:,None],expected)
                expected=torch.where(policy[:,None,:],expected,0.).sum(1)
                assert torch.equal(result.advantages,expected)
                state.setdefault('credit_calls',[]).append(dict(
                    per_event_exponent_before_sum=True,source_normalization=False,
                    coefficient=b.flatten().tolist(),active_d_min=float(d[estimated].min()),
                    active_d_max=float(d[estimated].max()),advantages_min=float(result.advantages.min()),
                    advantages_max=float(result.advantages.max()),exact_expression_equality=True))
                return result
            runner.attribute=observe_attribute;reward_readout.reward_event_token_credit=observe_credit
            started=time.perf_counter()
            try:output=ActorRolloutRefWorker.compute_dt_token_advantages(self,data)
            finally:runner.attribute=native_attribute;reward_readout.reward_event_token_credit=native_credit
            for i,row in enumerate(to_list_of_dict(data)):
                policy=row['policy_mask'].bool().cpu();target=row['target_mask'].bool().cpu();b=float(row['dt_direct_reward'])
                for key in ('dt_token_advantages','dt_q_estimates','dt_v_estimates'):
                    assert not bool(output.batch[key][i].cpu()[~policy].any())
                    assert bool(torch.isfinite(output.batch[key]).all())
                assert torch.equal(output.batch['dt_token_advantages'][i].cpu()[target],torch.full((int(target.sum()),),b))
                assert not bool(output.batch['dt_v_estimates'][i].cpu()[target].any())
            state.update(status='completed',seconds=time.perf_counter()-started,
                         report=producer.direct_readout.last_report,after=base.actor_state(self))
            assert state['before']==state['after']
            assert state['report']['credit_transform']=='per_event_exponential_then_credit_sum'
            state['bindings']={}
            for name in ('qwen35_answer_finite','qwen35_dense_finite_runner','qwen35_gdn_finite','counterfactual',
                         'reward_readout','deltatrace_credit','verl.workers.actor.dp_actor',
                         'verl.trainer.ppo.core_algos','verl.utils.experimental.torch_functional'):
                module=sys.modules[name];file=Path(inspect.getsourcefile(module))
                state['bindings'][name]=dict(path=str(file),SHA256=hashlib.sha256(file.read_bytes()).hexdigest())
            write(path,state)
            return output
        except BaseException as exc:
            state.update(status='failed',error=repr(exc));write(path,state);raise
        finally:
            if self._is_offload_param:offload_fsdp_model_to_cpu(self.actor_module_fsdp)


def main():
    from transformers import AutoTokenizer
    launch=json.loads((AUDIT/'prepared-launch.json').read_bytes())
    config=OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    for key,value in launch['options'].items():OmegaConf.update(config,key.lstrip('+'),value,force_add=True)
    config.actor_rollout_ref.actor.optim.total_training_steps=330
    env={key:value for key,value in os.environ.items() if key.startswith(('DT_','VERL_','MACA_')) or key in
         ('PYTHONPATH','LD_LIBRARY_PATH','PATH','VENV_PYTHON','MODEL_PATH','HF_HUB_OFFLINE','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR')}
    env.pop('CUDA_VISIBLE_DEVICES',None)
    state=dict(status='initializing_native_fresh_actor',started=time.time(),physical_gpus=[2,3],
               optimizer_updates=0,rollout_generation_calls=0,scope=__doc__)
    write(AUDIT/'native-behavior-status.json',state)
    tokenizer=AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'],local_files_only=True)
    small=base.requests(tokenizer,False);capacity=base.requests(tokenizer,True)
    # Use actual old v3 official group multipliers from its passive owner
    # artifact, rather than an arbitrary reward-scale test.
    records=sorted((ROOT/'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3/credit-records').glob('*/joint-*.pt'))
    for record in records:
        values=torch.load(record,map_location='cpu',weights_only=False)['rows']
        if len(values)>=4:
            coefficients=torch.tensor([float(row['reward']) for row in values[:4]])
            break
    else:raise FileNotFoundError('Recorded native official-GRPO coefficients unavailable')
    for data in (small,capacity):data.batch['dt_direct_reward']=coefficients.repeat(2)
    state['coefficient_artifact']=dict(path=str(record),SHA256=hashlib.sha256(record.read_bytes()).hexdigest(),
                                       coefficient_kind='Recorded native official GRPO group multipliers',values=coefficients.tolist())
    write(AUDIT/'native-behavior-status.json',state)
    ray.init(address='local',num_cpus=8,num_gpus=2,include_dashboard=False,_temp_dir='/tmp/ray-tc-exponential-v5',
             runtime_env=dict(env_vars=env,worker_process_setup_hook='metax_fsdp_event_wait.worker_setup'))
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
                            RayClassWithInitArgs(BehaviorWorker,config.actor_rollout_ref,'actor'))
        group.init_model()
        for data in (small,capacity):
            state['status']=data.meta_info['behavior_stage'];write(AUDIT/'native-behavior-status.json',state)
            group.verify_credit(data)
        state.update(status='completed',finished=time.time());write(AUDIT/'native-behavior-status.json',state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));write(AUDIT/'native-behavior-status.json',state);raise
    finally:ray.shutdown()


if __name__=='__main__':main()
