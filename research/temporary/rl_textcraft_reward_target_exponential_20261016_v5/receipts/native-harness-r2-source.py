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
spec=importlib.util.spec_from_file_location('run_native_behavior_v4',ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4/source/run_native_behavior_v4.py')
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
                own=kwargs['self_target_mask'];active=future&policy[:,None,:]&(b[:,:,None]!=0)
                estimated=active&~own
                # Native TextCraft has one settlement. These exact expression
                # checks verify representation/transport, not prediction accuracy.
                weighted=torch.where(active,b[:,:,None],0.)
                expected=weighted*-torch.expm1(-torch.where(estimated,d,0.))
                expected=torch.where(own,weighted,expected).sum(1)
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
                target=target&row.get('dt_future_event_mask',torch.ones_like(target)).bool().cpu()
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


def multitarget_requests(tokenizer,capacity=False):
    import copy
    import numpy as np
    from verl import DataProto
    from verl.utils.torch_functional import pad_2d_list_to_length
    artifacts=copy.deepcopy(torch.load(AUDIT/'native-multi-target-artifacts.pt',map_location='cpu',weights_only=False)[:4])
    for item in artifacts:
        if capacity:
            end=max(i for event in item['dt_reward_target_events'] for i in event['target_offsets'])+1
            item['response_ids']=item['response_ids'][:end]
            item['policy_mask']=item['policy_mask'][:end];item['target_mask']=item['target_mask'][:end]
            position=next(i for i,p in enumerate(item['policy_mask']) if p and not item['target_mask'][i])
            extra=32768-len(item['prompt_ids'])-end
            item['response_ids'][position:position]=[item['response_ids'][position]]*extra
            item['policy_mask'][position:position]=[True]*extra
            item['target_mask'][position:position]=[False]*extra
            for event in item['dt_reward_target_events']:
                event['target_offsets']=[i+extra if i>=position else i for i in event['target_offsets']]
                future=event['future_policy_mask'][:end]
                future[position:position]=[True]*extra
                event['future_policy_mask']=future
            assert len(item['prompt_ids'])+len(item['response_ids'])==32768
        item['retained_response_positions']=list(range(len(item['response_ids'])))
    responses=pad_2d_list_to_length([item['response_ids'] for item in artifacts],tokenizer.pad_token_id)
    for item in artifacts:item['retained_response_positions'] += [-1]*(responses.shape[-1]-len(item['response_ids']))
    objects=np.empty(len(artifacts),dtype=object);objects[:]=artifacts
    return DataProto.from_dict(tensors=dict(responses=responses,dt_grpo_multipliers=torch.full((4,),.9)),
        non_tensors=dict(traj_uid=np.array(['native-multi-'+str(i) for i in range(4)],dtype=object),dt_direct_target_artifact=objects))


def verify_native_multitarget(group,data,stage,tokenizer):
    from dt_training_batch import CREDIT_KEYS,compute_direct_target_credit
    returned={}
    class ObservedNativeGroup:
        world_size=group.world_size
        def compute_dt_token_advantages(self,requests):
            requests.meta_info['behavior_stage']=stage
            result=group.verify_credit(requests)
            for i,identity in enumerate(requests.non_tensor_batch['traj_uid']):
                returned[str(identity)]={key:result.batch[key][i].cpu().clone() for key in CREDIT_KEYS}
            return result
    output=compute_direct_target_credit(data,ObservedNativeGroup(),eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
    for i,item in enumerate(data.non_tensor_batch['dt_direct_target_artifact']):
        uid=str(data.non_tensor_batch['traj_uid'][i])
        for key in CREDIT_KEYS:
            expected=torch.zeros_like(output.batch[key][i])
            for event in item['dt_reward_target_events']:
                expected += returned[str((uid,'int',event['event_id']))][key]
            assert torch.equal(output.batch[key][i],expected)
    write(AUDIT/(stage+'-driver.json'),dict(status='completed',trajectories=4,target_events=8,
        component_credits=[-.1,1.],credit_source='User-supplied multi-target interface example; not extra TextCraft rewards',
        native_per_event_QVA_sum_exact=True,source_u_normalization=False,
        original_DP_pad_balance_collect_scatter=True,optimizer_updates=0,
        input_sha256=hashlib.sha256(data.batch['responses'].contiguous().numpy().tobytes()).hexdigest(),
        capacity_scope='Exact 32768 artificial storage fixture, not natural task quality' if stage.startswith('capacity') else 'Two original native parser payloads, distinct target effects'))


def main():
    from transformers import AutoTokenizer
    launch=json.loads((AUDIT/'prepared-launch.json').read_bytes())
    config=OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    for key,value in launch['options'].items():OmegaConf.update(config,key.lstrip('+'),value,force_add=True)
    config.actor_rollout_ref.actor.optim.total_training_steps=330
    env={key:value for key,value in os.environ.items() if key.startswith(('DT_','VERL_','MACA_')) or key in
         ('PYTHONPATH','LD_LIBRARY_PATH','PATH','VENV_PYTHON','MODEL_PATH','HF_HUB_OFFLINE','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR')}
    env.pop('CUDA_VISIBLE_DEVICES',None)
    env['PYTHONPATH'] += ':'+str(ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4/source')
    state=dict(status='initializing_native_fresh_actor',started=time.time(),physical_gpus=[2,3],
               optimizer_updates=0,rollout_generation_calls=0,scope=__doc__)
    write(AUDIT/'native-behavior-status.json',state)
    tokenizer=AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'],local_files_only=True)
    small=base.requests(tokenizer,False)
    # Use actual old v3 official group multipliers from its passive owner
    # artifact, rather than an arbitrary reward-scale test.
    records=sorted((ROOT/'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3/credit-records').glob('*/joint-*.pt'))
    for record in records:
        values=torch.load(record,map_location='cpu',weights_only=False)['rows']
        if len(values)>=4:
            coefficients=torch.tensor([float(row['reward']) for row in values[:4]])
            break
    else:raise FileNotFoundError('Recorded native official-GRPO coefficients unavailable')
    small.batch['dt_direct_reward']=coefficients.repeat(2)
    state['coefficient_artifact']=dict(path=str(record),SHA256=hashlib.sha256(record.read_bytes()).hexdigest(),
                                       coefficient_kind='Recorded native official GRPO group multipliers',values=coefficients.tolist())
    write(AUDIT/'native-behavior-status.json',state)
    ray.init(address='local',num_cpus=8,num_gpus=2,include_dashboard=False,_temp_dir='/tmp/ray-tc-exponential-v5',
             runtime_env=dict(env_vars=env,worker_process_setup_hook='metax_fsdp_event_wait.worker_setup'))
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
                            RayClassWithInitArgs(BehaviorWorker,config.actor_rollout_ref,'actor'))
        group.init_model()
        state['status']='native-small';write(AUDIT/'native-behavior-status.json',state)
        group.verify_credit(small)
        for capacity in (False,True):
            stage='capacity-multi-32768' if capacity else 'native-multi-target'
            state['status']=stage;write(AUDIT/'native-behavior-status.json',state)
            verify_native_multitarget(group,multitarget_requests(tokenizer,capacity),stage,tokenizer)
        state.update(status='completed',finished=time.time());write(AUDIT/'native-behavior-status.json',state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));write(AUDIT/'native-behavior-status.json',state);raise
    finally:ray.shutdown()


if __name__=='__main__':main()
