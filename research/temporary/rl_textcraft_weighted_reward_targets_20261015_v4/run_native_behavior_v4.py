"""Native two-rank v4 behavior and B4/rank 32768-token DT capacity.

No optimizer update, checkpoint, rollout generation or quality evaluation.
The small inputs are exact artifacts of the original TextCraft rollout/service
CPU fixture. The long inputs repeat an existing policy token before the real
settlement action solely to test capacity; they are not natural-task quality.
"""
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np
import psutil
import ray
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch,register
from verl.single_controller.ray import RayClassWithInitArgs,RayResourcePool,RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4'


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def actor_state(worker):
    digest=hashlib.sha256()
    for name,param in worker.actor_module_fsdp.named_parameters():
        if not param.requires_grad:continue
        value=param.to_local() if hasattr(param,'to_local') else param
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    steps=[]
    for item in worker.actor.actor_optimizer.state.values():
        if 'step' in item:
            value=item['step'];value=value.to_local() if hasattr(value,'to_local') else value
            steps.append(float(value))
    return dict(trainable_SHA256=digest.hexdigest(),scheduler=worker.actor_lr_scheduler.last_epoch,
                optimizer_steps=sorted(set(steps)),optimizer_states=len(worker.actor.actor_optimizer.state))


@ray.remote
class BehaviorWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def verify_credit(self,data):
        from deltatrace_rollout import DeltaTraceRolloutProducer
        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu,offload_fsdp_model_to_cpu
        from agent_system.multi_turn_rollout.utils import to_list_of_dict
        stage=data.meta_info['behavior_stage'];path=AUDIT/f'{stage}-rank{self.rank}.json'
        state=dict(stage=stage,status='starting',rank=self.rank,pid=os.getpid(),
                   birth=psutil.Process().create_time(),before=actor_state(self),
                   optimizer_updates=0,generated_rollouts=0,quality_evaluations=0)
        write(path,state)
        if self._is_offload_param:load_fsdp_model_to_gpu(self.actor_module_fsdp)
        try:
            if not hasattr(self,'_deltatrace_producer'):
                self._deltatrace_producer=DeltaTraceRolloutProducer(self.actor_module_fsdp,
                    eos_token_id=data.meta_info['eos_token_id'],pad_token_id=data.meta_info['pad_token_id'],
                    invalid_action_penalty_coef=0.)
            producer=self._deltatrace_producer;runner=producer.runner
            native_attribute=runner.attribute
            def checked_attribute(pair,mask,selection,*args,**kwargs):
                baseline=None
                if stage=='native-small':
                    original_selection=copy.copy(selection)
                    original_selection.credit_groups=None;original_selection.credit_seed_weights=None
                    baseline=native_attribute(pair,mask,original_selection,*args,**kwargs)
                actual=native_attribute(pair,mask,selection,*args,**kwargs)
                signed,detail=actual
                record=dict(target_span_deltas=detail['target_span_deltas'],
                    target_span_credits=detail['target_span_credits'],root_effect=detail['root_effect'],
                    signed_sum=float(signed.sum()),shape=list(pair.shape),
                    seconds=detail['complete_attribution_seconds_with_diagnostics'])
                if baseline is not None:
                    original,old_detail=baseline
                    slopes=[]
                    for sample in range(selection.batch):
                        group=detail['target_span_samples'].index(sample)
                        slopes.append(detail['target_span_credits'][group]/detail['target_span_deltas'][group])
                    expected=original.double()*torch.tensor(slopes,dtype=torch.float64)[:,None]
                    error=signed.double()-expected
                    record.update(original_root=old_detail['root_effect'],
                        weighted_vs_original_scaled_relative_L2=float(error.norm()/expected.norm()),
                        weighted_vs_original_scaled_max_abs=float(error.abs().max()),
                        comparison_scope='Same native finite operator at two actual executions; raw numerical differences, no borrowed tolerance.')
                state.setdefault('finite_calls',[]).append(record);write(path,state)
                return actual
            runner.attribute=checked_attribute
            start=time.perf_counter()
            try:output=ActorRolloutRefWorker.compute_dt_token_advantages(self,data)
            finally:runner.attribute=native_attribute
            rows=to_list_of_dict(data)
            for i,row in enumerate(rows):
                policy=row['policy_mask'].bool().cpu();target=row['target_mask'].bool().cpu()
                coefficient=float(row['dt_direct_reward'])
                assert all(not bool(output.batch[key][i].cpu()[~policy].any())
                           for key in ('dt_token_advantages','dt_q_estimates','dt_v_estimates'))
                assert torch.equal(output.batch['dt_token_advantages'][i].cpu()[target],
                                   torch.full((int(target.sum()),),coefficient))
                assert not bool(output.batch['dt_v_estimates'][i].cpu()[target].any())
                assert all(bool(torch.isfinite(output.batch[key]).all())
                           for key in ('dt_token_advantages','dt_q_estimates','dt_v_estimates'))
            state.update(status='completed',seconds=time.perf_counter()-start,
                         report=producer.direct_readout.last_report,after=actor_state(self))
            assert state['before']==state['after']
            import sys
            state['bindings']={}
            for name in ('qwen35_answer_finite','qwen35_dense_finite_runner','qwen35_gdn_finite',
                         'counterfactual','reward_readout','deltatrace_credit',
                         'verl.workers.actor.dp_actor','verl.trainer.ppo.core_algos'):
                module=sys.modules[name];file=Path(inspect.getsourcefile(module))
                state['bindings'][name]=dict(path=str(file),SHA256=hashlib.sha256(file.read_bytes()).hexdigest())
            write(path,state)
            return output
        except BaseException as exc:
            state.update(status='failed',error=repr(exc));write(path,state);raise
        finally:
            if self._is_offload_param:offload_fsdp_model_to_cpu(self.actor_module_fsdp)


def requests(tokenizer,capacity=False):
    from verl.utils.torch_functional import postprocess_data,pad_2d_list_to_length
    from verl.utils.model import compute_position_id_with_mask
    artifacts=torch.load(AUDIT/'native-zero-settlement-artifacts.pt',map_location='cpu',weights_only=False)[:4]
    fields={name:[] for name in ('prompt','response','policy','target','spans')}
    for original in artifacts:
        artifact=copy.deepcopy(original)
        if capacity:
            last=max(artifact['reward_target_spans'][0]['target_offsets'])+1
            # Original readout already omits observations to the right of its
            # causal target. This fixture has exactly 32768 required positions.
            response=artifact['response_ids'][:last]
            policy=artifact['policy_mask'][:last];target=artifact['target_mask'][:last]
            position=next(i for i,p in enumerate(policy) if p and not target[i])
            extra=32768-len(artifact['prompt_ids'])-len(response)
            token=response[position]
            artifact['response_ids']=response[:position]+[token]*extra+response[position:]
            artifact['policy_mask']=policy[:position]+[True]*extra+policy[position:]
            artifact['target_mask']=target[:position]+[False]*extra+target[position:]
            artifact['reward_target_spans'][0]['target_offsets']=[i+extra for i in artifact['reward_target_spans'][0]['target_offsets']]
            assert len(artifact['prompt_ids'])+len(artifact['response_ids'])==32768
        for key,name in (('prompt','prompt_ids'),('response','response_ids'),('policy','policy_mask'),('target','target_mask')):
            fields[key].append(artifact[name])
        fields['spans'].append([(span['target_offsets'],span['credit_fraction']) for span in artifact['reward_target_spans']])
    prompt_width=max(map(len,fields['prompt']))
    prompt,pmask=zip(*(postprocess_data(torch.tensor([ids]),torch.ones(1,len(ids),dtype=torch.long),
                                      prompt_width,tokenizer.pad_token_id,left_pad=True,truncation='error') for ids in fields['prompt']))
    prompt=torch.cat(prompt);pmask=torch.cat(pmask)
    response=pad_2d_list_to_length(fields['response'],tokenizer.pad_token_id)
    amask=torch.arange(response.shape[1])[None,:]<torch.tensor(list(map(len,fields['response'])))[:,None]
    attention=torch.cat((pmask,amask.long()),-1)
    spans=np.empty(4,dtype=object);spans[:]=fields['spans']
    data=DataProto.from_dict(tensors=dict(input_ids=torch.cat((prompt,response),-1),prompts=prompt,
        responses=response,attention_mask=attention,position_ids=compute_position_id_with_mask(attention),
        policy_mask=pad_2d_list_to_length(fields['policy'],0).bool(),
        target_mask=pad_2d_list_to_length(fields['target'],0).bool(),
        dt_direct_reward=torch.tensor([1.,-.5,.5,-1.])),
        non_tensors=dict(traj_uid=np.array(['native-interface-'+str(i) for i in range(4)],dtype=object),
                         dt_target_span_credits=spans),
        meta_info=dict(eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,
                       dt_target_semantics='native_joint_action_target',
                       behavior_stage='capacity-32768' if capacity else 'native-small'))
    # Both original DP ranks consume four rows: B4/rank, physical dual-card B8.
    return DataProto.concat([data,data])


def main():
    launch=json.loads((AUDIT/'prepared-launch.json').read_text())
    config=OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    for key,value in launch['options'].items():OmegaConf.update(config,key.lstrip('+'),value,force_add=True)
    config.actor_rollout_ref.actor.optim.total_training_steps=330
    env={key:value for key,value in os.environ.items() if key.startswith(('DT_','VERL_','MACA_')) or key in ('PYTHONPATH','LD_LIBRARY_PATH','PATH','VENV_PYTHON','MODEL_PATH','HF_HUB_OFFLINE','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR')}
    env.pop('CUDA_VISIBLE_DEVICES',None)
    state=dict(status='initializing_native_fresh_actor',started=time.time(),physical_gpus=[2,3],
               optimizer_updates=0,rollout_generation_calls=0,scope=__doc__)
    write(AUDIT/'native-behavior-status.json',state)
    tokenizer=AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'],local_files_only=True)
    small=requests(tokenizer,False)
    capacity_input=requests(tokenizer,True)
    ray.init(address='local',num_cpus=8,num_gpus=2,include_dashboard=False,
             _temp_dir='/tmp/ray-tc-weighted-v4',runtime_env=dict(env_vars=env,
             worker_process_setup_hook='metax_fsdp_event_wait.worker_setup'))
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
                            RayClassWithInitArgs(BehaviorWorker,config.actor_rollout_ref,'actor'))
        group.init_model()
        for data in (small,capacity_input):
            state['status']=data.meta_info['behavior_stage'];write(AUDIT/'native-behavior-status.json',state)
            group.verify_credit(data)
        state.update(status='completed',finished=time.time());write(AUDIT/'native-behavior-status.json',state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));write(AUDIT/'native-behavior-status.json',state);raise
    finally:ray.shutdown()


if __name__=='__main__':main()
