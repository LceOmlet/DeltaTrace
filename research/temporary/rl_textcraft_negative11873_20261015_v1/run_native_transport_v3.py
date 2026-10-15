"""Bounded DT source-token transport trace with original VERL checkpoint25.

Actor-only native initialization and native checkpoint restore, followed by
one saved B4 DT call per rank with passive layer30 component capture. No policy update, generation, evaluation,
new checkpoint, method change, or formal-worker intervention.
This is a new fixed-weight experiment, never an exact iteration24 replay.
"""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import psutil,ray,torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch,register
from verl.single_controller.ray import RayClassWithInitArgs,RayResourcePool,RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
DEST=ROOT/'receipts/textcraft-neg11873-rootcause-20261015-v1/layer30-mechanism-v3'
RUN=ROOT/'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3'
CHECKPOINT=DEST.parent/'retained-checkpoint-25/actor'
RECORD=RUN/'credit-records/rank1-pid3153050/joint-1792046604440421161.pt'
FOCUS_UID='52808f60-1f4b-4005-98f5-0c2179376ea3'
state=dict(scope=__doc__,phase='prepared',started_unix=time.time(),
           checkpoint=str(CHECKPOINT),checkpoint_step=25,original_incident_step=24,
           original_incident_theta23_available=False,optimizer_updates=0,scheduler_updates=0,
           rollout_calls=0,evaluation_calls=0,formal_training_changes=0)

def phase(name,**values):
    state.update(phase=name,unix=time.time(),**values)
    (DEST/'native-transport-v3-status.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(dict(phase=name,unix=state['unix'],**values)),flush=True)

def native_state(worker):
    h=hashlib.sha256()
    for name,p in worker.actor_module_fsdp.named_parameters():
        if not p.requires_grad:continue
        x=p.to_local() if hasattr(p,'to_local') else p
        h.update(name.encode());h.update(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    steps=[]
    for item in worker.actor.actor_optimizer.state.values():
        if 'step' in item:
            value=item['step'];value=value.to_local() if hasattr(value,'to_local') else value
            steps.append(float(value))
    return dict(trainable_SHA256=h.hexdigest(),scheduler=worker.actor_lr_scheduler.last_epoch,
                optimizer_steps=sorted(set(steps)),optimizer_state_count=len(worker.actor.actor_optimizer.state))

@ray.remote
class SourceTransportWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def trace_saved_DT(self,data):
        import inspect,sys
        from deltatrace_rollout import DeltaTraceRolloutProducer
        from observe_layer30_transport_v3 import observe_transport
        before=native_state(self)
        assert before['scheduler']==25 and before['optimizer_steps']==[100.0]
        if not hasattr(self,'_deltatrace_producer'):
            self._deltatrace_producer=DeltaTraceRolloutProducer(
                self.actor_module_fsdp,eos_token_id=data.meta_info['eos_token_id'],
                pad_token_id=data.meta_info['pad_token_id'],
                invalid_action_penalty_coef=self.config.actor.invalid_action_penalty_coef)
        producer=self._deltatrace_producer
        # Sort exactly as the owning readout does, solely to map its native B4 row.
        rows=sorted(zip(data.non_tensor_batch['traj_uid'].tolist(),
                        data.batch['attention_mask'].sum(-1).tolist()),key=lambda x:x[1])
        focus_row=next(i for i,(uid,_) in enumerate(rows) if uid==FOCUS_UID)
        runner=producer.runner
        flags={k:getattr(runner,k,None) for k in ['defer_diagnostics','reuse_native_prefix',
             'offload_replay_mixer','pin_replay_host','pin_root_host','compact_gdn_captures',
             'fa_coefficient_suffix','gdn_coefficient_suffix','copy_replay_captures']}
        bindings={}
        for name in ['qwen35_dense_finite_runner','qwen35_decoder_finite','qwen35_gdn_finite',
                     'qwen35_answer_finite','finite_fla_gpu','reward_readout','deltatrace_rollout','verl.workers.actor.dp_actor']:
            m=sys.modules[name];path=Path(m.__file__)
            bindings[name]=dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
        assert bindings['qwen35_dense_finite_runner']['SHA256']=='198f78db4abf58cd8461c5bab86d1ce9c325a46e4295c8cc54d83bd9e20e59cd'
        assert bindings['qwen35_gdn_finite']['SHA256']=='3f51f5f7569a4d6b410e7acef89635a50f2e6ecadfa43eb94d1fd4fdd655dfd1'
        assert bindings['verl.workers.actor.dp_actor']['SHA256']=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
        with observe_transport(runner,focus_row,1870) as (metadata,payload):
            output=super().compute_dt_token_advantages(data)
        after=native_state(self);assert before==after
        path=DEST/f'native-transport-v3-rank{self.rank}.pt'
        torch.save(dict(metadata=metadata,payload=payload,focus_row=focus_row,focus_uid=FOCUS_UID,
                        original_report=producer.direct_readout.last_report),path)
        report=dict(rank=self.rank,pid=os.getpid(),birth=psutil.Process().create_time(),
            before=before,after=after,bindings=bindings,runner_flags=flags,
            payload=str(path),bytes=path.stat().st_size,
            attribution_calls=producer.direct_readout.last_report['finite_trace_calls'],
            original_report=producer.direct_readout.last_report,
            production_attribute_observer=None,original_cache_offload_flags_unchanged=True,
            no_policy_update_generation_evaluation=True)
        (DEST/f'native-transport-v3-rank{self.rank}.json').write_text(json.dumps(report,indent=2)+'\n')
        return output

def saved_input(tokenizer):
    record=torch.load(RECORD,map_location='cpu',weights_only=True)
    assert hashlib.sha256(RECORD.read_bytes()).hexdigest()=='017784bbf8427966e6f1e56ee9f99f8e24786cef32705f4f203f27602be93062'
    rows=record['rows'];assert len(rows)==4
    focus=next(r for r in rows if r['traj_uid']==FOCUS_UID)
    positions=focus['suffix_positions'];selected_ids=focus['input_ids'][focus['prompt_length']:]
    same=(selected_ids==271)&focus['prior_source_mask'][positions]
    same_positions=positions[same]
    saved_shape=dict(trajectory_uid=FOCUS_UID,token_id=271,focus_original_position=1870,
        same_token_source_positions=same_positions.tolist(),
        same_token_signed_DT_values=focus['source_log_ratios'][same_positions].tolist(),
        source_record=str(RECORD),new_model_calls=0,
        interpretation='Same original token and EOS reference share an input embedding displacement; position-specific finite transport remains to be measured, not inferred from token identity.')
    (DEST/'saved-same-token-position-scope-v1.json').write_text(json.dumps(saved_shape,indent=2)+'\n')
    width=max(r['policy_mask'].numel() for r in rows)
    prompt_width=max(r['prompt_length'] for r in rows)
    fields={k:[] for k in ['input_ids','attention_mask','responses','policy_mask','target_mask','dt_direct_reward']}
    for r in rows:
        ids=torch.full((prompt_width+width,),tokenizer.pad_token_id,dtype=torch.long)
        attention=torch.zeros_like(ids);plen=r['prompt_length'];valid=r['input_ids']
        ids[prompt_width-plen:prompt_width]=valid[:plen];attention[prompt_width-plen:prompt_width]=1
        suffix=r['suffix_positions'];ids[prompt_width+suffix]=valid[plen:];attention[prompt_width+suffix]=1
        assert torch.equal(ids[attention.bool()],valid)
        fields['input_ids'].append(ids);fields['attention_mask'].append(attention)
        fields['responses'].append(ids[prompt_width:]);fields['policy_mask'].append(r['policy_mask'])
        fields['target_mask'].append(r['target_mask']);fields['dt_direct_reward'].append(torch.tensor(r['reward'],dtype=torch.float32))
    one=DataProto.from_dict(tensors={k:torch.stack(v) for k,v in fields.items()},
        non_tensors={'traj_uid':np.array([r['traj_uid'] for r in rows],dtype=object)},
        meta_info={'eos_token_id':tokenizer.eos_token_id,'pad_token_id':tokenizer.pad_token_id,
                   'dt_target_semantics':'native_joint_action_target'})
    # Two native DP ranks consume the identical original B4; these are not eight independent samples.
    return DataProto.concat([one,one])

def main():
    assert not (DEST/'native-transport-v3-rank0.json').exists(),'Do not repeat a completed diagnosis.'
    launch=json.loads((RUN/'launch.json').read_bytes())
    cfg=OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    for k,v in launch['options'].items():OmegaConf.update(cfg,k.lstrip('+'),v,force_add=True)
    cfg.actor_rollout_ref.actor.optim.total_training_steps=330
    env={k:v for k,v in launch['effective_environment'].items() if k not in ['CUDA_VISIBLE_DEVICES','RAY_ADDRESS']}
    env['PYTHONPATH']=str(DEST/'source')+os.pathsep+env['PYTHONPATH']
    env['DT_FSDP_EVENT_WAIT_RECEIPTS']=str(DEST/'native-transport-v3-eventwait')
    tokenizer=AutoTokenizer.from_pretrained(launch['options']['actor_rollout_ref.model.path'],local_files_only=True)
    batch=saved_input(tokenizer)
    (DEST/'native-transport-v3-config.yaml').write_text(OmegaConf.to_yaml(cfg))
    phase('initializing_original_actor_only_GPU23',physical_gpus=[2,3])
    ray.init(address='local',num_cpus=8,num_gpus=2,include_dashboard=False,
        _temp_dir='/tmp/ray-tc-neg11873-v3',runtime_env={'env_vars':env,
            'worker_process_setup_hook':'metax_fsdp_event_wait.worker_setup'})
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
                            RayClassWithInitArgs(SourceTransportWorker,cfg.actor_rollout_ref,'actor'))
        group.init_model();phase('native_checkpoint25_load')
        group.load_checkpoint(local_path=str(CHECKPOINT),del_local_after_load=False)
        phase('one_original_B4_DT_with_passive_existing_contractions')
        group.trace_saved_DT(batch)
        phase('completed_fixed_checkpoint25_transport_not_historical_replay',status='completed')
    except BaseException as exc:
        phase('failed',error=repr(exc));raise
    finally:
        ray.shutdown()

if __name__=='__main__':main()
