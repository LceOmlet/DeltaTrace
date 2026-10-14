"""Conditional native update -> fixed-seed gain probe; not a quality-score test.

Reuse the existing native derivative observer and unchanged VERL update_actor.
Only the prepared constant-positive PG ablation differs between the two arms.
There is no copied optimizer, loss, model, scheduler or training loop here.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import ray
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from observe_native_gradient_chain_v4 import ObservedWorker

SPEC_PATH=Path(os.environ['DT_NATIVE_DIAGNOSTIC_MANIFEST'])
SPEC=json.loads(SPEC_PATH.read_bytes())
DEST=Path(SPEC['destination'])
STATE=dict(scope=__doc__,manifest=str(SPEC_PATH),started_unix=time.time(),observations=[],
           native_optimizer_updates_executed=0,native_scheduler_updates_executed=0,
           formal_training_changed=False,GRPO_reruns=0,rollout_calls=0,DT_calls=0,validation_score_calls=0)

def phase(name,**fields):
    STATE.update(phase=name,observed_unix=time.time(),**fields)
    (DEST/'result.json').write_text(json.dumps(STATE,indent=2)+'\n')
    print(json.dumps(dict(phase=name,**fields)),flush=True)

def snapshot_native_state(worker,label):
    from pathlib import Path
    import hashlib,inspect,json,os,psutil,time,torch
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from observe_native_gradient_chain_v4 import digest
    tensors={}
    for name,parameter in worker.actor.actor_module.named_parameters():
        if parameter.requires_grad:
            value=parameter.detach()
            value=value.to_local() if hasattr(value,'to_local') else value
            tensors[name]=value.to(device='cpu',copy=True)
    path=DEST/f'{label}-rank{worker.rank}-lora-shards.pt'
    assert not path.exists();torch.save(tensors,path)
    steps=sorted({float(s['step']) for s in worker.actor_optimizer.state.values() if 'step' in s})
    source=Path(inspect.getsourcefile(ActorRolloutRefWorker))
    record=dict(rank=worker.rank,pid=os.getpid(),birth=psutil.Process().create_time(),label=label,
        scheduler=worker.actor_lr_scheduler.last_epoch,optimizer_steps=steps,trainable_SHA256=digest(worker.actor),
        trainable_snapshot=str(path),snapshot_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
        native_worker_path=str(source),native_worker_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
        diagnostic_parameter_artifact_not_formal_checkpoint=True)
    (DEST/f'{label}-rank{worker.rank}-state.json').write_text(json.dumps(record,indent=2)+'\n')
    return record

def restore_original_training_rng(worker,receipts):
    import hashlib,torch
    from pathlib import Path
    row=receipts[str(worker.rank)];path=Path(row['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['SHA256']
    # Native checkpoint owner restores CPU/device/numpy/python RNGs.
    worker.checkpoint_manager.load_rng_state(torch.load(path,map_location='cpu',weights_only=False))
    return dict(rank=worker.rank,original_captured_training_rng_restored=True,source=str(path))

def rpc(group,function,**kwargs):
    # The original public worker RPC executes these read/restore callbacks.
    return ray.get([w.execute_with_func_generator.remote(func=function,**kwargs) for w in group.workers])

def probe(group,case_name):
    case=next(c for c in SPEC['cases'] if c['name']==case_name)
    path=Path(case['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==case['SHA256']
    batch=DataProto.load_from_disk(path)
    batch.meta_info['global_token_num']=batch.batch['attention_mask'].sum(-1).tolist()
    batch.meta_info.pop('global_batch_info',None)
    bindings=group.set_case(case_name)
    assert all(r['scheduler']==case['checkpoint'] for r in bindings)
    original=group.compute_log_prob(batch)
    batch.batch['old_log_probs']=original.batch['old_log_probs']
    batch.batch['ref_log_prob']=batch.batch['old_log_probs'].clone()
    group.observe_chain(batch)
    records=[json.loads((DEST/f'iteration{case_name}-rank{rank}-PG.json').read_bytes()) for rank in (0,1)]
    return dict(case=case_name,state_kind=case['state_kind'],actual_bindings=bindings,
                fixed_seed_gradient_statistics=records[0]['gradient_statistics'])

def prepare_native_update(group,arm,data):
    """Optional diagnostic input preparation; original v47 uses the saved input."""
    return data

def main():
    assert SPEC['status']=='prepared_only_no_GPU_submission'
    assert not (DEST/'result.json').exists(), 'Do not repeat a live/completed diagnostic'
    launch=json.loads(Path(SPEC['original_launch']).read_bytes())
    config=OmegaConf.load(Path(os.environ['VERL_ROOT'])/'verl/trainer/config/ppo_trainer.yaml')
    for key,value in launch['options'].items():OmegaConf.update(config,key.lstrip('+'),value,force_add=True)
    config.actor_rollout_ref.actor.optim.total_training_steps=330
    env=dict(SPEC.get('worker_effective_environment',launch['effective_environment']))
    for key in ('CUDA_VISIBLE_DEVICES','RAY_ADDRESS','RAY_TMPDIR','VERL_GLOBAL_TOKEN_MEAN'):env.pop(key,None)
    env['PYTHONPATH']=str(DEST/'source')+os.pathsep+str(DEST.parent)+os.pathsep+env['PYTHONPATH']
    env['DT_NATIVE_DIAGNOSTIC_MANIFEST']=str(SPEC_PATH)
    env['DT_FSDP_EVENT_WAIT_RECEIPTS']=str(DEST/'original-eventwait')
    runtime=OmegaConf.to_container(config.ray_init.runtime_env,resolve=True)
    runtime.setdefault('env_vars',{}).update(env)
    for path,sha in SPEC['source_bindings'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
    phase('initializing_original_actor_only')
    ray.init(address='local',num_cpus=8,num_gpus=2,include_dashboard=False,
             _temp_dir='/tmp/ray-tc-common-update-v47',runtime_env=runtime)
    try:
        group=RayWorkerGroup(RayResourcePool([2],use_gpu=True,max_colocate_count=1),
            RayClassWithInitArgs(ObservedWorker,config.actor_rollout_ref,'actor'))
        group.init_model()
        for arm in SPEC['arms']:
            name=arm['name'];path=Path(arm['path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest()==arm['SHA256']
            phase('loading_original_checkpoint80',arm=name)
            group.load_checkpoint(local_path=SPEC['checkpoint_path'],del_local_after_load=False)
            before=rpc(group,snapshot_native_state,label=name+'-before')
            assert all(r['scheduler']==80 and r['optimizer_steps']==[320.] for r in before)
            before_probe=probe(group,name+'_before')
            rng=rpc(group,restore_original_training_rng,receipts=SPEC['captured_training_RNG_by_rank'])
            update=DataProto.load_from_disk(path)
            update=prepare_native_update(group,arm,update)
            phase('four_unchanged_native_optimizer_updates',arm=name)
            # Original fixed VERL owns losses, clipping, AdamW and scheduler.
            result=group.update_actor(update)
            STATE['native_optimizer_updates_executed']+=4
            STATE['native_scheduler_updates_executed']+=1
            after=rpc(group,snapshot_native_state,label=name+'-after')
            assert all(r['scheduler']==81 and r['optimizer_steps']==[324.] for r in after)
            after_probe=probe(group,name+'_after')
            STATE['observations'].append(dict(arm=name,before=before,after=after,
                before_probe=before_probe,after_probe=after_probe,original_RNG_restore=rng,
                original_native_update_metrics=result.meta_info['metrics']))
            phase('conditional_arm_complete',arm=name)
        phase('complete_conditional_update_to_gain_comparison_not_full_rootcause_acceptance')
    except BaseException as exc:
        phase('failed',exception=repr(exc));raise
    finally:ray.shutdown()

if __name__=='__main__':main()
