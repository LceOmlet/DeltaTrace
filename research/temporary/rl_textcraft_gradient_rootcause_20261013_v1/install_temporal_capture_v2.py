"""Passive native update artifacts for temporal DTPO root-cause regression.

Original update_policy owns all forwards, losses, backward, clipping and four
optimizer updates. Persist its input and existing returned metrics only. No
extra model/backward/optimizer/scheduler or random draws; no training logger.
"""
import hashlib
import json
import os
from pathlib import Path
import psutil
import ray
import runpy
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUN=ROOT/'runs/textcraft-dt-global-token-mean-temporal-20261014-v1'
ACTOR_SHA='ec0924e6abe561542ddd863bcf7040b995b19a34e500394a0604db6fc1dc5941'

def install_capture(worker):
    import hashlib,inspect,json,os,psutil,time
    from pathlib import Path
    from types import MethodType
    import torch
    from verl.trainer.ppo import core_algos
    owners=[w for w in worker.worker_dict.values() if getattr(w,'_is_actor',False)]
    assert len(owners)==1
    w=owners[0];actor=w.actor
    assert hashlib.sha256(Path(inspect.getsourcefile(type(actor))).read_bytes()).hexdigest()==ACTOR_SHA
    assert hashlib.sha256(Path(inspect.getsourcefile(core_algos)).read_bytes()).hexdigest()=='fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'
    assert not hasattr(actor,'_dtpo_temporal_capture_v2')
    original=actor.update_policy
    actor._dtpo_temporal_capture_v2=True
    directory=RUN/'temporal-update-records'/f'rank{w.rank}-pid{os.getpid()}'
    directory.mkdir(parents=True,exist_ok=False)
    def observed(self,data):
        before=w.actor_lr_scheduler.last_epoch
        iteration=before+1
        stem=f'iteration{iteration:04d}'
        record={'iteration':iteration,'rank':w.rank,'pid':os.getpid(),'birth':psutil.Process().create_time(),
          'before_unix':time.time(),'scheduler_before':before,'actor_SHA256':ACTOR_SHA,
          'capture_source':str(OUT/'install_temporal_capture_v2.py'),'original_owner_called_once':False,
          'added_model_backward_optimizer_scheduler_calls':0}
        if iteration<=100:
            try:
                snapshot=data.select(deepcopy=True)
                snapshot.batch=data.batch.clone(recurse=False)
                snapshot=snapshot.to('cpu')
                path=directory/(stem+'-input.pkl')
                snapshot.save_to_disk(str(path))
                weights={}
                for name,p in w.actor_module_fsdp.named_parameters():
                    if p.requires_grad:
                        value=p.detach();value=value.to_local() if hasattr(value,'to_local') else value
                        weights[name]=value.to('cpu',copy=True)
                weight_path=directory/(stem+'-lora-local.pt');torch.save(weights,weight_path)
                rng_path=directory/(stem+'-training-rng.pt');torch.save(w.checkpoint_manager.get_rng_state(),rng_path)
                rollout_rng_path=directory/(stem+'-rollout-rng.pt');torch.save(w.rollout_sharding_manager.gen_random_states.clone(),rollout_rng_path)
                record.update(input=str(path),input_bytes=path.stat().st_size,rows=len(snapshot),
                   lora_local=str(weight_path),training_rng=str(rng_path),rollout_rng=str(rollout_rng_path),
                   diagnostic_artifacts_not_formal_checkpoints=True,full_input_capture='completed',
                   global_token_mean_metadata='global_batch_info' in data.meta_info)
                del snapshot,weights
            except Exception as exc:
                record.update(full_input_capture='failed_optional_diagnostic',error_type=type(exc).__name__,error=str(exc))
        else:
            record['full_input_capture']='outside_predeclared_first100_artifact_window_training_continues'
        # The same original native method executes once with the untouched data.
        metrics=original(data=data)
        record.update(original_owner_called_once=True,after_unix=time.time(),
                      native_metrics=metrics,
                      native_gradient_norms_by_optimizer_minibatch=metrics.get('actor/grad_norm'),
                      scheduler_after_actor_before_worker_step=w.actor_lr_scheduler.last_epoch)
        try:(directory/(stem+'-record.json')).write_text(json.dumps(record,indent=2,default=str)+'\n')
        except OSError as exc:print(f'[passive temporal record unavailable] rank={w.rank} iteration={iteration} error={exc}',flush=True)
        return metrics
    actor.update_policy=MethodType(observed,actor)
    return {'pid':os.getpid(),'birth':psutil.Process().create_time(),'rank':w.rank,
            'directory':str(directory),'source_SHA256':hashlib.sha256((OUT/'install_temporal_capture_v2.py').read_bytes()).hexdigest(),
            'full_input_window':'first100 original complete iterations; not a training stop',
            'native_returned_metrics_window':'all original iterations','math_changed':False,
            'additional_model_backward_optimizer_scheduler_calls':0}

def main():
    launch=json.loads((RUN/'launch.json').read_bytes())
    driver=psutil.Process(launch['pid']);assert driver.create_time()==launch['birth'] and driver.status()!=psutil.STATUS_ZOMBIE
    receipt=RUN/'runtime-overrides/initial-state-and-temporal-capture-v2.json'
    assert not receipt.exists(),'Inspect prior receipt rather than reinstall callbacks'
    receipt.parent.mkdir(parents=True,exist_ok=True)
    for name in ('PYTHONPATH','LD_LIBRARY_PATH','MACA_PATH'):
        if name in launch['effective_environment']:assert os.environ.get(name)==launch['effective_environment'][name],name
    gcs=next(p for p in driver.children(recursive=True) if p.name()=='gcs_server')
    port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
    host=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
    reader=runpy.run_path(str(OUT/'read_worker_state.py'),run_name='original_worker_reader_library')['read_worker']
    passive=runpy.run_path(str(OUT/'install_passive_records_native_graph.py'),run_name='original_passive_recorder_library')['install_worker']
    ray.init(address=host+':'+port,log_to_driver=False)
    report={'started_unix':time.time(),'driver_pid':launch['pid'],'driver_birth':launch['birth'],
            'complete':False,'scope':__doc__,'source_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    receipt.write_text(json.dumps(report,indent=2)+'\n')
    try:
        names=[a for a in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
        assert len(names)==2
        actors=[ray.get_actor(a['name'],namespace=a['namespace']) for a in names]
        states=ray.get([a.execute_with_func_generator.remote(func=reader) for a in actors]);states.sort(key=lambda x:x['rank'])
        for state in states:
            assert (state['optimizer_states'],state['scheduler_last_epoch'])==(0,0),'Fresh initial state not observed; do not label initial'
            c=state['configuration']
            assert (c['lora_rank'],c['lora_alpha'],c['microbatch'],c['minibatch_per_rank'],c['context'],c['enforce_eager'])==(8,16,4,32,32768,False)
            assert c['entropy']==.001 and c['dual_clip']==3 and c['ppo_epochs']==1
            assert state['modules']['verl.workers.actor.dp_actor']['sha256']==ACTOR_SHA
            assert state['modules']['verl.workers.fsdp_workers']['sha256']=='b64b196e8079c38ea4b94cd9d911211f7aa7fa13434966d18c649eaab558e8ce'
            assert state['modules']['verl.utils.checkpoint.fsdp_checkpoint_manager']['sha256']=='f65b65383a7265b4d2808441427fb8a2ee891b84fcda4cdf1d961a3d96a47a01'
        report.update(initial_workers=states,initial_state_verified=True)
        receipt.write_text(json.dumps(report,indent=2)+'\n')
        passive.__globals__.update(FORMAL=str(RUN),WORKERS={w['pid']:w['birth'] for w in states})
        report['passive_credit_records']=ray.get([a.execute_with_func_generator.remote(func=passive) for a in actors])
        report['temporal_capture']=ray.get([a.execute_with_func_generator.remote(func=install_capture) for a in actors])
        report.update(complete=True,completed_unix=time.time(),status='actual_initial_owner_and_passive_temporal_capture_verified')
        receipt.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'receipt':str(receipt),'report':report}))
    finally:ray.shutdown()
if __name__=='__main__':main()
