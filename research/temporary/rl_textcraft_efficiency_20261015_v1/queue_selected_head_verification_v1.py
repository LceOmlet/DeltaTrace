"""Run a bounded head-only check on our own GPU at an existing RPC boundary.

Uses no other project's GPUs. The original formal worker/optimizer stay alive;
there is no formal optimizer step. All process RNG states and the temporary
head class dependency are restored before the original worker returns.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
RUN=ROOT/'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3'
DEST=ROOT/'receipts/textcraft-efficiency-20261015-v1'
WORKERS={3151306:1791993626.33,3153050:1791993643.98}


def verify_at_boundary(worker):
    import hashlib,importlib.util,json,os,pickle,random,time
    from pathlib import Path
    import numpy as np
    import psutil,torch
    assert psutil.Process().create_time()==WORKERS[os.getpid()]
    w=next(w for w in worker.worker_dict.values() if getattr(w,'_is_actor',False))
    if w.rank!=0:
        tick=time.monotonic()
        while not (DEST/'selected-head-worker-verification.json').exists() and time.monotonic()-tick<600:
            time.sleep(.25)
        return dict(rank=w.rank,pid=os.getpid(),birth=psutil.Process().create_time(),head_operator_calls=0)
    record=dict(unix=time.time(),pid=os.getpid(),birth=psutil.Process().create_time(),rank=w.rank,
        scheduler_before=w.actor_lr_scheduler.last_epoch,optimizer_identity_before=id(w.actor.actor_optimizer),
        physical_free_bytes_before=torch.cuda.mem_get_info()[0],physical_total_bytes=torch.cuda.mem_get_info()[1],
        formal_optimizer_scheduler_updates=0,DT_calls=0,transformer_calls=0)
    # This bounds this diagnostic's known full-weight/full-hidden/head work
    # buffers; it is a resource guard, not a numerical tolerance.
    if record['physical_free_bytes_before']<12*2**30:
        record['status']='not_run_insufficient_physical_free_memory_at_boundary'
        (DEST/'selected-head-worker-verification.json').write_text(json.dumps(record,indent=2)+'\n')
        return record
    py_state,np_state=random.getstate(),np.random.get_state()
    cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
    before=dict(cpu=hashlib.sha256(cpu_rng.numpy().tobytes()).hexdigest(),
        cuda=hashlib.sha256(gpu_rng.cpu().numpy().tobytes()).hexdigest(),
        python=hashlib.sha256(pickle.dumps(py_state)).hexdigest(),numpy=hashlib.sha256(pickle.dumps(np_state)).hexdigest())
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            spec=importlib.util.spec_from_file_location('selected_head_probe_owner',DEST/'verify_selected_head_owner_v1.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            result=module.verify(DEST)
        record.update(status='recorded_B4_and_full32768_head_owner_checks_passed',
            result=str(DEST/'head-selection-verification.json'),all_needed_outputs_gradients_exact=result['all_needed_outputs_gradients_exact'])
    except BaseException as exc:
        record.update(status='head_owner_check_failed_candidate_not_deployed',error=repr(exc))
    finally:
        random.setstate(py_state);np.random.set_state(np_state)
        after=dict(cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            cuda=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
            python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
            numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest())
        record.update(completed_unix=time.time(),RNG_before=before,RNG_after=after,RNG_restored=before==after,
            scheduler_after=w.actor_lr_scheduler.last_epoch,optimizer_identity_after=id(w.actor.actor_optimizer),
            physical_free_bytes_after=torch.cuda.mem_get_info()[0])
        assert record['RNG_restored'] and record['scheduler_before']==record['scheduler_after']
        assert record['optimizer_identity_before']==record['optimizer_identity_after']
        (DEST/'selected-head-worker-verification.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    if '--client' in sys.argv:
        import ray
        launch=json.loads((RUN/'launch.json').read_bytes());p=psutil.Process(launch['pid']);assert p.create_time()==launch['birth']
        gcs=next(c for c in p.children(recursive=True) if c.name()=='gcs_server')
        host=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
        port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
        ray.init(address=host+':'+port,log_to_driver=False)
        try:
            names=[a for a in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
            assert len(names)==2
            values=ray.get([ray.get_actor(a['name'],namespace=a['namespace']).execute_with_func_generator.remote(func=verify_at_boundary) for a in names],timeout=1800)
            (DEST/'selected-head-verification-client-result.json').write_text(json.dumps(dict(unix=time.time(),workers=values),indent=2)+'\n')
        finally:ray.shutdown()
    elif not (DEST/'selected-head-verification-client.json').exists():
        assert (DEST/'native-frozen-head-full-chunk.json').exists(),'Preserve the pending actual fixture capture'
        launch=json.loads((RUN/'launch.json').read_bytes());assert psutil.Process(launch['pid']).create_time()==launch['birth']
        env=dict(os.environ);env.update(launch['effective_environment']);env['CUDA_VISIBLE_DEVICES']=''
        stdout,stderr=DEST/'selected-head-verification.stdout.log',DEST/'selected-head-verification.stderr.log'
        with stdout.open('xb') as out,stderr.open('xb') as err:
            p=subprocess.Popen([env['VENV_PYTHON'],'-u',__file__,'--client'],env=env,
                stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        record=dict(unix=time.time(),pid=p.pid,birth=psutil.Process(p.pid).create_time(),stdout=str(stdout),stderr=str(stderr),
            scope='Bounded original head verification inside our own workers at RPC boundary; no formal update or restart')
        (DEST/'selected-head-verification-client.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
