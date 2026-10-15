"""Keep existing DT owner timing results for one naturally scheduled RPC.

No extra forward, attribution, tensor operation, backward or optimizer call.
Both temporary method adapters restore themselves after this single RPC.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
RUN = ROOT/'runs/textcraft-dt-grpo-multiplier-variance-only-20261014-v3'
DEST = ROOT/'receipts/textcraft-efficiency-20261015-v1'
WORKERS = {3151306: 1791993626.33, 3153050: 1791993643.98}


def install(worker):
    import hashlib
    import inspect
    import json
    import os
    from pathlib import Path
    import sys
    import time
    from types import MethodType
    import psutil

    assert psutil.Process().create_time() == WORKERS[os.getpid()]
    owners = [w for w in worker.worker_dict.values() if getattr(w, '_is_actor', False)]
    assert len(owners) == 1
    w = owners[0]
    producer = w._deltatrace_producer
    runner, readout = producer.runner, producer.direct_readout
    assert runner.defer_diagnostics
    assert not hasattr(readout, '_efficiency_existing_timings_v1')
    bindings = {}
    for name in ('deltatrace_rollout', 'reward_readout', 'native_prefix_leases',
                 'qwen35_dense_finite_runner', 'qwen35_gdn_finite', 'qwen35_decoder_finite',
                 'accelerated.qwen35.qwen35_code_local_capture',
                 'verl.workers.actor.dp_actor', 'verl.utils.experimental.torch_functional'):
        m = sys.modules.get(name)
        path = getattr(m, '__file__', None)
        if path:
            bindings[name] = dict(path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    original_attribution = runner.attribute
    original_trajectories = readout.trajectories
    attr_existed, trajectories_existed = 'attribute' in vars(runner), 'trajectories' in vars(readout)
    previous_attr, previous_trajectories = vars(runner).get('attribute'), vars(readout).get('trajectories')
    readout._efficiency_existing_timings_v1 = True
    record = dict(pid=os.getpid(), birth=psutil.Process().create_time(), rank=w.rank,
        installed_unix=time.time(), scheduler_at_install=w.actor_lr_scheduler.last_epoch,
        bindings=bindings, existing_owner_batches=[], additional_model_calls=0,
        additional_DT_calls=0, additional_tensor_operations=0, additional_collectives=0,
        additional_optimizer_calls=0, training_math_changes=0,
        flags={k:getattr(runner,k,None) for k in ('defer_diagnostics','copy_replay_captures',
            'offload_replay_mixer','pin_replay_host','pin_root_host','reuse_native_prefix',
            'fa_coefficient_suffix','gdn_coefficient_suffix','compact_gdn_captures')})

    def observed_attribution(instance, *args, **kwargs):
        result = original_attribution(*args, **kwargs)
        try:
            ids = args[0] if args else kwargs['paired_ids']
            record['existing_owner_batches'].append(dict(input_shape=list(ids.shape),
                owner_info=result[1], original_return_object_unchanged=True))
        except Exception as exc:
            record.setdefault('observation_errors', []).append(repr(exc))
        return result

    def observed_trajectories(instance, *args, **kwargs):
        record['started_unix'] = time.time()
        record['scheduler_before_DT'] = w.actor_lr_scheduler.last_epoch
        try:
            result = original_trajectories(*args, **kwargs)
            record['existing_readout_report'] = instance.last_report
            record['status'] = 'one_naturally_scheduled_DT_RPC_returned'
            return result
        except BaseException as exc:
            record.update(status='original_DT_RPC_raised', original_exception=repr(exc))
            raise
        finally:
            if attr_existed: runner.attribute = previous_attr
            else: del runner.attribute
            if trajectories_existed: instance.trajectories = previous_trajectories
            else: del instance.trajectories
            record['restored_unix'] = time.time()
            record['original_attribute_restored'] = runner.attribute == original_attribution
            record['original_trajectories_restored'] = instance.trajectories == original_trajectories
            try:
                path = DEST/f'rank{w.rank}-existing-DT-timings.json'
                with path.open('x') as f:json.dump(record,f,indent=2)
            except Exception as exc:
                print('Existing DT timing observation write failed: '+repr(exc),flush=True)

    runner.attribute = MethodType(observed_attribution, runner)
    readout.trajectories = MethodType(observed_trajectories, readout)
    return dict(pid=os.getpid(), birth=psutil.Process().create_time(), rank=w.rank,
        scheduler_at_install=w.actor_lr_scheduler.last_epoch, bindings=bindings,
        previous_readout_summary={k:v for k,v in readout.last_report.items()
            if k not in ('traces','actual_context_lengths','original_response_rows')},
        installed=True, extra_model_DT_optimizer_calls=0)


def status():
    value = dict(unix=time.time(), captures=[])
    for name in ('client.json','installation.json'):
        p=DEST/name
        if p.exists():value[name]=json.loads(p.read_bytes())
    for p in sorted(DEST.glob('rank*-existing-DT-timings.json')):
        value['captures'].append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                      value=json.loads(p.read_bytes())))
    if (DEST/'client.json').exists():
        c=json.loads((DEST/'client.json').read_bytes())
        try:value['client_alive']=psutil.Process(c['pid']).create_time()==c['birth'] and psutil.Process(c['pid']).status()!=psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:value['client_alive']=False
        value['stderr_tail']=Path(c['stderr']).read_text(errors='replace')[-2200:]
    return value


DEST.mkdir(exist_ok=True)
if '--client' in sys.argv:
    import ray
    launch=json.loads((RUN/'launch.json').read_bytes())
    p=psutil.Process(launch['pid']);assert p.create_time()==launch['birth']
    gcs=next(c for c in p.children(recursive=True) if c.name()=='gcs_server')
    host=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
    port=next(a.split('=',1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
    ray.init(address=host+':'+port,log_to_driver=False)
    try:
        names=[a for a in ray.util.list_named_actors(all_namespaces=True)
               if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
        assert len(names)==2
        workers=ray.get([ray.get_actor(a['name'],namespace=a['namespace']).execute_with_func_generator.remote(func=install)
                         for a in names],timeout=7200)
        assert {w['pid']:w['birth'] for w in workers}==WORKERS
        report=dict(unix=time.time(),workers=workers,source=__file__,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            status='one_RPC_existing_owner_timing_observation_installed',
            additional_model_DT_optimizer_calls=0,training_math_changes=0)
        for path in (DEST/'installation.json',RUN/'runtime-overrides/existing-DT-timings-efficiency-20261015-v1.json'):
            with path.open('x') as f:json.dump(report,f,indent=2)
    finally:ray.shutdown()
elif '--read' not in sys.argv and not (DEST/'client.json').exists():
    launch=json.loads((RUN/'launch.json').read_bytes())
    assert psutil.Process(launch['pid']).create_time()==launch['birth']
    env=dict(os.environ);env.update(launch['effective_environment']);env['CUDA_VISIBLE_DEVICES']=''
    stdout,stderr=DEST/'client.stdout.log',DEST/'client.stderr.log'
    with stdout.open('xb') as out,stderr.open('xb') as err:
        p=subprocess.Popen([env['VENV_PYTHON'],'-u',__file__,'--client'],env=env,
            stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    (DEST/'client.json').write_text(json.dumps(dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),
        stdout=str(stdout),stderr=str(stderr)),indent=2)+'\n')
print(json.dumps(status()))
