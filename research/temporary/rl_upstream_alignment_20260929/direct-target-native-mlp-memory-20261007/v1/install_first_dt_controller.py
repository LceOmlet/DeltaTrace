"""Prepared-only controller for one explicit new formal job's observer RPC.

Run on the provisioned host only after its new PID/birth/source are supplied.
Use the original WorkerDict execute_with_func_generator and Ray futures. A
60-second timeout waits on the SAME references; it never resubmits an RPC.
No process suspension, generation, forward, update or checkpoint call lives here.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_job(spec):
    import psutil
    driver = psutil.Process(spec['driver_pid'])
    if driver.create_time() != spec['driver_birth']:
        raise RuntimeError('Explicit driver PID creation time differs')
    if sha(spec['source']) != spec['source_sha256']:
        raise RuntimeError('Explicit formal source SHA differs')
    return driver


def load_bound(path, expected_sha, name):
    import importlib.util
    if sha(path) != expected_sha:
        raise RuntimeError(f'Prepared observer source SHA differs: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_worker(worker, *, spec):
    """Metadata and observer installation only, on the original worker RPC."""
    import inspect
    import os
    import psutil
    check_job(spec)
    owners, seen = [], set()
    for owner in worker.worker_dict.values():
        if (id(owner) not in seen and getattr(owner, 'actor', None) is not None
                and callable(getattr(owner, 'compute_dt_token_advantages', None))
                and callable(getattr(owner, 'update_actor', None))):
            seen.add(id(owner))
            if getattr(owner, '_deltatrace_producer', None) is not None:
                raise RuntimeError('Original producer already exists; first-DT install is too late')
            worker_path = inspect.getsourcefile(type(owner))
            actor_path = inspect.getsourcefile(type(owner.actor))
            for actual, relative in ((worker_path, 'verl/workers/fsdp_workers.py'),
                                     (actor_path, 'verl/workers/actor/dp_actor.py')):
                if Path(actual).resolve() != (Path(spec['verl']) / relative).resolve():
                    raise RuntimeError('Original VERL import differs from explicit checkout')
            if sha(worker_path) != spec['worker_sha256']:
                raise RuntimeError('Original VERL worker source SHA differs')
            if sha(actor_path) != spec['actor_sha256']:
                raise RuntimeError('Original PPO actor source SHA differs')
            if owner.actor.config.ppo_micro_batch_size_per_gpu != 4:
                raise RuntimeError('Original actor microbatch is not4')
            adapters = [dict(rank=cfg.r, alpha=cfg.lora_alpha)
                        for cfg in owner.actor.actor_module.peft_config.values()]
            if not adapters or any(cfg != dict(rank=8, alpha=16) for cfg in adapters):
                raise RuntimeError('Original LoRA rank/alpha differ from8/16')
            owners.append(dict(rank=owner.rank, worker_path=worker_path,
                worker_sha256=sha(worker_path), actor_path=actor_path,
                actor_sha256=sha(actor_path), microbatch=4, adapters=adapters))
    if len(owners) != 1:
        raise RuntimeError(f'Expected one original actor owner in WorkerDict: {owners}')
    import deltatrace_rollout
    producer_path = Path(inspect.getsourcefile(deltatrace_rollout)).resolve()
    if producer_path != (Path(spec['entry']) / 'deltatrace_rollout.py').resolve():
        raise RuntimeError('Actual producer import differs from explicit entry')
    source = json.loads(Path(spec['source']).read_bytes())
    if sha(producer_path) != source['entry_sha256']['deltatrace_rollout.py']:
        raise RuntimeError('Actual producer SHA differs from explicit source receipt')
    provenance = dict(driver_pid=spec['driver_pid'], driver_birth=spec['driver_birth'],
        source_path=spec['source'], source_sha256=spec['source_sha256'],
        controller_out=spec['out'], controller_sha256=spec['controller_sha256'],
        owner=owners[0], producer_path=str(producer_path), producer_sha256=sha(producer_path),
        purpose='Original formal sampling then first DT/DataProto capture and Event hold before first update')
    observer = load_bound(spec['observer_path'], spec['observer_sha256'],
                          'formal_first_dt_token_observer_v2')
    seam = load_bound(spec['first_dt_path'], spec['first_dt_sha256'],
                      'formal_first_dt_input_observer_v4')
    replay_scope_factory = None
    if spec['memory_path'] is not None:
        memory = load_bound(spec['memory_path'], spec['memory_sha256'],
                            'formal_passive_native_mlp_metadata')
        provenance.update(memory_path=spec['memory_path'],
                          memory_sha256=spec['memory_sha256'])

        def replay_scope_factory(owner, producer):
            return memory.passive_native_mlp(producer.runner.model,
                Path(spec['out']) / f'rank{owner.rank}-native-mlp.jsonl',
                source_identity=provenance)

    armed = seam.install_before_first_dt(worker, spec['out'], observer_module=observer,
        replay_scope_factory=replay_scope_factory, provenance=provenance, hold=True)
    return dict(pid=os.getpid(), birth=psutil.Process().create_time(),
                installed_unix=time.time(), owner=owners[0], armed=armed)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver-pid', type=int, required=True)
    parser.add_argument('--driver-birth', type=float, required=True)
    for name in ('source', 'source-sha256', 'entry', 'verl', 'out', 'first-dt-path',
                 'first-dt-sha256', 'observer-path', 'observer-sha256',
                 'actor-sha256', 'worker-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--memory-path')
    parser.add_argument('--memory-sha256')
    args = vars(parser.parse_args())
    if bool(args['memory_path']) != bool(args['memory_sha256']):
        parser.error('--memory-path and --memory-sha256 must be supplied together')
    return args


def main():
    import os
    import psutil
    spec = arguments()
    spec['controller_sha256'] = sha(__file__)
    driver = check_job(spec)
    for path, expected in ((spec['first_dt_path'], spec['first_dt_sha256']),
                           (spec['observer_path'], spec['observer_sha256'])):
        if sha(path) != expected:
            raise RuntimeError(f'Observer source SHA differs: {path}')
    if spec['memory_path'] and sha(spec['memory_path']) != spec['memory_sha256']:
        raise RuntimeError('Passive native MLP source SHA differs')
    out = Path(spec['out'])
    out.mkdir(parents=True, exist_ok=True)
    record = dict(status='connecting_no_RPC_submitted', observed_unix=time.time(),
        controller=dict(pid=os.getpid(), birth=psutil.Process().create_time()),
        explicit_job=spec, operations=dict(model=0, generation=0, update=0, checkpoint=0,
                                        process_suspend=0, process_resume=0))
    # A second controller must not create duplicate queued installation RPCs.
    with (out / 'controller-owner.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')

    def save(status, **fields):
        record.update(status=status, observed_unix=time.time(), **fields)
        (out / 'controller-status.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record), flush=True)

    for path in reversed([spec['entry'], spec['verl']]):
        sys.path.insert(0, path)
    import ray
    servers = [p for p in driver.children(recursive=True) if p.name() == 'gcs_server']
    if len(servers) != 1:
        raise RuntimeError('Explicit driver has no unique original GCS child')
    port = next(arg.split('=', 1)[1] for arg in servers[0].cmdline()
                if arg.startswith('--gcs_server_port='))
    ray.init(address=f'127.0.0.1:{port}', log_to_driver=False)
    try:
        actors = [row for row in ray.util.list_named_actors(all_namespaces=True)
                  if 'WorkerDict' in row['name']]
        if len(actors) != 2:
            raise RuntimeError(f'Explicit dual-card job has unexpected WorkerDict actors: {actors}')
        save('submitting_once', actors=actors, gcs_port=port)
        refs = [ray.get_actor(row['name'], namespace=row['namespace'])
                    .execute_with_func_generator.remote(func=partial(install_worker, spec=spec))
                for row in actors]
        save('RPCs_queued', refs=[ref.hex() for ref in refs], submissions=len(refs))
        timeouts = 0
        while True:
            try:
                results = ray.get(refs, timeout=60)
                break
            except ray.exceptions.GetTimeoutError:
                timeouts += 1
                save('same_RPCs_pending_not_resubmitted', wait_timeouts=timeouts)
                check_job(spec)
        if sorted(row['owner']['rank'] for row in results) != [0, 1]:
            raise RuntimeError('Installed original owner ranks are not0/1')
        save('observer_armed_not_training_health_proof', workers=results)
    except BaseException as exc:
        save('controller_error_no_resubmission',
             error_type=type(exc).__name__, error_message=str(exc))
        raise
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
