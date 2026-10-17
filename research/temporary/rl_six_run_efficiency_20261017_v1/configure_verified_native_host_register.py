"""Configure the existing native allocator only after the actual eight-rank pair.

This does not replace a model, finite rule, update, allocator or scheduler. The
current run receives a completed runtime receipt; untouched launch records
remain launch facts. Queued runs receive one common resource configuration.
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
AUDIT = ROOT / 'receipts/six-run-native-utilization-20261017-v1'
RUN = ROOT / 'runs/textcraft-dtpo-v6-8gpu-200-20261017-v1'
VERSION = 'native-pinned-host-register-20261017-v1'


def proof():
    rows = [json.loads((AUDIT / f'one-natural-DT-native-allocator-pair-v2-rank{rank}.json').read_bytes())
        for rank in range(8)]
    for rank, row in enumerate(rows):
        assert row['rank'] == rank
        assert row['status'] == 'same_input_original_owner_pair_completed_candidate_not_deployed'
        assert row['raw_DT_vector_equal'] and row['raw_DT_max_abs_difference'] == 0
        assert row['baseline_rng_unchanged'] and row['candidate_rng_unchanged']
        assert row['original_callable_restored'] and row['native_allocator_semantics_restored']
        assert row['scheduler_before'] == row['scheduler_after']
        assert row['optimizer_entries_before'] == row['optimizer_entries_after']
        assert row['candidate']['owner_info']['complete_attribution_seconds_with_diagnostics'] < row['baseline']['owner_info']['complete_attribution_seconds_with_diagnostics']
    return rows


def configure(worker):
    import hashlib
    import json
    import os
    from pathlib import Path
    import sys
    import psutil
    import torch

    actor, = [v for v in worker.worker_dict.values() if getattr(v, '_is_actor', False)]
    expected = {r['pid']: r['birth'] for r in json.loads((RUN /
        'runtime-overrides/actual-worker-imports-after-first-DT-v1.json').read_bytes())['results']}
    assert psutil.Process().create_time() == expected[os.getpid()]
    runner = actor._deltatrace_producer.runner
    assert 'attribute' not in vars(runner), 'The paired observation must already be restored'
    owner = Path(runner.attribute.__func__.__code__.co_filename)
    assert hashlib.sha256(owner.read_bytes()).hexdigest() == '6a112224b3d4e63d047a576d7df68ff1ee65e4bbacfcb0014f84c17bd82b8520'
    before = torch.cuda.memory._snapshot()['allocator_settings']
    assert before['pinned_use_cuda_host_register'] is False
    state_before = (actor.actor_lr_scheduler.last_epoch, len(actor.actor_optimizer.state))
    configuration = before['PYTORCH_CUDA_ALLOC_CONF'] + ',pinned_use_cuda_host_register:True'
    torch.cuda.memory._set_allocator_settings(configuration)
    after = torch.cuda.memory._snapshot()['allocator_settings']
    assert after['pinned_use_cuda_host_register'] is True
    assert all(after[k] == before[k] for k in before if k not in
        ('PYTORCH_CUDA_ALLOC_CONF', 'pinned_use_cuda_host_register'))
    assert state_before == (actor.actor_lr_scheduler.last_epoch, len(actor.actor_optimizer.state))
    return dict(pid=os.getpid(), birth=psutil.Process().create_time(), rank=actor.rank,
        allocator_before=before, allocator_after=after, applied_native_configuration=configuration,
        scheduler_epoch=state_before[0], optimizer_entries=state_before[1],
        actual_DT_source=str(owner), actual_DT_SHA256=hashlib.sha256(owner.read_bytes()).hexdigest(),
        model_calls=0, DT_calls=0, optimizer_updates=0, training_method_changes=0)


def publish_queued_resource_configuration():
    changes = []
    for task, methods in [('textcraft', ['ppo', 'grpo']), ('appworld', ['dt', 'ppo', 'grpo'])]:
        audit = ROOT / 'receipts' / (task+'-eight-gpu-sequential-20261017-v1')
        for method in methods:
            path = audit / (method+'-prepared.json')
            raw = path.read_bytes()
            row = json.loads(raw)
            assert not Path(row['run']).exists(), 'Never rewrite a launched run as preparation'
            original = row['effective_environment']['PYTORCH_CUDA_ALLOC_CONF']
            assert original == 'expandable_segments:False'
            row['effective_environment']['PYTORCH_CUDA_ALLOC_CONF'] = original + ',pinned_use_cuda_host_register:True'
            row.setdefault('completed_preparation_resource_overrides', []).append(dict(
                version=VERSION, native_allocator_only=True, verification=str(AUDIT / (VERSION+'-verification.json'))))
            path.write_text(json.dumps(row, indent=2)+'\n')
            changes.append(dict(task=row['task'], method=method, path=str(path),
                before_SHA256=hashlib.sha256(raw).hexdigest(),
                after_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
                native_configuration=row['effective_environment']['PYTORCH_CUDA_ALLOC_CONF'],
                native_workload_configuration_changes=0, source_changes=0))
    return changes


if __name__ == '__main__':
    rows = proof()
    if '--client' not in sys.argv:
        assert not (AUDIT / (VERSION+'-job.json')).exists()
        source = json.loads((RUN / 'source.json').read_bytes())
        env = dict(os.environ, **source['effective_environment'])
        env['CUDA_VISIBLE_DEVICES'] = ''
        with (AUDIT / (VERSION+'.stdout')).open('xb') as out, (AUDIT / (VERSION+'.stderr')).open('xb') as err:
            child = subprocess.Popen([env['VENV_PYTHON'], __file__, '--client'], env=env,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
        record = dict(pid=child.pid, birth=psutil.Process(child.pid).create_time(), submitted=time.time(),
            source=__file__, source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            status='native_resource_configuration_submitted_waiting_worker_RPC_boundary')
        (AUDIT / (VERSION+'-job.json')).write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record))
    else:
        import ray
        source = json.loads((RUN / 'source.json').read_bytes())
        driver = psutil.Process(source['pid'])
        assert driver.create_time() == source['birth']
        gcs = next(p for p in driver.children(recursive=True) if p.name() == 'gcs_server')
        host = next(v.split('=',1)[1] for v in gcs.cmdline() if v.startswith('--node-ip-address='))
        port = next(v.split('=',1)[1] for v in gcs.cmdline() if v.startswith('--gcs_server_port='))
        verification = dict(collected=time.time(), status='eight_rank_same_input_raw_DT_bit_equal_RNG_state_preserved_and_measured_faster',
            cases=[dict(rank=r['rank'], paired_input_shape=r['input_shape'], baseline_seconds=r['baseline']['owner_info']['complete_attribution_seconds_with_diagnostics'],
                candidate_seconds=r['candidate']['owner_info']['complete_attribution_seconds_with_diagnostics'],
                raw_DT_equal=r['raw_DT_vector_equal']) for r in rows],
            exact_32768_eight_rank_capacity_verified=False,
            scope='One natural B4 per rank; native allocator configuration only, not a convergence or full-iteration speed proof')
        (AUDIT / (VERSION+'-verification.json')).write_text(json.dumps(verification, indent=2)+'\n')
        ray.init(address=host+':'+port, log_to_driver=False)
        try:
            names = [a for a in ray.util.list_named_actors(all_namespaces=True)
                if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
            assert len(names) == 8
            results = ray.get([ray.get_actor(a['name'], namespace=a['namespace']).execute_with_func_generator.remote(func=configure)
                for a in names])
            changes = publish_queued_resource_configuration()
            record = dict(collected=time.time(), status='completed_native_allocator_configuration_at_original_worker_boundaries',
                resource_version=VERSION, method_version='v6_unchanged', driver_pid=driver.pid, driver_birth=driver.create_time(),
                source=__file__, source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                verification=str(AUDIT / (VERSION+'-verification.json')), current_worker_overrides=results,
                queued_only_resource_changes=changes, original_launch_facts_retained=True,
                training_restarts=0, model_calls=0, DT_calls=0, optimizer_updates=0,
                exact_32768_eight_rank_capacity_verified=False)
            for path in [AUDIT / (VERSION+'-completed.json'), RUN / ('runtime-overrides/'+VERSION+'.json')]:
                path.write_text(json.dumps(record, indent=2)+'\n')
            print(json.dumps(record), flush=True)
        finally:
            ray.shutdown()
