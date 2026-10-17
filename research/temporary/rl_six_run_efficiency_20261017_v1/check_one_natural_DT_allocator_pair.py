"""Compare the original DT owner with two native allocator configurations.

At an original worker RPC boundary, wrap exactly one future natural attribute
call. Return its original result to training; run one additional same-input DT
call only for allocator validation, then restore the original configuration and
callable. No optimizer, rollout, parser, model or finite rule is implemented.
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
VERSION = 'one-natural-DT-native-allocator-pair-v2'


def install(worker):
    import json
    import os
    from pathlib import Path
    import time
    from types import MethodType
    import psutil
    import torch

    actor, = [v for v in worker.worker_dict.values() if getattr(v, '_is_actor', False)]
    expected = {r['pid']: r['birth'] for r in json.loads((RUN /
        'runtime-overrides/actual-worker-imports-after-first-DT-v1.json').read_bytes())['results']}
    assert psutil.Process().create_time() == expected[os.getpid()]
    runner = actor._deltatrace_producer.runner
    cancellation = None
    if 'attribute' in vars(runner):
        pending = runner.attribute
        assert pending.__func__.__name__ == 'observe'
        assert Path(pending.__func__.__code__.co_filename) == AUDIT / 'source/check_one_natural_DT_allocator_pair.py'
        assert not (AUDIT / f'one-natural-DT-native-allocator-pair-v1-rank{actor.rank}.json').exists()
        cells = dict(zip(pending.__func__.__code__.co_freevars, pending.__func__.__closure__))
        previous = cells['delegate'].cell_contents
        del runner.attribute
        assert runner.attribute == previous
        del runner._native_allocator_pair_installed
        cancellation = dict(status='v1_cancelled_before_execution_no_allocator_change',
            reason='Actual VERL allocator settings differ from the launch environment; use call-time owner settings',
            previous_callable_restored=True, additional_DT_calls=0,
            native_allocator_settings=torch.cuda.memory._snapshot()['allocator_settings'])
        (AUDIT / f'one-natural-DT-native-allocator-pair-v1-cancelled-rank{actor.rank}.json').write_text(
            json.dumps(cancellation, indent=2)+'\n')
    assert 'attribute' not in vars(runner), 'Wait for prior one-RPC observations to restore'
    delegate = runner.attribute
    assert delegate.__func__.__name__ == 'attribute'
    assert not hasattr(runner, '_native_allocator_pair_installed')
    runner._native_allocator_pair_installed = True
    settings = torch.cuda.memory._snapshot()['allocator_settings']
    assert settings['pinned_use_cuda_host_register'] is False
    assert settings['pinned_num_register_threads'] == 1

    def observe(instance, *args, **kwargs):
        current = torch.cuda.memory._snapshot()['allocator_settings']
        assert current['pinned_use_cuda_host_register'] is False
        current_config = current['PYTORCH_CUDA_ALLOC_CONF']
        baseline_config = (current_config + ',pinned_use_cuda_host_register:False,pinned_num_register_threads:'
            + str(current['pinned_num_register_threads']))
        candidate_config = (current_config + ',pinned_use_cuda_host_register:True,pinned_num_register_threads:'
            + str(current['pinned_num_register_threads']))
        record = dict(rank=actor.rank, pid=os.getpid(), birth=psutil.Process().create_time(),
            started=time.time(), version=VERSION, baseline_config=baseline_config,
            candidate_config=candidate_config, original_allocator_settings=current,
            baseline_training_result_returned_unchanged=True, additional_DT_calls=1,
            additional_optimizer_calls=0, additional_rollout_calls=0,
            scheduler_before=actor.actor_lr_scheduler.last_epoch,
            optimizer_entries_before=len(actor.actor_optimizer.state),
            original_runner_source=delegate.__func__.__code__.co_filename)
        cpu_rng = torch.get_rng_state()
        gpu_rng = torch.cuda.get_rng_state()
        baseline = None
        try:
            ids = args[0] if args else kwargs['paired_ids']
            record['input_shape'] = list(ids.shape)
            before = dict(torch.cuda.memory.host_memory_stats())
            baseline = delegate(*args, **kwargs)
            after = dict(torch.cuda.memory.host_memory_stats())
            record['baseline'] = dict(owner_info=baseline[1], before_host=before, after_host=after)
            record['baseline_rng_unchanged'] = bool(torch.equal(cpu_rng, torch.get_rng_state())
                and torch.equal(gpu_rng, torch.cuda.get_rng_state()))
            torch.cuda.memory._set_allocator_settings(candidate_config)
            record['candidate_allocator_settings'] = torch.cuda.memory._snapshot()['allocator_settings']
            other_keys = [k for k in current if k not in ('PYTORCH_CUDA_ALLOC_CONF','pinned_use_cuda_host_register')]
            assert all(record['candidate_allocator_settings'][k] == current[k] for k in other_keys), 'Only the requested native pinned allocator backend may change'
            before = dict(torch.cuda.memory.host_memory_stats())
            candidate = delegate(*args, **kwargs)
            after = dict(torch.cuda.memory.host_memory_stats())
            record['candidate'] = dict(owner_info=candidate[1], before_host=before, after_host=after)
            record['raw_DT_vector_equal'] = bool(torch.equal(baseline[0], candidate[0]))
            record['raw_DT_max_abs_difference'] = float((baseline[0]-candidate[0]).abs().max())
            record['candidate_rng_unchanged'] = bool(torch.equal(cpu_rng, torch.get_rng_state())
                and torch.equal(gpu_rng, torch.cuda.get_rng_state()))
            record['status'] = 'same_input_original_owner_pair_completed_candidate_not_deployed'
            return baseline
        except Exception as exc:
            record.update(status='optional_allocator_pair_failed_candidate_rejected',
                error_type=type(exc).__name__, error=str(exc))
            if baseline is None:
                raise
            return baseline
        finally:
            torch.cuda.memory._set_allocator_settings(baseline_config)
            del runner.attribute
            record.update(finished=time.time(), original_callable_restored=runner.attribute == delegate,
                restored_allocator_settings=torch.cuda.memory._snapshot()['allocator_settings'],
                scheduler_after=actor.actor_lr_scheduler.last_epoch,
                optimizer_entries_after=len(actor.actor_optimizer.state))
            record['native_allocator_semantics_restored'] = all(
                record['restored_allocator_settings'][k] == current[k] for k in current
                if k != 'PYTORCH_CUDA_ALLOC_CONF')
            try:
                with (AUDIT / f'{VERSION}-rank{actor.rank}.json').open('x') as stream:
                    json.dump(record, stream, indent=2)
            except Exception as exc:
                print('Optional allocator-pair receipt unavailable: '+type(exc).__name__, flush=True)

    runner.attribute = MethodType(observe, runner)
    return dict(pid=os.getpid(), birth=psutil.Process().create_time(), rank=actor.rank,
        installed=True, scheduler=actor.actor_lr_scheduler.last_epoch,
        original_allocator_settings=settings, original_runner_source=delegate.__func__.__code__.co_filename,
        prior_unexecuted_v1_cancellation=cancellation,
        status='one_next_natural_attribute_pair_prepared_not_executed')


if __name__ == '__main__':
    if '--client' not in sys.argv:
        proof = json.loads((AUDIT / 'installed-native-pinned-allocator-behavior-v2.json').read_bytes())
        assert proof['status'] == 'installed_API_mixed_allocation_and_original_free_passed'
        assert not (AUDIT / (VERSION+'-job.json')).exists()
        source = json.loads((RUN / 'source.json').read_bytes())
        env = dict(os.environ, **source['effective_environment'])
        env['CUDA_VISIBLE_DEVICES'] = ''
        with (AUDIT / (VERSION+'.stdout')).open('xb') as out, (AUDIT / (VERSION+'.stderr')).open('xb') as err:
            child = subprocess.Popen([env['VENV_PYTHON'], __file__, '--client'], env=env,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
        record = dict(pid=child.pid, birth=psutil.Process(child.pid).create_time(), submitted=time.time(),
            source=__file__, source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            status='metadata_installation_pending', candidate_deployed=False)
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
        ray.init(address=host+':'+port, log_to_driver=False)
        try:
            names = [a for a in ray.util.list_named_actors(all_namespaces=True)
                if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
            assert len(names) == 8
            rows = ray.get([ray.get_actor(a['name'], namespace=a['namespace']).execute_with_func_generator.remote(func=install)
                for a in names])
            record = dict(collected=time.time(), status='installed_one_native_attribute_pair', workers=rows,
                additional_DT_calls_per_rank_when_executed=1, optimizer_updates=0, candidate_deployed=False)
            (AUDIT / (VERSION+'-installation.json')).write_text(json.dumps(record, indent=2)+'\n')
        finally:
            ray.shutdown()
