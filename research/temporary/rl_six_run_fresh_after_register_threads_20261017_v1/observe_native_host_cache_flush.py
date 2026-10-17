"""Measure existing host-cache frees in one natural DT call; never add a free."""
import ast
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT / 'receipts/six-run-native-utilization-20261017-v1/source/observe_native_host_allocation_time.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '2368842e65e043ad98bd44b811a246e73ea90ad7dd28959ebd2787000ec01606'
code = source.read_text().replace('receipts/six-run-native-utilization-20261017-v1',
    'receipts/six-run-fresh-after-register-threads-20261017-v1')
code = code.replace('runs/textcraft-dtpo-v6-8gpu-200-20261017-v1',
                    'runs/textcraft-dtpo-v6-8gpu-200-20261017-v3')
code = code.replace('host-counter-observer', 'host-cache-flush-observer')
tree = ast.parse(code)
main = tree.body.pop()
assert isinstance(main, ast.If) and '__name__' in ast.unparse(main.test)
exec(compile(ast.Module(body=tree.body, type_ignores=[]), str(source), 'exec'), globals())


def install(worker):
    import hashlib
    import json
    import os
    from pathlib import Path
    import time
    from types import MethodType
    import psutil
    import torch

    actor, = [v for v in worker.worker_dict.values() if getattr(v, '_is_actor', False)]
    expected = {r['pid']:r['birth'] for r in json.loads((RUN /
        'runtime-overrides/actual-worker-imports-after-first-DT-v1.json').read_bytes())['results']}
    assert psutil.Process().create_time() == expected[os.getpid()]
    runner = actor._deltatrace_producer.runner
    delegate = runner.attribute
    assert hashlib.sha256(Path(delegate.__func__.__code__.co_filename).read_bytes()).hexdigest() == '6a112224b3d4e63d047a576d7df68ff1ee65e4bbacfcb0014f84c17bd82b8520'
    assert not hasattr(runner, '_native_host_cache_flush_observed')
    runner._native_host_cache_flush_observed = True
    had_instance_attribute = 'attribute' in vars(runner)
    prior_instance_attribute = vars(runner).get('attribute')

    def observe(instance, *args, **kwargs):
        original_free = torch._C._host_emptyCache
        record = dict(pid=os.getpid(), birth=psutil.Process().create_time(),rank=actor.rank,
            started=time.time(),source=__file__,source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scheduler_before=actor.actor_lr_scheduler.last_epoch,
            optimizer_entries_before=len(actor.actor_optimizer.state),
            input_shape=list((args[0] if args else kwargs['paired_ids']).shape),
            flush_calls=[],additional_model_DT_optimizer_tensor_collective_or_sync_calls=0,
            original_natural_result_returned=True,flush_policy_changed=False)
        changed = False

        def stats():
            try:
                values = torch.cuda.memory.host_memory_stats()
                return {k:v for k,v in values.items() if k.startswith((
                    'allocated_bytes.', 'active_bytes.', 'reserved_bytes.',
                    'host_alloc_time.', 'host_free_time.')) or k in ('num_host_alloc','num_host_free')}
            except Exception as exc:
                return dict(observation_unavailable=type(exc).__name__)

        def counted_free():
            before = stats()
            started = time.perf_counter()
            try:
                return original_free()
            finally:
                elapsed = time.perf_counter()-started
                record['flush_calls'].append(dict(native_free_seconds=elapsed,
                    before=before,after=stats()))

        try:
            record['allocator_before'] = torch.cuda.memory._snapshot()['allocator_settings']
            record['host_before'] = stats()
            try:
                torch._C._host_emptyCache = counted_free
                changed = True
            except Exception as exc:
                record['optional_wrapping_unavailable'] = type(exc).__name__
            result = delegate(*args, **kwargs)
            record['owner_seconds_with_optional_observation'] = result[1].get('complete_attribution_seconds_with_diagnostics')
            record['status'] = 'one_original_natural_attribute_returned'
            return result
        except BaseException as exc:
            record.update(status='original_attribute_raised',exception_type=type(exc).__name__)
            raise
        finally:
            if changed:
                torch._C._host_emptyCache = original_free
            if had_instance_attribute:
                runner.attribute = prior_instance_attribute
            else:
                del runner.attribute
            record.update(finished=time.time(),native_free_callable_restored=torch._C._host_emptyCache is original_free,
                original_attribute_restored=runner.attribute == delegate,host_after=stats(),
                scheduler_after=actor.actor_lr_scheduler.last_epoch,
                optimizer_entries_after=len(actor.actor_optimizer.state))
            record['scope'] = 'One unchanged natural B4, existing frees only. Native API timing includes waiting; counter reads perturb timing. No new model/DT/free/optimizer/sync or policy change.'
            try:
                with (AUDIT / f'one-natural-DT-host-cache-flush-rank{actor.rank}.json').open('x') as stream:
                    json.dump(record,stream,indent=2)
            except Exception as exc:
                print('Optional host-cache observation unavailable: '+type(exc).__name__,flush=True)

    runner.attribute = MethodType(observe,runner)
    return dict(pid=os.getpid(),birth=psutil.Process().create_time(),rank=actor.rank,
        scheduler=actor.actor_lr_scheduler.last_epoch,installed=True,
        additional_model_DT_tensor_optimizer_calls=0,flush_policy_changed=False)


exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), 'exec'), globals())
