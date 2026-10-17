"""A1/B8-cold/B8-warm/A1 on one natural native DT input; return A1.

Only the original pinned allocator page-mapping thread count is varied. The
existing process/Ray submission owner is reused; no model or trainer is copied.
All candidate work is diagnostic and the original setting/callable is restored.
"""
import ast
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT / 'receipts/six-run-native-utilization-20261017-v1/source/check_one_natural_DT_allocator_pair_v2.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '346f369e3b2b7ed58825673b6e697e0df7e0e4bdfa7f68a6efb9713711315326'
code = source.read_text().replace('receipts/six-run-native-utilization-20261017-v1',
    'receipts/six-run-fresh-after-efficiency-20261017-v1').replace(
    'runs/textcraft-dtpo-v6-8gpu-200-20261017-v1', 'runs/textcraft-dtpo-v6-8gpu-200-20261017-v2')
code = code.replace("VERSION = 'one-natural-DT-native-allocator-pair-v2'",
                    "VERSION = 'one-natural-DT-register-threads-ABBA-v1'")
code = code.replace("AUDIT / 'installed-native-pinned-allocator-behavior-v2.json'",
                    "AUDIT / 'fresh-eight-rank-first-DT-verified.json'")
code = code.replace("assert proof['status'] == 'installed_API_mixed_allocation_and_original_free_passed'",
                    "assert proof['status'] == 'fresh_actual_eight_rank_DT_execution_and_imports_verified'")
code = code.replace('additional_DT_calls_per_rank_when_executed=1',
                    'additional_DT_calls_per_rank_when_executed=3')
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
    assert delegate.__func__.__name__ == 'attribute'
    assert hashlib.sha256(Path(delegate.__func__.__code__.co_filename).read_bytes()).hexdigest() == '6a112224b3d4e63d047a576d7df68ff1ee65e4bbacfcb0014f84c17bd82b8520'
    had_instance_attribute = 'attribute' in vars(runner)
    prior_instance_attribute = vars(runner).get('attribute')
    assert not hasattr(runner, '_native_register_threads_ABBA_installed')
    settings = torch.cuda.memory._snapshot()['allocator_settings']
    assert settings['pinned_use_cuda_host_register'] is True and settings['pinned_num_register_threads'] == 1
    runner._native_register_threads_ABBA_installed = True

    def observe(instance, *args, **kwargs):
        record = dict(rank=actor.rank,pid=os.getpid(),birth=psutil.Process().create_time(),
            started=time.time(),version=VERSION,candidate_deployed=False,
            additional_DT_calls_planned=3,additional_DT_calls_invoked=0,
            additional_optimizer_calls=0,additional_rollout_calls=0,
            training_returns_original_natural_result=True,
            source=__file__,source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            original_runner_source=delegate.__func__.__code__.co_filename,trials=[])
        baseline = None
        natural_started = False
        settings_changed = False
        restore_config = None
        current = None
        try:
            current = torch.cuda.memory._snapshot()['allocator_settings']
            record['original_allocator_settings'] = current
            if not (current['pinned_use_cuda_host_register'] is True and current['pinned_num_register_threads'] == 1):
                natural_started = True
                baseline = delegate(*args, **kwargs)
                record['status'] = 'optional_probe_abstained_actual_allocator_changed'
                return baseline
            restore_config = current['PYTORCH_CUDA_ALLOC_CONF'] + ',pinned_use_cuda_host_register:True,pinned_num_register_threads:1'
            candidate_config = current['PYTORCH_CUDA_ALLOC_CONF'] + ',pinned_use_cuda_host_register:True,pinned_num_register_threads:8'
            record.update(baseline_config=restore_config,candidate_config=candidate_config,
                scheduler_before=actor.actor_lr_scheduler.last_epoch,
                optimizer_entries_before=len(actor.actor_optimizer.state),
                input_shape=list((args[0] if args else kwargs['paired_ids']).shape))
            cpu_rng = torch.get_rng_state()
            gpu_rng = torch.cuda.get_rng_state()

            def call(label, additional=False):
                before = dict(torch.cuda.memory.host_memory_stats())
                if additional:
                    record['additional_DT_calls_invoked'] += 1
                value = delegate(*args, **kwargs)
                after = dict(torch.cuda.memory.host_memory_stats())
                record['trials'].append(dict(arm=label,owner_info=value[1],before_host=before,after_host=after))
                return value

            natural_started = True
            baseline = call('A1_natural')
            record['baseline_CPU_CUDA_rng_unchanged'] = bool(torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state()))
            settings_changed = True
            torch.cuda.memory._set_allocator_settings(candidate_config)
            candidate_settings = torch.cuda.memory._snapshot()['allocator_settings']
            record['candidate_allocator_settings'] = candidate_settings
            assert candidate_settings['pinned_num_register_threads'] == 8
            assert all(candidate_settings[k] == current[k] for k in current if k not in ('PYTORCH_CUDA_ALLOC_CONF','pinned_num_register_threads'))
            cold = call('B8_first', additional=True)
            warm = call('B8_warm', additional=True)
            torch.cuda.memory._set_allocator_settings(restore_config)
            repeat = call('A1_after', additional=True)
            record['raw_vector_equal'] = {name:bool(torch.equal(baseline[0],value[0])) for name,value in [('B8_first',cold),('B8_warm',warm),('A1_after',repeat)]}
            record['raw_vector_max_abs'] = {name:float((baseline[0]-value[0]).abs().max()) for name,value in [('B8_first',cold),('B8_warm',warm),('A1_after',repeat)]}
            record['CPU_CUDA_rng_unchanged_after_all'] = bool(torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state()))
            record['status'] = 'same_input_native_ABBA_completed_candidate_not_deployed'
            return baseline
        except Exception as exc:
            record.update(status='optional_register_threads_probe_failed_candidate_rejected',error_type=type(exc).__name__,error=str(exc))
            if baseline is not None:
                return baseline
            if natural_started:
                raise
            record['status'] = 'optional_probe_unavailable_original_natural_call_only'
            return delegate(*args, **kwargs)
        finally:
            if settings_changed:
                torch.cuda.memory._set_allocator_settings(restore_config)
            if had_instance_attribute:
                runner.attribute = prior_instance_attribute
            else:
                del runner.attribute
            restored = torch.cuda.memory._snapshot()['allocator_settings']
            record.update(finished=time.time(),original_callable_restored=runner.attribute == delegate,
                restored_allocator_settings=restored,scheduler_after=actor.actor_lr_scheduler.last_epoch,
                optimizer_entries_after=len(actor.actor_optimizer.state))
            record['native_allocator_semantics_restored'] = current is not None and all(restored[k] == current[k] for k in current if k != 'PYTORCH_CUDA_ALLOC_CONF')
            try:
                with (AUDIT / f'{VERSION}-rank{actor.rank}.json').open('x') as stream:
                    json.dump(record,stream,indent=2)
            except Exception as exc:
                print('Optional register-thread observation unavailable: '+type(exc).__name__,flush=True)

    runner.attribute = MethodType(observe,runner)
    return dict(pid=os.getpid(),birth=psutil.Process().create_time(),rank=actor.rank,
        installed=True,scheduler=actor.actor_lr_scheduler.last_epoch,
        original_allocator_settings=settings,additional_DT_calls_when_executed=3,
        candidate_deployed=False,scope='One original natural B4 and three same-input native DT replays; A1/B8-first/B8-warm/A1-after')


exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), 'exec'), globals())
