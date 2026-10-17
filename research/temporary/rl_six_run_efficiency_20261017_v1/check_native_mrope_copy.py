"""Measure the observed native strided position copy on private tiny buffers.

The existing vLLM, sampler, graph buffers and training tensors are untouched.
Both arms call torch.Tensor.copy_; no model or optimizer is executed.
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
VERSION = 'native-MRoPE-integer-copy-20261017-v1'


def check(worker):
    import hashlib
    import inspect
    import os
    from pathlib import Path
    import time
    import psutil
    import torch

    owner, = [w for w in worker.worker_dict.values() if getattr(w, '_is_actor', False)]
    runner = owner.rollout_sharding_manager.model_runner
    source = Path(inspect.getfile(type(runner)))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == 'e8a73e1ccbfff37b880cd47ba9c115db572aef3c9d4d9eee50b18b729117f9bb'
    positions = runner.mrope_positions
    assert positions.cpu.shape == positions.gpu.shape == (3, 32769)
    assert positions.cpu.is_pinned() and positions.cpu.dtype == torch.int64
    assert positions.cpu.stride() == positions.gpu.stride() == (32769, 1)
    state = (owner.actor_lr_scheduler.last_epoch, len(owner.actor_optimizer.state))
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    cpu = torch.empty_like(positions.cpu, pin_memory=True)
    cpu.copy_(torch.arange(cpu.numel(), dtype=cpu.dtype).view_as(cpu))
    baseline = torch.empty_like(positions.gpu)
    candidate = torch.empty_like(positions.gpu)

    def original(n):
        baseline[:, :n].copy_(cpu[:, :n], non_blocking=True)

    def rowwise(n):
        for axis in range(3):
            candidate[axis, :n].copy_(cpu[axis, :n], non_blocking=True)

    def measure(fn, n, repetitions):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        tick = time.perf_counter()
        for _ in range(repetitions):
            fn(n)
        enqueue = time.perf_counter()-tick
        end.record()
        end.synchronize()
        return dict(repetitions=repetitions, host_enqueue_seconds=enqueue,
            synchronized_wall_seconds=time.perf_counter()-tick,
            stream_elapsed_seconds=start.elapsed_time(end)/1000)

    started = time.time()
    cases = []
    for n in (1, 4, 8, 32, 128, 1024, 32768):
        if time.time()-started > 8:
            break
        baseline.fill_(-9)
        candidate.fill_(-9)
        for _ in range(4):
            original(n)
            rowwise(n)
        torch.cuda.synchronize()
        expected = torch.full_like(cpu, -9)
        expected[:, :n].copy_(cpu[:, :n])
        baseline_value, candidate_value = baseline.cpu(), candidate.cpu()
        case = dict(tokens=n, dtype=str(cpu.dtype), row_stride=cpu.stride(0),
            baseline_full_buffer_equal_expected=torch.equal(baseline_value, expected),
            candidate_full_buffer_equal_expected=torch.equal(candidate_value, expected),
            full_buffer_pair_equal=torch.equal(baseline_value, candidate_value), trials=[])
        for trial in range(3):
            order = ('original', 'rowwise') if trial % 2 == 0 else ('rowwise', 'original')
            result = dict(order=order)
            for name in order:
                result[name] = measure(original if name == 'original' else rowwise, n, 64)
            case['trials'].append(result)
        cases.append(case)

    cpu_profiles = []
    if owner.rank == 0:
        for name, function in [('original', original), ('rowwise', rowwise)]:
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                             record_shapes=True) as profiler:
                    function(32)
                torch.cuda.synchronize()
                cpu_profiles.append(dict(arm=name, activities=['CPU'], model_calls=0,
                    operations=[dict(name=op.key, calls=op.count, cpu_time_total_us=op.cpu_time_total,
                                     input_shapes=op.input_shapes)
                                for op in profiler.key_averages(group_by_input_shape=True)]))
            except Exception as exc:
                cpu_profiles.append(dict(arm=name, status='optional_CPU_operator_profile_unavailable',
                    error_type=type(exc).__name__, message=str(exc)[:400]))

    assert state == (owner.actor_lr_scheduler.last_epoch, len(owner.actor_optimizer.state))
    rng_unchanged = torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
    result = dict(collected=time.time(), started=started, pid=os.getpid(), birth=psutil.Process().create_time(),
        rank=owner.rank, status='private_native_integer_copy_cases_completed_candidate_not_deployed',
        copied_buffer_shape=list(cpu.shape), cases=cases, CPU_operator_profiles=cpu_profiles,
        all_planned_sizes_completed=len(cases)==7, RNG_unchanged=rng_unchanged,
        scheduler_epoch=state[0], optimizer_entries=state[1], native_vllm_source=str(source),
        native_vllm_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),
        training_vllm_configuration_or_tensor_changes=0, model_calls=0, optimizer_updates=0,
        timing_scope='Private exact-layout copy completion, including CPU enqueue and device waiting. Not a full vLLM generation speed comparison.',
        capacity_scope='Approximately2.4MiB scratch host/device buffers per worker; not a model32768 capacity test')
    (AUDIT / f'{VERSION}-rank{owner.rank}.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    source = json.loads((RUN / 'source.json').read_bytes())
    if '--client' not in sys.argv:
        assert not (AUDIT / (VERSION+'-job.json')).exists()
        env = dict(os.environ, **source['effective_environment'])
        env['CUDA_VISIBLE_DEVICES'] = ''
        with (AUDIT / (VERSION+'.stdout')).open('xb') as out, (AUDIT / (VERSION+'.stderr')).open('xb') as err:
            child = subprocess.Popen([env['VENV_PYTHON'], __file__, '--client'], env=env,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
        record = dict(pid=child.pid, birth=psutil.Process(child.pid).create_time(), submitted=time.time(),
            source=__file__, source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            status='bounded_private_native_copy_check_pending_original_worker_boundary')
        (AUDIT / (VERSION+'-job.json')).write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record))
    else:
        import ray
        driver = psutil.Process(source['pid'])
        assert driver.create_time() == source['birth']
        gcs = next(p for p in driver.children(recursive=True) if p.name() == 'gcs_server')
        host = next(v.split('=',1)[1] for v in gcs.cmdline() if v.startswith('--node-ip-address='))
        port = next(v.split('=',1)[1] for v in gcs.cmdline() if v.startswith('--gcs_server_port='))
        expected = {r['pid']:r['birth'] for r in json.loads((RUN /
            'runtime-overrides/actual-worker-imports-after-first-DT-v1.json').read_bytes())['results']}
        ray.init(address=host+':'+port, log_to_driver=False)
        try:
            names = [a for a in ray.util.list_named_actors(all_namespaces=True)
                if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
            assert len(names) == 8
            rows = ray.get([ray.get_actor(a['name'], namespace=a['namespace']).execute_with_func_generator.remote(func=check)
                for a in names])
            assert all(expected[r['pid']] == r['birth'] for r in rows)
            (AUDIT / (VERSION+'-completed.json')).write_text(json.dumps(dict(collected=time.time(),
                status='eight_rank_private_native_position_copy_comparison_completed_candidate_not_deployed',
                source=__file__, source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                results=rows), indent=2)+'\n')
        finally:
            ray.shutdown()
