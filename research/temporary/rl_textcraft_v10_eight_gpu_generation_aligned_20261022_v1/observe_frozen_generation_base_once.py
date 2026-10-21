"""Read existing frozen vLLM parameters once after an ordinary native sync.

The old, verified tensor hash functions and public vLLM apply_model own the
read. This callback performs no forward, generation, DT or optimizer call.
The original sync runs once, and the callback restores it before observation.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from prepare_campaign import ROOT, AUDIT, GEN, VERSION, sha


def install_once(worker):
    import hashlib
    import json
    import os
    from pathlib import Path
    import psutil
    from types import MethodType
    import time

    # Exact role selection used by the original inspect_worker metadata owner.
    actors = [item for item in worker.worker_dict.values() if getattr(item, '_is_actor', False)]
    assert len(actors) == 1
    worker = actors[0]
    directory = Path(OUTPUT)
    filename = directory / f'rank{worker.rank}.json'
    marker = '_formal_frozen_generation_base_read_v2'
    if hasattr(worker, marker):
        return dict(rank=worker.rank, state=getattr(worker, marker), already_installed=True)
    state = dict(status='waiting_next_original_natural_sync')
    setattr(worker, marker, state)
    manager = worker.rollout_sharding_manager
    original_update = manager.update_params

    def read_after_original_sync(instance, *args, **kwargs):
        try:
            answer = original_update(*args, **kwargs)
        finally:
            # Original sync exceptions propagate normally. No retry or fallback.
            instance.update_params = original_update
        observed = dict(rank=worker.rank, pid=os.getpid(), birth=psutil.Process().create_time(),
            role=worker.role, started_unix=time.time(), original_sync_restored=True,
            model_forward_calls=0, additional_generate_calls=0, DT_calls=0, optimizer_calls=0,
            owner_hash_source=HASH_OWNER, owner_hash_source_SHA256=HASH_OWNER_SHA,
            observer_source=OBSERVER_SOURCE, observer_source_SHA256=OBSERVER_SHA,
            baseline_source=BASELINE_SOURCE, baseline_source_SHA256=BASELINE_SHA,
            timing_scope='One passive CPU transfer/hash observation during native sync; not a pure operator timing sample')
        try:
            actual = instance.inference_engine.apply_model(inspect_live_engine)
            observed['live_vllm_parameters'] = actual
            observed['baseline_parameter_counts'] = [len(x) for x in BASELINE]
            observed['actual_parameter_counts'] = [len(x) for x in actual]
            observed['different_base_parameter_keys'] = [
                sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
                for a, b in zip(actual, BASELINE)]
            observed['all_base_parameters_exactly_match_baseline'] = (
                len(actual) == len(BASELINE) and not any(observed['different_base_parameter_keys']))
            trainable = [name for name, parameter in instance.module.named_parameters() if parameter.requires_grad]
            observed['trainable_parameter_count'] = len(trainable)
            observed['trainable_non_LoRA_parameter_names'] = [name for name in trainable if 'lora_' not in name]
            steps = sorted({float(value['step']) for value in worker.actor_optimizer.state.values() if 'step' in value})
            observed['actual_optimizer_state_step_values'] = steps
            observed['status'] = 'completed_passive_existing_frozen_base_read'
        except Exception as error:
            # The owner result remains valid when optional measurement is absent.
            observed.update(status='optional_frozen_base_measurement_unavailable',
                error_type=type(error).__name__, error=str(error))
        observed['finished_unix'] = time.time()
        state.update(status=observed['status'], receipt=str(filename))
        try:
            filename.write_text(json.dumps(observed, indent=2) + '\n')
        except Exception as error:
            state.update(status='optional_observation_receipt_unavailable', error_type=type(error).__name__)
        return answer

    manager.update_params = MethodType(read_after_original_sync, manager)
    return dict(rank=worker.rank, pid=os.getpid(), birth=psutil.Process().create_time(), state=state,
                role=worker.role, model_forward_calls=0, DT_calls=0, optimizer_calls=0)


if __name__ == '__main__':
    source_job = json.loads((AUDIT / 'campaign-state.json').read_text())['jobs'][0]
    assert source_job['version'] == VERSION
    if '--client' not in sys.argv:
        import subprocess
        environment = dict(os.environ)
        environment.update(source_job['effective_environment'])
        environment['CUDA_VISIBLE_DEVICES'] = ''
        environment.pop('MACA_VISIBLE_DEVICES', None)
        raise SystemExit(subprocess.call([environment['VENV_PYTHON'], '-X', 'utf8', __file__, '--client'], env=environment))
    import psutil
    import ray
    assert psutil.Process(source_job['pid']).create_time() == source_job['birth']
    OUTPUT = str(AUDIT / 'passive-active-frozen-base-read-v2')
    Path(OUTPUT).mkdir(exist_ok=False)
    HASH_OWNER = str(GEN / 'inspect_native_initial_generation.py')
    HASH_OWNER_SHA = 'c7d26dfc1e8384206c151766f96cee7a0da493c2234b7884ce14bd221550a449'
    assert sha(HASH_OWNER) == HASH_OWNER_SHA
    tree = ast.parse(Path(HASH_OWNER).read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('tensor_record', 'inspect_live_engine')]
    assert len(functions) == 2
    namespace = dict(__name__='__main__', hashlib=hashlib)
    exec(compile(ast.Module(body=functions, type_ignores=[]), HASH_OWNER, 'exec'), namespace)
    inspect_live_engine = namespace['inspect_live_engine']
    BASELINE_SOURCE = str(GEN / 'initial-sync-DP2-v1/rank0/sync-observation.json')
    BASELINE_SHA = sha(BASELINE_SOURCE)
    BASELINE = json.loads(Path(BASELINE_SOURCE).read_text())['syncs'][0]['live_vllm_parameters']
    assert [len(x) for x in BASELINE] == [379]
    OBSERVER_SOURCE = __file__
    OBSERVER_SHA = sha(__file__)
    driver = psutil.Process(source_job['pid'])
    gcs = next(p for p in driver.children(recursive=True) if p.name() == 'gcs_server')
    port = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
    host = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
    ray.init(address=host + ':' + port, log_to_driver=False)
    try:
        named = [x for x in ray.util.list_named_actors(all_namespaces=True)
                 if 'WorkerDict_' in x['name'] and 'register_center' not in x['name']]
        assert len(named) == 8
        results = ray.get([ray.get_actor(x['name'], namespace=x['namespace']).execute_with_func_generator.remote(func=install_once)
                           for x in named])
        record = dict(status='passive_once_per_rank_frozen_base_read_queued_on_existing_natural_sync',
            submitted_unix=time.time(), source_job=dict(pid=source_job['pid'], birth=source_job['birth']),
            source=OBSERVER_SOURCE, SHA256=OBSERVER_SHA, results=results,
            additional_model_forward_calls=0, additional_generate_calls=0, DT_calls=0, optimizer_calls=0,
            scope=__doc__)
        (AUDIT / 'passive-frozen-base-read-request.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record))
    finally:
        ray.shutdown()
