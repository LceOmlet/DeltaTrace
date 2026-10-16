"""Read actual worker imports/configuration through the existing VERL RPC."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/appworld-native-ppo-speed-20261016-v1'
task = sys.argv[1]
assert task in ('SQL', 'AppWorld')
job = json.loads((ROOT / 'runs/appworld-native-ppo-speed-20261016-v1' / task.lower() / 'job.json').read_bytes())
prepared = json.loads(Path(job['prepared']).read_bytes())
if '--client' not in sys.argv:
    env = dict(os.environ)
    env.update(prepared['effective_environment'])
    env['CUDA_VISIBLE_DEVICES'] = ''
    env.pop('MACA_VISIBLE_DEVICES', None)
    raise SystemExit(subprocess.call([env['VENV_PYTHON'], '-X', 'utf8', __file__, task, '--client'], env=env))

import psutil
import ray

BINDINGS = job['source_bindings']
def inspect_worker(worker):
    import hashlib
    import os
    from pathlib import Path
    import sys
    import psutil
    actors = [item for item in worker.worker_dict.values() if getattr(item, '_is_actor', False)]
    assert len(actors) == 1
    actor = actors[0]
    modules = {}
    names = ['dt_training_batch', 'reward_readout', 'counterfactual', 'deltatrace_rollout',
             'owner_trajectory_batch', 'sql_owner_rollout', 'loop_owner_rollout',
             'verl.workers.fsdp_workers', 'verl.workers.actor.dp_actor', 'verl.trainer.ppo.core_algos',
             'verl.utils.experimental.torch_functional', 'verl.models.transformers.qwen3_vl',
             'verl.workers.critic.dp_critic']
    names += [n for n in sys.modules if n.rsplit('.', 1)[-1] in
              ('qwen35_gdn_finite', 'qwen35_answer_finite', 'qwen35_retained_capture', 'qwen35_dense_finite_runner')]
    for name in dict.fromkeys(names):
        module = sys.modules.get(name)
        path = getattr(module, '__file__', None)
        if path and Path(path).is_file():
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            expected = BINDINGS.get(path)
            modules[name] = dict(path=path, SHA256=digest, expected=expected,
                                 matches_prepared_source=None if expected is None else digest == expected)
        else:
            modules[name] = dict(imported=False)
    roles = {name:type(item).__module__ + '.' + type(item).__name__ for name,item in worker.worker_dict.items()}
    # Original colocated role registration, rather than an invented _is_critic flag.
    critic = worker.worker_dict.get('critic')
    critic_info = dict(roles=roles, initialized=False)
    if critic is not None:
        native_critic = getattr(critic, 'critic', None)
        critic_info.update(initialized=native_critic is not None,
            type=type(native_critic).__module__ + '.' + type(native_critic).__name__,
            epochs=critic.config.ppo_epochs, minibatch=critic.config.ppo_mini_batch_size,
            microbatch=critic.config.ppo_micro_batch_size_per_gpu, model_path=critic.config.model.path)
    return dict(pid=os.getpid(), birth=psutil.Process().create_time(), rank=actor.rank,
                DT_CREDIT_MODE=os.environ.get('DT_CREDIT_MODE'), DT_ROOT=os.environ.get('DT_ROOT'),
                VERL_ROOT=os.environ.get('VERL_ROOT'), VENV_PYTHON=os.environ.get('VENV_PYTHON'),
                lora_rank=actor.config.model.lora_rank, lora_alpha=actor.config.model.lora_alpha,
                microbatch_per_gpu=actor.config.actor.ppo_micro_batch_size_per_gpu,
                worker_minibatch=actor.config.actor.ppo_mini_batch_size,
                enforce_eager=actor.config.rollout.enforce_eager,
                free_cache_engine=actor.config.rollout.free_cache_engine, modules=modules, native_critic=critic_info,
                model_calls=0, DT_calls=0, optimizer_updates=0, method_changes=0)

driver = psutil.Process(job['pid'])
assert driver.create_time() == job['birth']
gcs = next(p for p in driver.children(recursive=True) if p.name() == 'gcs_server')
port = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
host = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
ray.init(address=host + ':' + port, log_to_driver=False)
try:
    named = [x for x in ray.util.list_named_actors(all_namespaces=True)
             if 'WorkerDict_' in x['name'] and 'register_center' not in x['name']]
    assert len(named) == 2
    results = ray.get([ray.get_actor(x['name'], namespace=x['namespace']).execute_with_func_generator.remote(func=inspect_worker)
                       for x in named])
    assert all(m.get('matches_prepared_source') is not False for r in results for m in r['modules'].values())
    receipt = AUDIT / ('actual-worker-imports-' + task.lower() + '.json')
    record = dict(task=task, source_commit=job['source_commit'], collected_unix=time.time(),
                  script_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  job=dict(pid=job['pid'], birth=job['birth']), results=results,
                  scope='Read-only native VERL worker RPC; absent lazy modules explicitly remain unverified')
    record['metadata_attempt1'] = 'Read-only client incorrectly assumed a native _is_critic flag; assertion failed without model/DT/optimizer calls. Corrected to original worker_dict critic role.'
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(receipt=str(receipt), results=results)))
finally:
    ray.shutdown()
