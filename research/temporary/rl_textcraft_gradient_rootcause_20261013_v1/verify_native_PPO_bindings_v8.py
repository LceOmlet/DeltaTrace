"""Read native PPO worker bindings and states through the existing Ray RPC."""
import json
from pathlib import Path
import time
import sys
import re

import psutil
import ray

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'


def read(worker):
    import hashlib
    import inspect
    import os
    from pathlib import Path
    import psutil
    from omegaconf import OmegaConf

    def binding(obj, methods):
        cls = type(obj)
        path = Path(inspect.getsourcefile(cls))
        return {'class': cls.__module__ + '.' + cls.__qualname__, 'path': str(path),
                'SHA256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'loaded_methods': {name: {'filename': inspect.unwrap(getattr(cls, name)).__code__.co_filename,
                                          'name': inspect.unwrap(getattr(cls, name)).__qualname__,
                                          'instance_override': name in vars(obj)}
                                   for name in methods if hasattr(cls, name)}}

    roles = []
    for name, w in worker.worker_dict.items():
        row = {'role': name, 'rank': w.rank, 'worker': binding(w, ['update_actor', 'compute_values', 'update_critic'])}
        if getattr(w, '_is_actor', False):
            row['actor'] = binding(w.actor, ['update_policy', '_forward_micro_batch', '_optimizer_step'])
            row['scheduler_last_epoch'] = w.actor_lr_scheduler.last_epoch
            row['optimizer_state_entries'] = len(w.actor_optimizer.state)
            row['configuration'] = {key: OmegaConf.select(w.config, key) for key in [
                'model.lora_rank', 'model.lora_alpha', 'actor.ppo_mini_batch_size',
                'actor.ppo_micro_batch_size_per_gpu', 'actor.ppo_epochs', 'actor.entropy_coeff',
                'rollout.max_model_len', 'rollout.enforce_eager', 'rollout.free_cache_engine']}
        if hasattr(w, 'critic'):
            row['critic'] = binding(w.critic, ['compute_values', 'update_critic', '_forward_micro_batch', '_optimizer_step'])
            row['scheduler_last_epoch'] = w.critic_lr_scheduler.last_epoch
            row['optimizer_state_entries'] = len(w.critic_optimizer.state)
            row['configuration'] = {key: OmegaConf.select(w.config, key) for key in [
                'model.lora_rank', 'model.path', 'strategy', 'ppo_mini_batch_size',
                'ppo_micro_batch_size_per_gpu', 'ppo_epochs', 'grad_clip', 'optim.lr']}
        roles.append(row)
    from verl.trainer.ppo import core_algos
    p = Path(inspect.getsourcefile(core_algos))
    return {'pid': os.getpid(), 'birth': psutil.Process().create_time(), 'roles': roles,
            'core': {'path': str(p), 'SHA256': hashlib.sha256(p.read_bytes()).hexdigest()},
            'new_model_calls': 0, 'new_backward_optimizer_scheduler_calls': 0,
            'copied_training_function_bodies': 0}


version = sys.argv[1] if len(sys.argv)>1 else 'v8'
assert re.fullmatch(r'v[0-9]+',version)
job = json.loads((OUT / f'native-PPO-control-{version}-job.json').read_bytes())
driver = psutil.Process(job['pid'])
assert driver.create_time() == job['birth'] and driver.status() != psutil.STATUS_ZOMBIE
gcs = next(p for p in driver.children(recursive=True) if p.name() == 'gcs_server')
port = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--gcs_server_port='))
host = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--node-ip-address='))
ray.init(address=host + ':' + port, log_to_driver=False)
try:
    names = [a for a in ray.util.list_named_actors(all_namespaces=True)
             if 'WorkerDict_' in a['name'] and 'register_center' not in a['name']]
    assert len(names) == 2
    refs = [ray.get_actor(a['name'], namespace=a['namespace']).execute_with_func_generator.remote(func=read)
            for a in names]
    workers = ray.get(refs, timeout=600)
    report = {'unix': time.time(), 'job': job, 'workers': workers, 'scope': __doc__,
              'status': 'actual_native_PPO_actor_critic_binding_and_state_readback',
              'note': 'Scheduler/optimizer state is sampled at RPC execution time, not necessarily initial state.'}
    dest = Path(job['run']) / f'runtime-overrides/native-PPO-bindings-{version}.json'
    dest.parent.mkdir(exist_ok=True)
    assert not dest.exists(), 'Read existing receipt rather than overwrite an earlier binding observation.'
    dest.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'receipt': str(dest), 'report': report}), flush=True)
finally:
    ray.shutdown()
