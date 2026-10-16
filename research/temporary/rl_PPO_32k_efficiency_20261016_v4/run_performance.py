"""Measure the changed short-input dispatch through the unchanged native PPO owner."""
import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT/'receipts/PPO-32k-native-efficiency-20261016-v4/performance'
CANDIDATE = ROOT/'candidates/PPO-32k-native-efficiency-20261016-v4/verl-owner'
if '--native' not in sys.argv:
    import re
    p = json.loads((ROOT/'candidates/appworld-native-ppo-speed-20261016-v1/prepared.json').read_bytes())
    smi = subprocess.check_output(['mx-smi'], text=True)
    assert not any(re.match(r'\|\s*(?:6|7)\s+\d+\s+\S+', line) for line in smi.splitlines()), 'Physical6/7 occupied'
    env = dict(os.environ); env.update(p['effective_environment'])
    env['VERL_ROOT'] = str(CANDIDATE); env['PYTHONPATH'] = str(CANDIDATE)+':'+env['PYTHONPATH']
    env['CUDA_VISIBLE_DEVICES'] = '6,7'; env['FLASH_ATTENTION_DETERMINISTIC'] = '0'
    env['DT_FSDP_EVENT_WAIT_RECEIPTS'] = str(AUDIT/'worker-runtime')
    env.pop('MACA_VISIBLE_DEVICES', None)
    raise SystemExit(subprocess.call([env['VENV_PYTHON'], __file__, '--native'], env=env))

import numpy as np
import pickle
import psutil
import random
import ray
import statistics
import torch
from omegaconf import OmegaConf
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import CriticWorker

helper = ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/source/run_same_session.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest() == 'f4a22d506ba958a0259ab39b6407852269d42438bc5614eb5fa2b71352f8bd09'
tree = ast.parse(helper.read_text())
helpers = [v for v in tree.body if isinstance(v, ast.FunctionDef) and v.name in ('write', 'local', 'fingerprint', 'state')]
assert len(helpers) == 4
exec(compile(ast.Module(body=helpers, type_ignores=[]), str(helper), 'exec'), globals())

@ray.remote
class PerformanceCritic(CriticWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self):
        random.seed(20261016); np.random.seed(20261016); torch.manual_seed(20261016)
        super().init_model()
        self._plain = {id(layer): (layer, layer._gradient_checkpointing_func)
                       for layer in self.critic_module.model.language_model.layers}
        self._async = None
        return dict(rank=self.rank, pid=os.getpid(), birth=psutil.Process().create_time(),
                    physical_CVD=os.environ['CUDA_VISIBLE_DEVICES'], FA_deterministic=os.environ['FLASH_ATTENTION_DETERMINISTIC'],
                    sources={name: dict(path=sys.modules[name].__file__, SHA256=hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest())
                             for name in ('verl.workers.fsdp_workers', 'verl.workers.critic.dp_critic',
                                          'verl.trainer.ppo.core_algos', 'verl.utils.activation_offload')})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def memory_mode(self, mode):
        if mode == 'adaptive' and self._async is None:
            from verl.utils.activation_offload import enable_activation_offloading
            enable_activation_offloading(self.critic_module.model.language_model, 'fsdp2', True,
                                         preserve_hf_checkpoint=True, native_checkpoint_max_seq_len=8192)
            self._async = {id(layer): layer._gradient_checkpointing_func
                           for layer in self.critic_module.model.language_model.layers}
        for key, (layer, fn) in self._plain.items():
            layer._gradient_checkpointing_func = self._async[key] if mode == 'adaptive' else fn

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def restore_probe(self, full=False):
        # The native checkpoint owns every restore. A full digest is measured
        # once per mode; later calls use a bounded restore/RNG/step probe.
        if full:
            return dict(rank=self.rank, full_SHA256=hashlib.sha256(json.dumps(state(self), sort_keys=True).encode()).hexdigest())
        optimizer = self.critic_optimizer.state_dict()
        return dict(rank=self.rank, scheduler=self.critic_lr_scheduler.state_dict(),
                    steps={str(k): float(v['step']) for k, v in optimizer['state'].items()},
                    RNG=dict(torch=fingerprint(torch.get_rng_state()), cuda=[fingerprint(v) for v in torch.cuda.get_rng_state_all()],
                             python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
                             numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest()))

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def verify_native_dispatch(self, data):
        fn = next(iter(self._async.values()))
        closure = dict(zip(fn.__code__.co_freevars, [c.cell_contents for c in fn.__closure__]))
        owner = closure['handler']._offload_ctx.offload_handler
        push, pop = owner.tensor_push, owner.tensor_pop
        counts = dict(push=0, pop=0)
        def counted_push(*args, **kwargs):
            counts['push'] += 1
            return push(*args, **kwargs)
        def counted_pop(*args, **kwargs):
            counts['pop'] += 1
            return pop(*args, **kwargs)
        owner.tensor_push, owner.tensor_pop = counted_push, counted_pop
        try:
            result = super().update_critic(data)
        finally:
            owner.tensor_push, owner.tensor_pop = push, pop
        assert counts == dict(push=0, pop=0), counts
        write(AUDIT/f'native-short-dispatch-rank{self.rank}.json', dict(status='passed', counts=counts,
              criterion='Original HF checkpoint called directly; zero native offload save/restore callbacks', observer_removed=True))
        return result

AUDIT.mkdir(parents=True, exist_ok=True)
prepared = json.loads((ROOT/'candidates/appworld-native-ppo-speed-20261016-v1/prepared.json').read_bytes())
cfg = OmegaConf.load(CANDIDATE/'verl/trainer/config/ppo_trainer.yaml')
for key, value in prepared['options'].items():
    OmegaConf.update(cfg, key.lstrip('+'), value, force_add=True)
cfg.critic.model.enable_activation_offload = False
cfg.critic.optim.total_training_steps = 2
data = torch.load(ROOT/'receipts/PPO-32k-native-efficiency-20261016-v1/baseline-fit/native-input-with-GAE.pt', weights_only=False, map_location='cpu')
checkpoint = ROOT/'receipts/PPO-32k-native-efficiency-20261016-v2/performance-exec2/numerical-warm-state-original-VERL'
status = dict(status='initializing', started=time.time(), source_commit=os.environ['PPO_32K_SOURCE_COMMIT'],
              physical_gpus=[6,7], pid=os.getpid(), birth=psutil.Process().create_time(),
              checkpoint=str(checkpoint), formal_mutations=0, model='Qwen3.5-9B', batch_per_rank=4,
              FA_deterministic=False, scope=__doc__, full_digest_once_per_mode=True,
              every_call_native_full_restore=True, checkpoint_create_calls=0,
              long_input_receipts_reused='Unchanged32768 branch; v2 numeric-reference-v3, exact-transport and performance-exec2')
write(AUDIT/'status.json', status)
env = dict(os.environ); env.pop('CUDA_VISIBLE_DEVICES', None)
ray.init(address='local', num_cpus=8, num_gpus=2, include_dashboard=False,
         _temp_dir='/tmp/ray-PPO32k-native-fast-dispatch-v4',
         runtime_env=dict(env_vars=env, worker_process_setup_hook='metax_fsdp_event_wait.worker_setup'))
try:
    group = RayWorkerGroup(RayResourcePool([2], use_gpu=True, max_colocate_count=1), RayClassWithInitArgs(PerformanceCritic, cfg.critic))
    status['workers'] = group.initialize()
    group.memory_mode('adaptive')
    status['status'] = 'verify_native_short_dispatch'; write(AUDIT/'status.json', status)
    group.load_checkpoint(str(checkpoint), del_local_after_load=False)
    baseline_probe = group.restore_probe()
    group.verify_native_dispatch(data)
    status['native_short_dispatch'] = 'passed'
    measurements = []; digests = {}
    for mode in ['plain','adaptive','adaptive','plain','plain','adaptive']:
        status['status'] = f'timing_{mode}_{len(measurements)+1}'; write(AUDIT/'status.json', status)
        group.memory_mode(mode)
        group.load_checkpoint(str(checkpoint), del_local_after_load=False)
        assert group.restore_probe() == baseline_probe
        full = mode not in digests
        if full:
            digests[mode] = group.restore_probe(full=True)
            assert all(v == next(iter(digests.values())) for v in digests.values())
        start = time.perf_counter(); result = group.update_critic(data)
        measurements.append(dict(mode=mode, seconds=time.perf_counter()-start,
                                 native_metrics=result.meta_info['metrics'], restore_probe_exact=True,
                                 fresh_full_digest_verified=full))
        status['measurements'] = measurements; write(AUDIT/'status.json', status)
    a = [v['seconds'] for v in measurements if v['mode']=='plain']
    b = [v['seconds'] for v in measurements if v['mode']=='adaptive']
    status['comparison'] = dict(context=8192, reference_seconds=a, adaptive_seconds=b,
                                median_ratio=statistics.median(b)/statistics.median(a))
    status.update(status='completed', finished=time.time()); write(AUDIT/'status.json', status)
except BaseException as exc:
    status.update(status='failed', error=repr(exc), finished=time.time()); write(AUDIT/'status.json', status)
    raise
finally:
    ray.shutdown()
print(json.dumps(status, indent=2))
