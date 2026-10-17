"""Measure the original DT owner and a resource-only candidate on GPU6.

This is a bounded, same-state B4/32768 resource comparison, not a training run.
Reuse the already executed native verification worker and its original fixture.
No model, credit, optimizer, parser, or checkpoint behavior is implemented here.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/dt-native-replay-host-lifetime-20261017-v3'
HARNESS = ROOT / 'receipts/textcraft-reward-target-exponential-20261016-v5/source/run_native_behavior_v5.py'
HARNESS_SHA = 'ae4ba91b1978d572c17a5ab4589c0d2804be1eefb8d59d1de308f17d22815541'


def original_harness():
    assert hashlib.sha256(HARNESS.read_bytes()).hexdigest() == HARNESS_SHA
    spec = importlib.util.spec_from_file_location('original_native_DT_behavior_v5', HARNESS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if '--client' not in sys.argv:
    import subprocess
    source = json.loads((ROOT / 'runs/appworld-dtpo-reward-target-exponential-variance-formal-20261017-v6/source.json').read_bytes())
    env = dict(os.environ)
    env.update(source['effective_environment'])
    old = env['DT_ROOT']
    candidate = str(ROOT / 'candidates/dt-native-replay-host-lifetime-20261017-v3/deltatrace')
    env['DT_ROOT'] = candidate
    env['PYTHONPATH'] = env['PYTHONPATH'].replace(old, candidate)
    env['CUDA_VISIBLE_DEVICES'] = '6'
    env.pop('MACA_VISIBLE_DEVICES', None)
    result = subprocess.run([env['VENV_PYTHON'], '-X', 'utf8', __file__, '--client', *sys.argv[1:]],
                            env=env, capture_output=True, text=True)
    (AUDIT / 'client.stdout').write_text(result.stdout)
    (AUDIT / 'client.stderr').write_text(result.stderr)
    print(result.stdout, end='')
    if result.returncode:
        print(result.stderr, file=sys.stderr, end='')
    raise SystemExit(result.returncode)

import psutil
import ray
import torch
from omegaconf import OmegaConf
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.native_host_memory import host_memory_lease


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@ray.remote
class ResourceWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def verify_resource(self, data):
        import reward_readout
        harness = original_harness()
        harness.AUDIT = AUDIT
        harness.base.AUDIT = AUDIT
        original = harness.BehaviorWorker.__ray_metadata__.modified_class.verify_credit
        stage = data.meta_info['behavior_stage']
        previous_flag = os.environ.get('DT_NATIVE_HOST_CACHE_REPLAY_LAYER')
        os.environ['DT_NATIVE_HOST_CACHE_REPLAY_LAYER'] = data.meta_info['resource_release_flag']
        native_trace = reward_readout.trace_token_attribution
        vectors = []

        def record_native_vector(*args, **kwargs):
            signed, magnitude, detail = native_trace(*args, **kwargs)
            vectors.append(signed.detach().cpu().clone())
            return signed, magnitude, detail

        reward_readout.trace_token_attribution = record_native_vector
        try:
            # Equalize inactive pinned-cache state, using the same already
            # accepted native API; persistent compiler/model caches stay intact.
            torch._C._host_emptyCache()
            torch.cuda.memory.reset_peak_host_memory_stats()
            before_host = dict(torch.cuda.memory.host_memory_stats())
            before_physical = {
                'collected': time.time(),
                'MemAvailable_KiB': int(next(x.split()[1] for x in
                    Path('/proc/meminfo').read_text().splitlines()
                    if x.startswith('MemAvailable:'))),
                'XTT_single_pool_KiB': int(Path(
                    '/sys/class/drm/card1/device/mem_info_xtt_used').read_text()),
            }
            started = time.monotonic()
            result = original(self, data)
            torch.save(vectors, AUDIT / f'{stage}-vectors-rank{self.rank}.pt')
            write(AUDIT / f'{stage}-native-resource-rank{self.rank}.json', dict(
                pid=os.getpid(), birth=psutil.Process().create_time(),
                before_host=before_host, after_host=dict(torch.cuda.memory.host_memory_stats()),
                before_physical=before_physical, finished=time.time(),
                native_RPC_and_verification_seconds=time.monotonic()-started,
                PSS_bytes=psutil.Process().memory_full_info().pss,
                RSS_bytes=psutil.Process().memory_info().rss,
                native_allocated_current_is_not_physical_RAM=True,
                optimizer_updates=0, rollout_generation_calls=0))
            return result
        finally:
            reward_readout.trace_token_attribution = native_trace
            if previous_flag is None:
                os.environ.pop('DT_NATIVE_HOST_CACHE_REPLAY_LAYER', None)
            else:
                os.environ['DT_NATIVE_HOST_CACHE_REPLAY_LAYER'] = previous_flag


if __name__ == '__main__':
    from transformers import AutoTokenizer
    import fcntl
    source = json.loads((ROOT / 'runs/appworld-dtpo-reward-target-exponential-variance-formal-20261017-v6/source.json').read_bytes())
    config_path = ROOT / 'receipts/dtpo-exponential-credit-variance-20261017-v6/AppWorld_DTPO-effective-config.yaml'
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == source['effective_config_SHA256']
    config = OmegaConf.load(config_path)
    config.actor_rollout_ref.actor.optim.total_training_steps = source['budget']['iterations']
    harness = original_harness()
    tokenizer = AutoTokenizer.from_pretrained(os.environ['MODEL_PATH'], local_files_only=True)
    data = harness.base.requests(tokenizer, capacity=True).chunk(2)[0]
    assert len(data) == 4
    assert data.batch['attention_mask'].sum(-1).tolist() == [32768] * 4
    assert callable(harness.BehaviorWorker.__ray_metadata__.modified_class.verify_credit)
    if '--preflight' in sys.argv:
        assert not torch.cuda.is_initialized()
        print(json.dumps(dict(status='CPU_preflight_passed_no_native_worker_started',
            fixture_rows=len(data), original_context_lengths=data.batch['attention_mask'].sum(-1).tolist(),
            original_harness_SHA256=HARNESS_SHA, config_SHA256=source['effective_config_SHA256'],
            candidate_DT_ROOT=os.environ['DT_ROOT'], CUDA_initialized=False,
            original_actor_verification_method_reused=True)))
        raise SystemExit(0)
    lock = open(AUDIT / 'resource-comparison-single-execution.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = AUDIT / 'native-comparison-job.json'
    if state_path.exists():
        raise RuntimeError('Preserve the prior execution receipt instead of relaunching it')
    state = dict(status='starting', pid=os.getpid(), birth=psutil.Process().create_time(),
                 physical_gpus=[6], world_size=1, B_per_rank=4, context=32768,
                 optimizer_updates=0, rollout_generation_calls=0, checkpoint_calls=0,
                 full_two_rank_capacity_verified=False, four_resident_safety_verified=False)
    write(state_path, state)
    state['input_SHA256'] = hashlib.sha256(data.batch['input_ids'].numpy().tobytes()).hexdigest()
    env = {k: v for k, v in os.environ.items() if k.startswith(('DT_', 'VERL_', 'MACA_'))
           or k in ('PYTHONPATH', 'LD_LIBRARY_PATH', 'PATH', 'VENV_PYTHON', 'MODEL_PATH',
                    'HF_HUB_OFFLINE', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR')}
    env.pop('CUDA_VISIBLE_DEVICES', None)
    try:
        ray.init(address='local', num_cpus=8, num_gpus=1, include_dashboard=False,
                 _temp_dir='/tmp/ray-DT-host-lifetime-v3',
                 runtime_env=dict(env_vars=env, worker_process_setup_hook='metax_fsdp_event_wait.worker_setup'))
        group = RayWorkerGroup(RayResourcePool([1], use_gpu=True, max_colocate_count=1),
                               RayClassWithInitArgs(ResourceWorker, config.actor_rollout_ref, 'actor'))
        state['status'] = 'waiting_existing_resource_lease_for_bounded_comparison'
        write(state_path, state)
        # This diagnostic uses the already deployed admission boundary until its
        # actual peak is known. This does not introduce another training lock.
        with host_memory_lease('bounded_DT_native_host_lifetime_comparison'):
            state.update(status='native_actor_initializing', admitted=time.time())
            write(state_path, state)
            group.init_model()
            for stage, flag in [('baseline-warmup-32768', '0'),
                                ('baseline-warm-32768', '0'),
                                ('candidate-warm-32768', '1')]:
                data.meta_info.update(behavior_stage=stage, resource_release_flag=flag)
                state.update(status=stage, stage_started=time.time())
                write(state_path, state)
                group.verify_resource(data)
                state.setdefault('completed_stages', []).append(stage)
                write(state_path, state)
        baseline = torch.load(AUDIT / 'baseline-warm-32768-vectors-rank0.pt',
                              map_location='cpu', weights_only=False)
        candidate = torch.load(AUDIT / 'candidate-warm-32768-vectors-rank0.pt',
                               map_location='cpu', weights_only=False)
        assert len(baseline) == len(candidate)
        state.update(status='completed', finished=time.time(),
                     native_DT_vectors_equal=[torch.equal(a, b) for a, b in zip(baseline, candidate)],
                     scope='Single native B4 worker; two-rank/four-resident capacity remains unproven')
        write(state_path, state)
        print(json.dumps(state))
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), failed=time.time())
        write(state_path, state)
        raise
    finally:
        ray.shutdown()
