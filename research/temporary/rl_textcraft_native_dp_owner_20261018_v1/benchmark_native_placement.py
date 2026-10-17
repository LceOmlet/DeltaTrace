"""Bounded original TaskRunner generation ABBA, without DT or optimizer calls.

Use the original trainer's first real batch and its complete original owner
rollouts. A calls the former driver placement; B places that same owner loop
inside native DP workers. Official greedy mode controls sampling for this
diagnostic. It is not natural-training throughput, a quality test or a formal
iteration. No precision threshold is introduced; exact output parity is
reported separately from the timing, and must hold for a matched speed claim.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/textcraft-native-dp-owner-20261018-v1/native-benchmark'
CANDIDATE = ROOT / 'candidates/textcraft-native-dp-owner-20261018-v1'
BASE_RUN = ROOT / 'runs/textcraft-dtpo-v6-8gpu-200-20261017-v3'

if '--native' not in sys.argv:
    source = json.loads((BASE_RUN / 'source.json').read_bytes())
    env = dict(os.environ, **source['effective_environment'])
    env['VERL_ROOT'] = str(CANDIDATE / 'verl-owner')
    env['PYTHONPATH'] = str(CANDIDATE / 'entry')+':'+env['VERL_ROOT']+':'+env['PYTHONPATH']
    env['CUDA_VISIBLE_DEVICES'] = '' if '--preflight' in sys.argv else '0,1,2,3,4,5,6,7'
    env['DT_FSDP_EVENT_WAIT_RECEIPTS'] = str(AUDIT / 'worker-runtime')
    env.pop('MACA_VISIBLE_DEVICES', None)
    assert env['VENV_PYTHON'] == '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python'
    if '--preflight' not in sys.argv:
        import psutil
        try:
            driver = psutil.Process(source['pid'])
            assert driver.create_time() != source['birth'], 'Current tuning job must first be stopped through its original owner'
        except psutil.NoSuchProcess:
            pass
    argv = [env['VENV_PYTHON'], '-X', 'utf8', __file__, '--native']
    if '--preflight' in sys.argv:
        argv.append('--preflight')
    raise SystemExit(subprocess.call(argv, env=env))

import psutil
import ray
import torch
from omegaconf import OmegaConf
from unittest.mock import patch
from verl.trainer import main_ppo
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector

AUDIT.mkdir(parents=True, exist_ok=True)
cfg = OmegaConf.load(ROOT / 'receipts/textcraft-eight-gpu-sequential-20261017-v3/dt-effective-config.yaml')
assert cfg.trainer.n_gpus_per_node == 8 and cfg.trainer.total_training_steps == 200
assert cfg.data.train_batch_size == 32 and cfg.env.rollout.n == 8
assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 64
assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 4
assert cfg.actor_rollout_ref.rollout.max_model_len == 32768
assert not cfg.trainer.val_before_train and cfg.trainer.resume_mode == 'disable'
cfg.trainer.experiment_name = 'native-DP-owner-generation-ABBA-20261018-v1'
cfg.trainer.default_local_dir = str(AUDIT / 'unused-checkpoints')
cfg.ray_init.runtime_env.env_vars.PYTHONPATH = os.environ['PYTHONPATH']
cfg.ray_init.runtime_env.env_vars.VERL_ROOT = os.environ['VERL_ROOT']
cfg.ray_init.runtime_env.env_vars.DT_FSDP_EVENT_WAIT_RECEIPTS = os.environ['DT_FSDP_EVENT_WAIT_RECEIPTS']
state = dict(status='prepared_native_generation_ABBA', started=time.time(), pid=os.getpid(),
    birth=psutil.Process().create_time(), source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    source_commit=os.environ.get('NATIVE_DP_BENCH_COMMIT'), physical_gpus=list(range(8)),
    formal_iterations=0, optimizer_updates=0, DT_calls=0, checkpoint_create_calls=0,
    control='Original native do_sample=False; same first original trainer batch,32groups x8,complete original max30 owner loop',
    effective_configuration=OmegaConf.to_container(cfg, resolve=True), cases=[])


def save():
    p = AUDIT / 'status.json'
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2, allow_nan=False)+'\n')
    temp.replace(p)


def worker_state(wrapper):
    import hashlib
    import inspect
    import os
    from pathlib import Path
    import psutil

    w, = [x for x in wrapper.worker_dict.values() if getattr(x, '_is_actor', False)]
    sources = (inspect.getfile(type(w)), inspect.getfile(type(w.actor)))
    return dict(pid=os.getpid(), birth=psutil.Process().create_time(), rank=w.rank,
        scheduler=w.actor_lr_scheduler.last_epoch, optimizer_entries=len(w.actor_optimizer.state),
        owner_cache_closed=getattr(w, '_native_textcraft_episode_owners', None) is None,
        modules={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        gen_random_states=w.rollout_sharding_manager.gen_random_states.clone())


def restore_rng(wrapper, states):
    w, = [x for x in wrapper.worker_dict.values() if getattr(x, '_is_actor', False)]
    assert not getattr(w, '_owner_rollout_context_open', False)
    w.rollout_sharding_manager.gen_random_states = states[w.rank].clone()
    return w.rank


original_fit = main_ppo.RayPPOTrainer.fit
original_loop = TrajectoryCollector.multi_turn_loop
holder = []


class BenchmarkFinished(Exception):
    pass


def measured_loop(collector, gen_batch, actor_rollout_wg, envs, is_train=True):
    assert is_train and len(gen_batch) == 32
    handles = actor_rollout_wg.workers
    initial = ray.get([h.execute_with_func_generator.remote(func=worker_state) for h in handles])
    assert len(initial) == 8 and all(r['scheduler'] == 0 and r['optimizer_entries'] == 0 for r in initial)
    expected = json.loads((AUDIT.parent / 'receipts/remote-candidate-prepared-v2.json').read_bytes())
    for row in initial:
        assert row['modules'][str(CANDIDATE / 'verl-owner/verl/workers/fsdp_workers.py')] == expected['worker_SHA256']
    rng = {r['rank']: r['gen_random_states'] for r in initial}
    state['workers_initial'] = [{k: v for k, v in r.items() if k != 'gen_random_states'} for r in initial]
    torch.save(gen_batch, AUDIT / 'original-first-gen-batch.pt')
    candidate = envs.collect_native_trajectories
    outputs = []
    try:
        for index, mode in enumerate(('A-global', 'B-local-first', 'B-local-repeat', 'A-global-repeat')):
            from functools import partial
            ray.get([h.execute_with_func_generator.remote(func=partial(restore_rng, states=rng)) for h in handles])
            envs.collect_native_trajectories = (envs._collect_native_trajectories
                if mode.startswith('A-') else candidate)
            data = gen_batch.select(deepcopy=True)
            data.meta_info = dict(data.meta_info, do_sample=False, global_steps=0)
            state.update(status='running_'+mode, current_case=index)
            save()
            started = time.perf_counter()
            output = original_loop(collector, data, actor_rollout_wg, envs, is_train=True)
            elapsed = time.perf_counter() - started
            assert len(output) == 256
            assert len(set(output.non_tensor_batch['uid'])) == 32
            assert len(set(output.non_tensor_batch['traj_uid'])) == 256
            path = AUDIT / (str(index)+'-'+mode+'.pt')
            torch.save(output, path)
            equal = (set(output.batch.keys()) == set(outputs[0].batch.keys()) and all(
                        output.batch[k].dtype == outputs[0].batch[k].dtype and
                        torch.equal(output.batch[k], outputs[0].batch[k]) for k in output.batch)
                     if outputs else True)
            state['cases'].append(dict(mode=mode, seconds=elapsed, rows=len(output),
                policy_tokens=int(output.batch['response_mask'].sum()),
                total_input_tokens=int(output.batch['attention_mask'].sum()),
                original_tool_calls=float(output.batch['task_rounds'].sum()),
                original_reward_mean=float(output.batch['rm_scores'].sum(-1).mean()),
                all_original_tensor_fields_equal_first_case=equal, output=str(path),
                output_SHA256=hashlib.sha256(path.read_bytes()).hexdigest()))
            outputs.append(output)
            save()
    finally:
        envs.collect_native_trajectories = candidate
    state['matched_original_tensor_workload'] = all(x['all_original_tensor_fields_equal_first_case'] for x in state['cases'])
    final = ray.get([h.execute_with_func_generator.remote(func=worker_state) for h in handles])
    assert all(r['scheduler'] == 0 and r['optimizer_entries'] == 0 for r in final)
    state['workers_after_cases'] = [{k: v for k, v in r.items() if k != 'gen_random_states'} for r in final]
    state.update(status='native_generation_ABBA_cases_completed', finished_cases=time.time())
    save()
    raise BenchmarkFinished()


def fit_probe(trainer):
    holder.append(trainer)
    try:
        original_fit(trainer)
    except BenchmarkFinished:
        return


if '--preflight' in sys.argv:
    state.update(status='CPU_configuration_preflight_passed_not_GPU_validation', model_calls=0)
    save()
    (AUDIT / 'CPU-preflight.json').write_text(json.dumps(state, indent=2)+'\n')
    print(state['status'])
    raise SystemExit(0)

save()
env = dict(os.environ)
env.pop('CUDA_VISIBLE_DEVICES', None)
ray.init(address='local', num_cpus=16, num_gpus=8, include_dashboard=False,
    _temp_dir='/tmp/native-dp-owner-bench-20261018-v1',
    runtime_env=dict(env_vars=env, worker_process_setup_hook='metax_fsdp_event_wait.worker_setup'))
try:
    with patch.object(main_ppo.RayPPOTrainer, 'fit', fit_probe), \
         patch.object(TrajectoryCollector, 'multi_turn_loop', measured_loop):
        main_ppo.TaskRunner.__ray_metadata__.modified_class.run(None, cfg)
    assert len(state['cases']) == 4
    if holder:
        holder[0].envs.close()
        holder[0].val_envs.close()
        closed = ray.get([h.execute_with_func_generator.remote(func=worker_state)
                         for h in holder[0].actor_rollout_wg.workers])
        assert all(r['owner_cache_closed'] for r in closed)
        state['all8_native_episode_owner_caches_closed'] = True
    state.update(status='completed_native_generation_ABBA', finished=time.time())
    save()
except BaseException as error:
    state.update(status='failed', error=repr(error), finished=time.time())
    save()
    raise
finally:
    ray.shutdown()
print(json.dumps(dict(status=state['status'], matched_workload=state['matched_original_tensor_workload'],
                      cases=state['cases']), indent=2))
