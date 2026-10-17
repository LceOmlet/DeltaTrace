"""Original TaskRunner/VERL/vLLM TP1-vs-TP2 probes; no training or new model code."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT/'receipts/native-rollout-TP-efficiency-20261018-v1'
BASE_RUN = ROOT/'runs/textcraft-dtpo-v6-8gpu-200-20261017-v3'
TP = int(os.environ['RL_ROLLOUT_TP'])
assert TP in (1, 2)
OUT = AUDIT/f'TP{TP}'

if '--native' not in sys.argv:
    source = json.loads((BASE_RUN/'source.json').read_bytes())
    env = dict(os.environ, **source['effective_environment'])
    env['CUDA_VISIBLE_DEVICES'] = '' if '--preflight' in sys.argv else '0,1,2,3,4,5,6,7'
    env['DT_FSDP_EVENT_WAIT_RECEIPTS'] = str(OUT/'worker-runtime')
    for key in ('RAY_ADDRESS', 'RAY_TMPDIR', 'MACA_VISIBLE_DEVICES'):
        env.pop(key, None)
    argv = [env['VENV_PYTHON'], '-X', 'utf8', __file__, '--native']
    if '--preflight' in sys.argv:
        argv.append('--preflight')
    raise SystemExit(subprocess.call(argv, env=env))

import psutil
import ray
from omegaconf import OmegaConf
from unittest.mock import patch
from verl.trainer import main_ppo

OUT.mkdir(parents=True, exist_ok=True)
cfg = OmegaConf.load(ROOT/'receipts/textcraft-eight-gpu-sequential-20261017-v3/dt-effective-config.yaml')
assert cfg.trainer.n_gpus_per_node == 8 and cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 4
assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size == 1
assert cfg.actor_rollout_ref.rollout.max_model_len == 32768
cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = TP
cfg.trainer.experiment_name = f'native-generation-TP{TP}-efficiency-only-20261018-v1'
cfg.trainer.default_local_dir = str(OUT/'unused-checkpoints')
cfg.ray_init.runtime_env.env_vars.DT_FSDP_EVENT_WAIT_RECEIPTS = os.environ['DT_FSDP_EVENT_WAIT_RECEIPTS']
record = dict(status='prepared_native_generation_probe', TP=TP, actor_DP=8, physical_gpus=list(range(8)),
    pid=os.getpid(), birth=psutil.Process().create_time(), started=time.time(),
    source_commit=os.environ.get('RL_CAMPAIGN_COMMIT'),
    source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    formal_iterations=0, DT_calls=0, optimizer_updates=0, checkpoint_create_calls=0,
    effective_configuration=OmegaConf.to_container(cfg, resolve=True),
    input_SHA256=hashlib.sha256((AUDIT/'matched-inputs.json').read_bytes()).hexdigest())


def save():
    temp = OUT/'status.tmp'
    temp.write_text(json.dumps(record, indent=2)+'\n')
    temp.replace(OUT/'status.json')


def probe_worker(wrapper, cases):
    import hashlib
    import importlib.util
    import inspect
    import os
    from pathlib import Path
    import time
    from types import MethodType, SimpleNamespace
    import torch
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    w, = [x for x in wrapper.worker_dict.values() if getattr(x, '_is_actor', False)]
    tp = w.rollout_sharding_manager.tp_size
    assert tp in (1, 2) and getattr(w, '_owner_rollout_context_open', False)
    assert w.actor_lr_scheduler.last_epoch == 0 and len(w.actor_optimizer.state) == 0
    root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
    out = root/'receipts/native-rollout-TP-efficiency-20261018-v1'/f'TP{tp}'
    owner = root/'receipts/upstream-alignment-20260929/verify_vllm_batching_owner.py'
    spec = importlib.util.spec_from_file_location('native_vllm_owner_tests', owner)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    engine = w.rollout.inference_engine
    ids = list(engine.llm_engine.list_loras())
    assert len(ids) == 1
    adapter = LoRARequest(str(ids[0]), ids[0], '/simon-stub-path')
    runner = SimpleNamespace(llm=engine, get_inputs=lambda rows, **kwargs: [dict(prompt_token_ids=p) for p in rows])
    runner._final_steps_generate_w_logprobs = harness.owner('vllm015-conftest.py', '_final_steps_generate_w_logprobs', 'VllmRunner')
    runner.generate_w_logprobs = MethodType(harness.owner('vllm015-conftest.py', 'generate_w_logprobs', 'VllmRunner'), runner)
    result = dict(rank=w.rank, tp=tp, scheduler=0, optimizer_entries=0, cases=[],
        modules={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            (inspect.getfile(type(w)), inspect.getfile(type(w.actor)), inspect.getfile(type(engine.llm_engine)))})
    for case, global_prompts in cases.items():
        engines = 8//tp
        assert len(global_prompts) % engines == 0
        width = len(global_prompts)//engines
        first = (w.rank//tp)*width
        prompts = global_prompts[first:first+width]
        trials = ('cold', 'hot') if case in ('dense', 'tail') else ('cold',)
        for trial in trials:
            if trial == 'cold':
                assert engine.reset_prefix_cache()
            num_logprobs = 1 if case in ('dense', 'tail') else 5
            params = SamplingParams(temperature=0, min_tokens=128, max_tokens=128,
                ignore_eos=True, logprobs=num_logprobs, seed=0)
            started = time.perf_counter()
            values = runner.generate_w_logprobs(prompts, params, lora_request=adapter, use_tqdm=False)
            elapsed = time.perf_counter()-started
            assert len(values) == width and all(len(v[0]) == 128 for v in values)
            path = out/f'{case}-{trial}-rank{w.rank}.pt'
            torch.save(dict(outputs=values, global_indices=list(range(first, first+width)),
                prompts=prompts, TP=tp, rank=w.rank), path)
            result['cases'].append(dict(case=case, trial=trial, seconds=elapsed,
                global_requests=len(global_prompts), unique_engines=engines,
                requests_per_engine=width, fixed_output_tokens_per_request=128,
                num_logprobs=num_logprobs,
                prefix_lengths=list(map(len, prompts)), outputs=str(path)))
            (out/f'rank{w.rank}.json').write_text(json.dumps(result, indent=2)+'\n')
    assert w.actor_lr_scheduler.last_epoch == 0 and len(w.actor_optimizer.state) == 0
    result.update(status='completed_original_native_generation_owner_probe', test_sources=harness.report['sources'])
    (out/f'rank{w.rank}.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def fit_probe(trainer):
    from functools import partial
    cases = json.loads((AUDIT/'matched-inputs.json').read_bytes())['cases']
    handles = trainer.actor_rollout_wg.workers
    assert len(handles) == 8
    record.update(status='native_generation_owner_probe_running')
    save()
    trainer.actor_rollout_wg.begin_rollout_context()
    try:
        results = ray.get([h.execute_with_func_generator.remote(func=partial(probe_worker, cases=cases)) for h in handles])
    finally:
        trainer.actor_rollout_wg.end_rollout_context()
    assert len(results) == 8 and all(x['tp'] == TP and x['scheduler'] == x['optimizer_entries'] == 0 for x in results)
    record.update(status='completed_native_generation_probe', results=results, finished=time.time())
    save()
    trainer.envs.close()
    trainer.val_envs.close()


if '--preflight' in sys.argv:
    record.update(status='CPU_preflight_passed_not_GPU_validation')
    save()
    print(record['status'])
    raise SystemExit(0)

save()
env = dict(os.environ)
env.pop('CUDA_VISIBLE_DEVICES', None)
ray.init(address='local', num_cpus=16, num_gpus=8, include_dashboard=False,
    _temp_dir=f'/tmp/native-rollout-TP{TP}-probe-20261018-v1',
    runtime_env=dict(env_vars=env, worker_process_setup_hook='metax_fsdp_event_wait.worker_setup'))
try:
    with patch.object(main_ppo.RayPPOTrainer, 'fit', fit_probe):
        main_ppo.TaskRunner.__ray_metadata__.modified_class.run(None, cfg)
    assert record['status'] == 'completed_native_generation_probe'
except BaseException as error:
    record.update(status='failed', error=repr(error), finished=time.time())
    save()
    raise
finally:
    ray.shutdown()
print(json.dumps(dict(status=record['status'], TP=TP, output=str(OUT))))
