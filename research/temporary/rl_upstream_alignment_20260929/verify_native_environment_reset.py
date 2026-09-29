"""Exercise native task construction/reset without loading the policy model."""
import json
import os
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
import ray
from agent_system.environments.env_manager import make_envs

root = Path(os.environ['VERL_ROOT'])
audit = Path(os.environ['DT_RUNTIME_ROOT']) / 'receipts/upstream-alignment-20260929'
cfg = OmegaConf.load(root / 'verl/trainer/config/ppo_trainer.yaml')
cfg.data.train_batch_size, cfg.data.val_batch_size = 4, 1
cfg.env.rollout.n = 1
cfg.env.resources_per_worker.runtime_env = {'env_vars': {'CUDA_VISIBLE_DEVICES': '', 'MACA_VISIBLE_DEVICES': ''}}
result = dict(scope=__doc__, tasks={}, status='running')
started = time.perf_counter()
ray.init(num_cpus=8, num_gpus=0, include_dashboard=False)
try:
    for task in ('Webshop', 'AppWorld'):
        cfg.env.env_name = task
        np.random.seed(2026)
        train, val = make_envs(cfg)
        try:
            observations, info = train.reset(kwargs=None)
            result['tasks'][task] = dict(observations=len(observations['text']),
                observation_chars=[len(text) for text in observations['text']],
                infos=len(info), history_length=cfg.env.history_length)
            print(task, result['tasks'][task], flush=True)
        finally:
            train.envs.close()
            val.envs.close()
            del train, val
    result['status'] = 'passed_native_reset_only'
except Exception as exc:
    result.update(status='failed', error=repr(exc))
    raise
finally:
    result['seconds'] = time.perf_counter()-started
    (audit / 'native-environment-reset.json').write_text(json.dumps(result, indent=2) + '\n')
    ray.shutdown()
