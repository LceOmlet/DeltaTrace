"""Passive probability capture around the original trainer's real two-step loop.

The test-only TaskRunner calls the owning method unchanged. No generation, reward,
advantage, sampler, optimizer or parameter synchronization is implemented here.
"""
import inspect
import json
import os
from pathlib import Path
import sys
import time
import ray
import torch
from omegaconf import OmegaConf
from verl.trainer import main_ppo
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

audit = Path(os.environ['DT_RUNTIME_ROOT'])/'receipts/upstream-alignment-20260929'
directory = audit/'probability-native-trainer'
directory.mkdir(exist_ok=True)

def observe(label, data, rank=None, step=None):
    width = data.batch['responses'].shape[-1]
    mask = data.batch['attention_mask'][:, -width:].bool()
    record = dict(time_unix=time.time(), phase=label, rank=rank, step=step,
                  rows=len(data), valid_tokens=int(mask.sum()), fields={})
    for name in ('rollout_log_probs', 'old_log_probs'):
        if name not in data.batch:
            continue
        value = data.batch[name][mask]
        finite = torch.isfinite(value)
        record['fields'][name] = dict(dtype=str(value.dtype), nan=int(value.isnan().sum()),
            inf=int(value.isinf().sum()), min=float(value[finite].min()) if finite.any() else None,
            max=float(value[finite].max()) if finite.any() else None)
        if not finite.all():
            path = directory/f'{label}-rank{rank}-step{step}-{time.time_ns()}.pt'
            torch.save(data, path)
            record['artifact'] = str(path)
    with (directory/f'observations-rank{rank}.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    print('PROBABILITY_OBSERVATION '+json.dumps(record), flush=True)

OriginalTaskRunner = main_ppo.TaskRunner.__ray_metadata__.modified_class

@ray.remote(num_cpus=1)
class ObservedTaskRunner(OriginalTaskRunner):
    def run(self, config):
        code = RayPPOTrainer.fit.__code__
        lines, first = inspect.getsourcelines(RayPPOTrainer.fit)
        stops = {}
        for index, line in enumerate(lines, start=first):
            if 'old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)' in line:
                stops[index] = 'before_old_logprob'
            if 'rollout_probs = torch.exp(rollout_old_log_probs)' in line:
                stops[index] = 'before_probability_metrics'
        assert len(stops) == 2
        def trace(frame, event, arg):
            if frame.f_code is not code:
                return None
            if event == 'line' and frame.f_lineno in stops:
                observe(stops[frame.f_lineno], frame.f_locals['batch'],
                        step=frame.f_locals['self'].global_steps)
            return trace
        previous = sys.gettrace()
        sys.settrace(trace)
        try:
            return super().run(config)
        finally:
            sys.settrace(previous)

if __name__ == '__main__':
    raw = (directory/'resolved-config.log').read_text()
    config = OmegaConf.create(raw[raw.index('data:\n'):])
    main_ppo.TaskRunner = ObservedTaskRunner
    try:
        main_ppo.run_ppo(config)
    finally:
        ray.shutdown()
