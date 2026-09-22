"""Compare unchanged WebShop workers with and without CPU-only Ray runtime.

Uses the owner's worker and actual task data, including its original reward.
The only changed input is accelerator visibility for this CPU environment.
"""
import argparse
import json
from pathlib import Path
import time

import ray

from agent_system.environments.env_package.webshop.envs import WebshopWorker


def inspect_worker(_):
    from pathlib import Path
    import os
    import torch
    status = Path('/proc/self/status').read_text()
    return {
        'pid': os.getpid(),
        'memory_kib': {line.split(':')[0]: int(line.split(':')[1].split()[0])
                       for line in status.splitlines()
                       if line.startswith(('VmRSS:', 'RssAnon:', 'RssFile:'))},
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'maca_visible_devices': os.environ.get('MACA_VISIBLE_DEVICES'),
        'cuda_initialized': torch.cuda.is_initialized(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--webshop-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    kwargs = dict(observation_mode='text', num_products=None, human_goals=False,
                  file_path=str(args.webshop_root/'data/items_shuffle_1000.json'),
                  attr_path=str(args.webshop_root/'data/items_ins_v2_1000.json'))
    ray.init(num_cpus=2, include_dashboard=False)
    worker_type = ray.remote(num_cpus=.1)(WebshopWorker)
    workers = []
    report = {'started': time.time(), 'owner': WebshopWorker.__module__,
              'env_kwargs': kwargs, 'checks': []}
    try:
        workers.append(worker_type.remote(0, kwargs))
        workers.append(worker_type.options(runtime_env={'env_vars': {
            'CUDA_VISIBLE_DEVICES': '', 'MACA_VISIBLE_DEVICES': '',
        }}).remote(0, kwargs))
        resets = ray.get([w.reset.remote(500) for w in workers])
        assert resets[0] == resets[1], 'Original reset observations differ'
        report['workers'] = ray.get([w.__ray_call__.remote(inspect_worker) for w in workers])
        goal = ray.get(workers[0].get_goals.remote())[500]
        def step(action):
            outputs = ray.get([w.step.remote(action) for w in workers])
            assert outputs[0] == outputs[1], f'Original step differs: {action}'
            report['checks'].append({'action': action, 'equal': True,
                                     'reward': outputs[0][1], 'done': outputs[0][2],
                                     'task_score': outputs[0][3].get('task_score')})
            return outputs[0]

        step(f"search[{goal['name']}]")
        available = ray.get(workers[0].get_available_actions.remote())['clickables']
        assert goal['asin'].lower() in [str(a).lower() for a in available], available
        asin = next(a for a in available if str(a).lower() == goal['asin'].lower())
        step(f'click[{asin}]')
        for value in goal['goal_options'].values():
            step(f'click[{value}]')
        purchase = step('click[Buy Now]')
        assert purchase[2], 'Parity sequence must reach actual reward evaluation'
        assert report['workers'][1]['cuda_visible_devices'] == ''
        assert report['workers'][1]['maca_visible_devices'] == ''
        report.update(status='passed', finished=time.time())
    except Exception as exc:
        report.update(status='failed', error=repr(exc), finished=time.time())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        for worker in workers:
            ray.kill(worker)
        ray.shutdown()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
