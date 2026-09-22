"""Measure pinned Ray serialization at paper batch/turn counts and 32k width.

This is a transport capacity fixture, not a new environment rollout or a claim
about task reward/DT quality. Original IDs, masks and per-row tensor ownership
are tested separately in test_rollout_storage.py.
"""
import argparse
import gc
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from ray._private.serialization import SerializationContext
from tensordict import TensorDict
from verl import DataProto
from agent_system.multi_turn_rollout.utils import to_list_of_dict


def rows(batch_size, clone):
    ids = torch.arange(batch_size * 32768).reshape(batch_size, 32768)
    batch = DataProto(batch=TensorDict(dict(
        input_ids=ids, prompts=ids[:, :-1024], responses=ids[:, -1024:],
        attention_mask=torch.ones_like(ids), position_ids=torch.arange(32768).repeat(batch_size, 1),
    ), batch_size=[batch_size]), non_tensor_batch={
        'traj_uid': np.array([str(i) for i in range(batch_size)], dtype=object),
        'active_masks': np.ones(batch_size, dtype=bool), 'rewards': np.zeros(batch_size),
    })
    return to_list_of_dict(batch, clone_tensors=clone)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['Webshop', 'Sokoban', 'AppWorld'], required=True)
    parser.add_argument('--scale', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads(args.scale.read_text())['tasks'][args.task]
    count = config['episodes_per_iteration'];turns = int(config['env']['MAX_STEPS'])
    context = SerializationContext(None)
    result = dict(task=args.task, scope=__doc__, batch=count, turns=turns, width=32768, phases=[])

    def record(phase, **values):
        result['phases'].append(dict(phase=phase, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, **values))
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result['phases'][-1]), flush=True)

    for clone in [False, True]:
        small = rows(16, clone)
        start = time.perf_counter();payload = context.serialize(small)
        record('bounded_baseline', clone_tensors=clone, batch=16, turns=1,
               serialized_bytes=payload.total_bytes, seconds=time.perf_counter()-start)
        del payload, small;gc.collect()
    start = time.perf_counter()
    episodes = [[] for _ in range(count)]
    for _ in range(turns):
        for episode, row in zip(episodes, rows(count, True)):
            episode.append(row)
    logical = sum(t.numel()*t.element_size() for ep in episodes for row in ep
                  for t in row.values() if isinstance(t, torch.Tensor))
    record('full_scale_rows', logical_tensor_bytes=logical, seconds=time.perf_counter()-start)
    start = time.perf_counter();payload = context.serialize(episodes)
    record('full_scale_serialized', serialized_bytes=payload.total_bytes, seconds=time.perf_counter()-start)
    # Only a small pickle header per tensor is expected, not N copies of batch storage.
    assert logical <= payload.total_bytes < logical + count*turns*4096
    result['status'] = 'passed'
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
