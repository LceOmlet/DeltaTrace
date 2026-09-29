"""Small real owner sleep/wake check; preserve weights and initialize discarded pages."""
import json
from pathlib import Path
import time
import torch
from vllm_metax.device_allocator.cumem import CuMemAllocator

root = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/upstream-alignment-20260929')
torch.cuda.set_device(0)
pool = CuMemAllocator.get_instance()
with pool.use_memory_pool('weights'):
    weight = torch.full((262144,), 7., device='cuda')
with pool.use_memory_pool('kv_cache'):
    state = torch.full((1048576,), float('nan'), device='cuda')
result = dict(scope=__doc__, weight_bytes=weight.numel()*weight.element_size(),
              state_bytes=state.numel()*state.element_size(), cycles=[])
for cycle in range(3):
    state.fill_(float('nan'))
    torch.cuda.synchronize()
    pool.sleep(offload_tags=('weights',))
    started = time.perf_counter()
    pool.wake_up(tags=['weights'])
    pool.wake_up(tags=['kv_cache'])
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    zeros = bool(torch.equal(state, torch.zeros_like(state)))
    preserved = bool(torch.equal(weight, torch.full_like(weight,7)))
    row = dict(cycle=cycle, wake_seconds=seconds, discarded_zero=zeros, weights_preserved=preserved)
    result['cycles'].append(row)
    print(json.dumps(row), flush=True)
    assert zeros and preserved
(root/'metax-native-pool-initialization.json').write_text(json.dumps(result,indent=2)+'\n')
del state, weight
