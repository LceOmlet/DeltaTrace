"""Exercise pinned VERL Ray device ownership, without loading model weights."""
import json
import os

import ray
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


@ray.remote
class DeviceWorker(Worker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def inspect_device(self):
        import torch
        value = torch.ones(1, device='cuda')
        return dict(rank=self.rank, local_rank=os.environ.get('LOCAL_RANK'),
            cuda=os.environ.get('CUDA_VISIBLE_DEVICES'), maca=os.environ.get('MACA_VISIBLE_DEVICES'),
            count=torch.cuda.device_count(), device=value.device.index,
            name=torch.cuda.get_device_name(), ray_ids=ray.get_gpu_ids())


ray.init(num_cpus=4, include_dashboard=False)
try:
    group = RayWorkerGroup(RayResourcePool([2], use_gpu=True, max_colocate_count=1), RayClassWithInitArgs(DeviceWorker))
    print(json.dumps(dict(status='passed_native_ray_device_creation', devices=group.inspect_device())), flush=True)
finally:
    ray.shutdown()
