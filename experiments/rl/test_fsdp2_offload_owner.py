"""Exercise the installed VERL memory owner with a real PEFT/FSDP2 root.

Device copies must preserve values exactly. This is not an attention/PPO
numerical-tolerance test and does not impose a multi-step equivalence gate.
"""
import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu


class ToyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)
        self.register_buffer('scale', torch.tensor(3.25))

    def forward(self, x):
        return self.proj(x) * self.scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires actual GPU device copies')
def test_peft_wrapper_uses_existing_fsdp2_memory_owner(tmp_path):
    created = not torch.distributed.is_initialized()
    if created:
        torch.distributed.init_process_group(
            'nccl', init_method='file://' + str(tmp_path / 'rendezvous'), rank=0, world_size=1,
        )
    try:
        model = get_peft_model(ToyModule().cuda(), LoraConfig(r=1, target_modules=['proj']))
        root = model.get_base_model()
        fully_shard(root, mesh=init_device_mesh('cuda', (1,)))
        def local_cpu(tensor):
            value = tensor.to_local() if hasattr(tensor, 'to_local') else tensor
            return value.detach().cpu().clone()
        before = {k: local_cpu(v) for k, v in model.state_dict().items()}
        offload_fsdp_model_to_cpu(model)
        assert all(v.device.type == 'cpu' for v in model.parameters())
        assert all(v.device.type == 'cpu' for v in model.buffers())
        load_fsdp_model_to_gpu(model)
        assert all(v.device.type == 'cuda' for v in model.parameters())
        assert all(v.device.type == 'cuda' for v in model.buffers())
        assert model.get_base_model() is root
        for name, value in model.state_dict().items():
            assert torch.equal(local_cpu(value), before[name]), name
    finally:
        if created:
            torch.distributed.destroy_process_group()
