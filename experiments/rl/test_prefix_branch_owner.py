"""CPU control-flow checks for the native FSDP prefix branch seam.

Execute the actual integration method; doubles record the existing collective
contract only. This does not emulate FSDP, a model, or DT numerical propagation.
"""
import ast
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


OWNER_PATH = Path(__file__).with_name('deltatrace_rollout.py')


def owner_method():
    source = ast.parse(OWNER_PATH.read_text(encoding='utf8'))
    owner = next(node for node in source.body
                 if isinstance(node, ast.ClassDef)
                 and node.name == '_Qwen35CausalOwnerView')
    method = next(node for node in owner.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == 'synchronize_prefix_start')
    return method


def boundary(monkeypatch, *, sharded, mesh_size, mesh_ndim, group_enabled):
    calls = []

    class Scalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class Mesh:
        ndim = mesh_ndim

        def size(self):
            return mesh_size

        def __getitem__(self, key):
            calls.append(('mesh', key))
            assert key == 'fsdp'
            return self

        def get_group(self):
            return 'native-fsdp-group'

    class DTensor:
        device_mesh = Mesh()

    tensor_module = ModuleType('torch.distributed.tensor')
    tensor_module.DTensor = DTensor
    monkeypatch.setitem(sys.modules, 'torch.distributed.tensor', tensor_module)

    def tensor(value, *, device, dtype):
        calls.append(('tensor', value, device, dtype))
        return Scalar(value)

    def all_reduce(value, *, op, group):
        calls.append(('all_reduce', value.value, op, group))
        value.value = min(value.value, group_enabled)

    torch = SimpleNamespace(tensor=tensor, long='long', distributed=SimpleNamespace(
        all_reduce=all_reduce, ReduceOp=SimpleNamespace(MIN='native-min')))
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[owner_method()], type_ignores=[]),
                 str(OWNER_PATH), 'exec'), namespace)
    view = SimpleNamespace(lm_head=SimpleNamespace(
        weight=DTensor() if sharded else object()), execution_device='native-device')
    return lambda prefix: namespace['synchronize_prefix_start'](view, prefix), calls


@pytest.mark.parametrize('prefix', [0, 64, 7296, 8640])
@pytest.mark.parametrize('mesh_ndim', [1, 2])
def test_nonzero_ranks_preserve_local_prefix_and_collective(monkeypatch, prefix, mesh_ndim):
    call, calls = boundary(monkeypatch, sharded=True, mesh_size=2,
                           mesh_ndim=mesh_ndim, group_enabled=1)
    assert call(prefix) == prefix
    assert [entry for entry in calls if entry[0] == 'tensor'] == [
        ('tensor', int(prefix > 0), 'native-device', 'long')]
    assert [entry for entry in calls if entry[0] == 'all_reduce'] == [
        ('all_reduce', int(prefix > 0), 'native-min', 'native-fsdp-group')]
    assert [entry for entry in calls if entry[0] == 'mesh'] == (
        [('mesh', 'fsdp')] if mesh_ndim > 1 else [])


@pytest.mark.parametrize('prefix', [0, 64, 8640])
def test_one_zero_rank_disables_prefix_branch_everywhere(monkeypatch, prefix):
    call, calls = boundary(monkeypatch, sharded=True, mesh_size=2,
                           mesh_ndim=1, group_enabled=0)
    assert call(prefix) == 0
    assert len([entry for entry in calls if entry[0] == 'all_reduce']) == 1


@pytest.mark.parametrize('sharded,mesh_size', [(False, 2), (True, 1)])
def test_unsharded_and_single_rank_keep_original_path(monkeypatch, sharded, mesh_size):
    call, calls = boundary(monkeypatch, sharded=sharded, mesh_size=mesh_size,
                           mesh_ndim=1, group_enabled=0)
    for prefix in [0, 64, 8640]:
        assert call(prefix) == prefix
    assert calls == []
