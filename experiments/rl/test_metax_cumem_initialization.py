"""Check only the backport seam: mapping order, device and checked owner memset."""
import ast
from contextlib import contextmanager
from types import SimpleNamespace
import pytest
from patch_metax_cumem_initialization import OLD, NEW, patch_source


def test_owner_map_then_initialize_on_handle_device():
    events = []
    @contextmanager
    def device(index):
        events.append(('device', index))
        yield
        events.append(('restore_device', index))
    ns = dict(HandleType=tuple, torch=SimpleNamespace(cuda=SimpleNamespace(device=device)),
              python_create_and_map=lambda *args: events.append(('map', args)),
              libcudart=SimpleNamespace(cudaMemset=lambda *args: events.append(('memset', args))))
    exec(compile(ast.parse(NEW), '<owner boundary>', 'exec'), ns)
    ns['create_and_map']((1, 4096, 123456, 999))
    assert events == [('map', (1,4096,123456,999)), ('device',1),
                      ('memset',(123456,0,4096)), ('restore_device',1)]
    assert patch_source(OLD) == NEW
    assert patch_source(NEW) == NEW


def test_failed_mapping_is_not_initialized():
    def failed(*args):
        raise RuntimeError('owner mapping failure')
    ns = dict(HandleType=tuple, python_create_and_map=failed)
    exec(compile(ast.parse(NEW), '<owner boundary>', 'exec'), ns)
    with pytest.raises(RuntimeError, match='owner mapping failure'):
        ns['create_and_map']((0,4096,123456,999))
