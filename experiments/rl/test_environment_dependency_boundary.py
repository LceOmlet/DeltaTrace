"""Environment leaves must not replace the already working model packages."""
import importlib
import os
from pathlib import Path
import site
import sys
import torch
from torch import nn
import pytest


@pytest.mark.parametrize('variable',['TEXTCRAFT_EXTRAS','LOOP_EXTRAS'])
def test_environment_dependencies_do_not_shadow_model_runtime(variable):
    modules=['accelerate','transformers','torch','vllm']
    before={n:Path(importlib.util.find_spec(n).origin).resolve() for n in modules}
    assert all('loop-extras' not in str(p) for p in before.values()),before
    original=sys.path.copy()
    try:
        site.addsitedir(os.environ[variable])
        after={n:Path(importlib.util.find_spec(n).origin).resolve() for n in modules}
        assert before==after
        # Reproduce the exact HF/accelerate parameter metadata boundary which
        # failed with LOOP's old training dependency, using the real owner API.
        from accelerate import init_empty_weights
        with init_empty_weights():
            layer=nn.Linear(4,4)
            parameter=nn.Parameter(torch.zeros(4,4))
            parameter._is_hf_initialized=True
            layer.weight=parameter
        assert layer.weight.device.type=='meta'
    finally:
        sys.path[:]=original
