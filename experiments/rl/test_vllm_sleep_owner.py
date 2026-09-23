"""Execute the installed owner's load_model method with observable contexts."""
import ast
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from patch_vllm_sleep import NEW, OLD, patch_source


def test_native_load_enters_weight_pool_and_config():
    path = os.environ.get("VLLM_OWNER_GPU_WORKER")
    if not path:
        pytest.skip("set VLLM_OWNER_GPU_WORKER to the installed owner source")
    source = Path(path).read_text()
    assert NEW in source
    tree = ast.parse(source)
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Worker")
    load = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "load_model")
    events = []

    @contextmanager
    def context(name):
        events.append(("enter", name))
        try:
            yield
        finally:
            events.append(("exit", name))

    namespace = {"os": os, "set_current_vllm_config": lambda config: context("config")}
    exec(compile(ast.Module(body=[load], type_ignores=[]), path, "exec"), namespace)
    worker = SimpleNamespace(
        vllm_config=None,
        _maybe_get_memory_pool_context=lambda tag: context(tag),
        model_runner=SimpleNamespace(load_model=lambda **kwargs: events.append(("load", kwargs))),
    )
    namespace["load_model"](worker)
    assert events == [
        ("enter", "weights"), ("enter", "config"),
        ("load", {"eep_scale_up": os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1"}),
        ("exit", "config"), ("exit", "weights"),
    ]
    assert patch_source(source) == source
    assert patch_source(source.replace(NEW, OLD, 1)) == source
