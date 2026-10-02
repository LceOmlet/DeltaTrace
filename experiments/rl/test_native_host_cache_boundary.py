"""Check the resource seam against the actual downloaded owner source.

The mocks check lifecycle placement only.  Allocator behavior is validated by
the separate real PyTorch pinned-transfer receipt; this is no numerical test
of DT or PPO and introduces no numerical tolerance.
"""
import argparse
import ast
import contextlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from patch_native_host_cache import ENABLE_ENV, METHODS, patch_source

SOURCE = None
OWNER_PATH = None
RESULT = object()


def source_without_release(tree):
    for method in ast.walk(tree):
        if isinstance(method, (ast.FunctionDef, ast.Try)):
            for attribute in ("body", "finalbody"):
                if not hasattr(method, attribute):
                    continue
                setattr(method, attribute, [node for node in getattr(method, attribute)
                        if not (isinstance(node, ast.If) and ENABLE_ENV in ast.unparse(node.test))])
    return ast.dump(tree, include_attributes=False)


class OwnerBoundaryTest(unittest.TestCase):
    def setUp(self):
        if SOURCE is None:
            self.skipTest("Pass --owner with the actual downloaded owner source")

    def test_only_native_resource_statements_added(self):
        self.assertEqual(ast.dump(ast.parse(SOURCE), include_attributes=False),
                         source_without_release(ast.parse(patch_source(SOURCE))))
        self.assertEqual(patch_source(patch_source(SOURCE)), patch_source(SOURCE))

    def test_dt_result_and_default_lifecycle_are_original(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                result, calls = self.call_dt(enabled=enabled)
                self.assertIs(result, RESULT)
                self.assertEqual(calls, ["load", "attribute", "offload"] + (["release"] if enabled else []))

    def test_dt_failure_and_original_cleanup_are_preserved(self):
        calls = []
        with self.assertRaisesRegex(ValueError, "original DT failure"):
            self.call_dt(enabled=True, error=True, calls=calls)
        self.assertEqual(calls, ["load", "attribute", "offload", "release"])

    @staticmethod
    def call_dt(*, enabled, error=False, calls=None):
        calls = [] if calls is None else calls
        tree = ast.parse(patch_source(SOURCE))
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name == "ActorRolloutRefWorker")
        method = next(node for node in owner.body if isinstance(node, ast.FunctionDef)
                      and node.name == METHODS[1])
        method.decorator_list = []
        # Execute the actual owner function, not a copied substitute for it.
        # Its producer is already initialized, as in later formal phases.
        def attribute(data):
            calls.append("attribute")
            if error:
                raise ValueError("original DT failure")
            return RESULT
        class Timer:
            def __init__(self, **kwargs): self.last = 0
            def __enter__(self): return self
            def __exit__(self, *args): return False
        torch = SimpleNamespace(cuda=SimpleNamespace(memory=SimpleNamespace(
            host_memory_stats=lambda: {"reserved_bytes.current": 0})),
            _C=SimpleNamespace(_host_emptyCache=lambda: calls.append("release")))
        namespace = dict(os=os, torch=torch, Timer=Timer, DataProto=object,
            load_fsdp_model_to_gpu=lambda model: calls.append("load"),
            offload_fsdp_model_to_cpu=lambda model: calls.append("offload"))
        exec(compile(ast.Module(body=[method], type_ignores=[]), OWNER_PATH, "exec"), namespace)
        worker = SimpleNamespace(_is_offload_param=True, actor_module_fsdp=object(),
            _deltatrace_producer=SimpleNamespace(attribute_prepared_batch=attribute))
        # The local import is part of the actual owner; avoid importing a model.
        module = SimpleNamespace(DeltaTraceRolloutProducer=object)
        with patch.dict(os.environ, {ENABLE_ENV: "1" if enabled else "0"}), \
                patch.dict("sys.modules", {"deltatrace_rollout": module}), \
                contextlib.redirect_stdout(io.StringIO()):
            result = namespace[METHODS[1]](worker, object())
        return result, calls


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True, type=Path)
    args, remaining = parser.parse_known_args()
    OWNER_PATH = str(args.owner)
    SOURCE = args.owner.read_text(encoding="utf-8")
    RESULT = object()
    unittest.main(argv=[__file__, *remaining])
