"""Standard-library API/lifecycle tests, not FA/GDN numerical validation.

Real owner CPU decoder-hook validation is a separate test. These structural
fixtures only exercise metadata alias accounting and context/hook cleanup.
"""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from diagnose_qwen35_root_capture_inventory import NativeRootCaptureInventory


class Storage:
    def __init__(self, pointer, size):
        self.pointer, self.size = pointer, size

    def data_ptr(self):
        return self.pointer

    def nbytes(self):
        return self.size


class TensorMetadata:
    dtype, device = "float32", "cpu"

    def __init__(self, storage, shape=(2, 2), offset=0):
        self.storage, self.shape, self.offset = storage, shape, offset

    def untyped_storage(self):
        return self.storage

    def stride(self):
        return (self.shape[1], 1)

    def storage_offset(self):
        return self.offset

    def numel(self):
        return self.shape[0] * self.shape[1]

    def element_size(self):
        return 4


class Handle:
    def __init__(self, hooks, key):
        self.hooks, self.key = hooks, key

    def remove(self):
        self.hooks.pop(self.key, None)


class Layer:
    block_type = "linear_attention"

    def __init__(self):
        self.pre, self.post = {}, {}
        self.calls = 0

    def register_forward_pre_hook(self, hook, with_kwargs):
        self.pre[id(hook)] = hook
        return Handle(self.pre, id(hook))

    def register_forward_hook(self, hook, with_kwargs=False, always_call=False):
        self.post[id(hook)] = (hook, with_kwargs, always_call)
        return Handle(self.post, id(hook))

    def run(self, output, fail=False):
        args, kwargs = (output,), {}
        for hook in tuple(self.pre.values()):
            assert hook(self, args, kwargs) is None
        self.calls += 1
        try:
            if fail:
                raise ValueError("original layer failure")
        except BaseException:
            for hook, with_kwargs, always in tuple(self.post.values()):
                if always:
                    assert hook(self, args, kwargs, None) is None
            raise
        for hook, with_kwargs, always in tuple(self.post.values()):
            result = hook(self, args, kwargs, output) if with_kwargs else hook(self, args, output)
            assert result is None
        return output


class Capture:
    def __init__(self, layer, decoder=False, fail_enter=False):
        self.layer, self.decoder, self.fail_enter = layer, decoder, fail_enter
        self.values, self.endpoints, self.calls = {}, {}, {}
        self.input_shape = (2, 2)
        self.coefficient_start = 0
        self.closed = False
        self.handle = None

    def __enter__(self):
        if self.fail_enter:
            raise ValueError("original capture failure")
        if self.decoder:
            def output(_layer, _args, _output):
                self.calls["decoder"] = self.calls.get("decoder", 0) + 1
            self.handle = self.layer.register_forward_hook(output)
        return self

    def __exit__(self, *_args):
        self.closed = True
        if self.handle:
            self.handle.remove()


class InventoryTests(unittest.TestCase):
    def factory(self, external=True, fail_enter=False):
        self.captures = []
        def create(index, layer, args, kwargs):
            dc, mc = Capture(layer, decoder=True), Capture(layer, fail_enter=fail_enter)
            shared, prefix = Storage(100 + index, 16), Storage(1000 + index, 64)
            dc.values.update(gate=TensorMetadata(shared), up=TensorMetadata(shared, (1, 2), 2))
            mc.values.update(key=TensorMetadata(prefix), mask=None)
            self.captures.append((dc, mc))
            return dc, mc, {"native_cache": TensorMetadata(prefix)} if external else {}
        return create

    def test_original_output_and_owner_post_order(self):
        layer, output = Layer(), object()
        with NativeRootCaptureInventory([layer], self.factory()) as inventory:
            self.assertIs(layer.run(output), output)
        report = inventory.report()
        self.assertEqual(layer.calls, 1)
        self.assertEqual(report["rows"][0]["decoder_calls"], {"decoder": 1})
        self.assertTrue(report["rows"][0]["metadata_read_after_context_exit"])
        self.assertEqual(report["rows"][0]["gdn_coefficient_start"], 0)
        self.assertTrue(all(c.closed for c in self.captures[0]))
        self.assertFalse(layer.pre or layer.post)
        self.assertFalse(report["active_capture_remaining"])
        json.dumps(report)

    def test_exact_alias_accounting_and_external_exclusion(self):
        layer = Layer()
        with NativeRootCaptureInventory([layer], self.factory()) as inventory:
            layer.run(object())
        row = inventory.report()["rows"][0]
        self.assertEqual(row["logical_payload_bytes"], 40)
        self.assertEqual(row["unique_storage_bytes"], 80)
        self.assertEqual(row["external_alias_storage_bytes"], 64)
        self.assertEqual(row["storage_without_known_external_alias_bytes"], 16)
        self.assertEqual(row["storage_groups"][0]["fields"], ["decoder.gate", "decoder.up"])
        self.assertTrue(all(not c.values and not c.endpoints for c in self.captures[0]))

    def test_missing_external_map_is_not_excluded(self):
        layer = Layer()
        with NativeRootCaptureInventory([layer], self.factory(external=False)) as inventory:
            layer.run(object())
        row = inventory.report()["rows"][0]
        self.assertFalse(row["external_mapping_supplied"])
        self.assertEqual(row["external_alias_storage_bytes"], 0)
        self.assertEqual(row["storage_without_known_external_alias_bytes"], 80)

    def test_external_callback_reads_after_original_context_exit(self):
        layer, called_after_exit = Layer(), []
        original_factory = self.factory()
        def factory(*args):
            dc, mc, _old_mapping = original_factory(*args)
            def late_mapping():
                called_after_exit.append(dc.closed and mc.closed)
                return {"updated_owner_cache": mc.values["key"]}
            return dc, mc, late_mapping
        with NativeRootCaptureInventory([layer], factory) as inventory:
            layer.run(object())
        self.assertEqual(called_after_exit, [True])
        row = inventory.report()["rows"][0]
        self.assertEqual(row["external_alias_storage_bytes"], 64)
        self.assertEqual(row["storage_groups"][1]["external_aliases"], ["updated_owner_cache"])

    def test_original_layer_exception_closes_contexts(self):
        layer = Layer()
        with self.assertRaisesRegex(ValueError, "original layer failure"):
            with NativeRootCaptureInventory([layer], self.factory()) as inventory:
                layer.run(object(), fail=True)
        self.assertEqual(inventory.report()["rows"][0]["status"], "original_call_raised")
        self.assertTrue(all(c.closed for c in self.captures[0]))
        self.assertFalse(layer.pre or layer.post)

    def test_mixer_enter_exception_removes_decoder_hook(self):
        layer = Layer()
        with self.assertRaisesRegex(ValueError, "original capture failure"):
            with NativeRootCaptureInventory([layer], self.factory(fail_enter=True)) as inventory:
                layer.run(object())
        self.assertTrue(self.captures[0][0].closed)
        self.assertFalse(layer.pre or layer.post)
        self.assertFalse(inventory.report()["active_capture_remaining"])

    def test_second_call_observed_but_not_recaptured(self):
        layer = Layer()
        with NativeRootCaptureInventory([layer], self.factory()) as inventory:
            layer.run(object())
            layer.run(object())
        self.assertEqual(len(self.captures), 1)
        self.assertEqual(inventory.report()["seen_layer_calls"], [2])
        self.assertEqual(len(inventory.report()["rows"]), 1)


if __name__ == "__main__":
    unittest.main()
