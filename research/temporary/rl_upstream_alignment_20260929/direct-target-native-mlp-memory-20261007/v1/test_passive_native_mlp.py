"""CPU hook/binding tests only; not Qwen, PEFT, DT or capacity validation."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
import weakref

import torch
from torch import nn

from passive_native_mlp import passive_native_mlp


class Projection(nn.Module):
    """Original Torch Linear child with named Identities for hook coverage."""
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(2, 2, bias=False)
        self.lora_A = nn.ModuleDict({'existing': nn.Identity()})
        self.lora_B = nn.ModuleDict({'existing': nn.Identity()})

    def forward(self, value):
        result = self.base_layer(value)
        value = self.lora_A['existing'](value)
        self.lora_B['existing'](value)
        return result


class ModuleWithNamedChildren(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = Projection()
        self.up_proj = Projection()
        self.act_fn = nn.SiLU()
        self.down_proj = Projection()

    def forward(self, value):
        # This is an interface-only owner, deliberately not a Qwen/PEFT MLP.
        return self.down_proj(self.act_fn(self.up_proj(self.gate_proj(value))))


class Owner:
    def __init__(self, layer):
        self.model = SimpleNamespace(language_model=SimpleNamespace(layers=[layer]))
        self.calls = 0
        self.arguments = None

    def replay_finite_layer(self, layer, replay, *args, **kwargs):
        self.calls += 1
        self.arguments = (id(layer), args, kwargs)
        return replay()


class PassiveNativeMLPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dt-passive-mlp-')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'events.jsonl'
        self.layer = nn.Module()
        self.layer.block_type = 'interface_test_only'
        self.layer.mlp = ModuleWithNamedChildren().eval()
        self.owner = Owner(self.layer)

    def records(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def hook_counts(self):
        return {id(module): (len(module._forward_pre_hooks), len(module._forward_hooks))
                for module in self.layer.modules()}

    def assert_no_tensor(self, value):
        self.assertNotIsInstance(value, torch.Tensor)
        if isinstance(value, dict):
            for item in value.values():
                self.assert_no_tensor(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                self.assert_no_tensor(item)

    def test_original_call_return_arguments_metadata_and_existing_hooks(self):
        seen = []
        handle = self.layer.mlp.up_proj.register_forward_hook(
            lambda module, args, output: seen.append(id(output)))
        self.addCleanup(handle.remove)
        counts = self.hook_counts()
        result_holder = []

        def replay():
            with torch.no_grad():
                result = self.layer.mlp(torch.ones(1, 3, 2))
            result_holder.append(result)
            return result

        with passive_native_mlp(self.owner, self.path,
                                source_identity={'status': 'prepared-only CPU interface test'},
                                torch_module=torch) as state:
            result = self.owner.replay_finite_layer(self.layer, replay, 17, marker='same')
            self.assertIs(result, result_holder[0])
            self.assertEqual(self.hook_counts(), counts)
        self.assertEqual(self.owner.calls, 1)
        self.assertEqual(self.owner.arguments, (id(self.layer), (17,), {'marker': 'same'}))
        self.assertNotIn('replay_finite_layer', vars(self.owner))
        self.assertEqual(len(seen), 1)
        self.assertEqual(state['replay_calls'], 1)
        self.assertEqual(state['observation_errors'], [])
        records = self.records()
        pre = [r for r in records if r['phase'] == 'native_mlp_module_pre']
        post = [r for r in records if r['phase'] == 'native_mlp_module_post']
        names = {'mlp', 'gate', 'up', 'silu', 'down'}
        names.update(f'{name}.base_layer' for name in ('gate', 'up', 'down'))
        names.update(f'{name}.{family}.existing' for name in ('gate', 'up', 'down')
                     for family in ('lora_A', 'lora_B'))
        self.assertEqual({r['module'] for r in pre}, names)
        self.assertEqual({r['module'] for r in post}, names)
        self.assertEqual(len(pre), len(names))
        self.assertEqual(len(post), len(names))
        for record in pre:
            self.assertEqual(record['layer']['layer_index'], 0)
            self.assertEqual(record['inputs'][0]['shape'], [1, 3, 2])
            self.assertEqual(record['inputs'][0]['dtype'], 'torch.float32')
            self.assertEqual(record['inputs'][0]['stride'], [6, 2, 1])
            self.assertEqual(record['inputs'][0]['logical_bytes'], 24)
            self.assertGreaterEqual(record['inputs'][0]['storage_bytes'], 24)
            self.assertEqual(record['allocator']['allocated_bytes'], None)
            self.assertIn('not mx-smi physical', record['allocator']['scope'])
        self.assert_no_tensor(records)
        self.assert_no_tensor(state)

    def test_original_base_layer_observed_once_without_replacement(self):
        bases = {name: getattr(self.layer.mlp, f'{name}_proj').base_layer
                 for name in ('gate', 'up', 'down')}
        calls = dict.fromkeys(bases, 0)
        handles = []
        for name, module in bases.items():
            def count(_module, _args, _output, name=name):
                calls[name] += 1
            handles.append(module.register_forward_hook(count))
        self.addCleanup(lambda: [handle.remove() for handle in handles])
        counts = self.hook_counts()
        with passive_native_mlp(self.owner, self.path, torch_module=torch):
            with torch.no_grad():
                self.owner.replay_finite_layer(
                    self.layer, lambda: self.layer.mlp(torch.ones(1, 2, 2)))
        self.assertEqual(calls, dict.fromkeys(bases, 1))
        self.assertEqual(self.hook_counts(), counts)
        self.assertEqual(self.owner.calls, 1)
        records = self.records()
        for name, original in bases.items():
            post = [record for record in records
                    if record['phase'] == 'native_mlp_module_post'
                    and record['module'] == f'{name}.base_layer']
            self.assertEqual(len(post), 1)
            self.assertEqual(post[0]['module_identity']['object_id'], id(original))
            self.assertEqual(post[0]['output']['shape'], [1, 2, 2])
            self.assertEqual(post[0]['output']['dtype'], 'torch.float32')
            self.assertEqual(post[0]['output']['storage_bytes'], 16)

    def test_exception_identity_hooks_and_inherited_binding_restored(self):
        failure = RuntimeError('original owner failure')
        counts = self.hook_counts()

        def replay():
            with torch.no_grad():
                self.layer.mlp(torch.ones(1, 1, 2))
            raise failure

        with self.assertRaises(RuntimeError) as caught:
            with passive_native_mlp(self.owner, self.path, torch_module=torch):
                self.owner.replay_finite_layer(self.layer, replay)
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.owner.calls, 1)
        self.assertEqual(self.hook_counts(), counts)
        self.assertNotIn('replay_finite_layer', vars(self.owner))
        phases = [r['phase'] for r in self.records()]
        self.assertIn('native_mlp_replay_exception', phases)
        self.assertEqual(phases[-1], 'native_mlp_observer_binding_restored')

    def test_exact_existing_instance_binding_restored(self):
        original = self.owner.replay_finite_layer

        def existing_wrapper(_self, layer, replay, *args, **kwargs):
            return original(layer, replay, *args, **kwargs)

        binding = MethodType(existing_wrapper, self.owner)
        self.owner.replay_finite_layer = binding
        sentinel = object()
        with passive_native_mlp(self.owner, self.path, torch_module=torch):
            self.assertIs(self.owner.replay_finite_layer(self.layer, lambda: sentinel), sentinel)
        self.assertIs(vars(self.owner)['replay_finite_layer'], binding)
        self.assertEqual(self.owner.calls, 1)

    def test_partial_module_failure_preserves_prior_branch_metadata(self):
        failure = RuntimeError('original up module failure')

        def fail(_module, _inputs, _output):
            raise failure

        handle = self.layer.mlp.up_proj.register_forward_hook(fail)
        self.addCleanup(handle.remove)
        counts = self.hook_counts()
        with self.assertRaises(RuntimeError) as caught:
            with passive_native_mlp(self.owner, self.path, torch_module=torch):
                with torch.no_grad():
                    self.owner.replay_finite_layer(
                        self.layer, lambda: self.layer.mlp(torch.ones(1, 2, 2)))
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.owner.calls, 1)
        self.assertEqual(self.hook_counts(), counts)
        self.assertNotIn('replay_finite_layer', vars(self.owner))
        post_names = {record['module'] for record in self.records()
                      if record['phase'] == 'native_mlp_module_post'}
        self.assertIn('up.lora_B.existing', post_names)
        self.assertNotIn('up', post_names)
        self.assertNotIn('mlp', post_names)
        self.assertIn('native_mlp_replay_exception',
                      [record['phase'] for record in self.records()])

    def test_no_tensor_retention_while_observer_remains_installed(self):
        with passive_native_mlp(self.owner, self.path, torch_module=torch) as state:
            value = torch.ones(1, 3, 2)
            input_ref = weakref.ref(value)
            with torch.no_grad():
                result = self.owner.replay_finite_layer(self.layer, lambda: self.layer.mlp(value))
            output_ref = weakref.ref(result)
            del value, result
            gc.collect()
            self.assertIsNone(input_ref())
            self.assertIsNone(output_ref())
            self.assert_no_tensor(state)
            self.assert_no_tensor(self.records())


if __name__ == '__main__':
    unittest.main(verbosity=2)
