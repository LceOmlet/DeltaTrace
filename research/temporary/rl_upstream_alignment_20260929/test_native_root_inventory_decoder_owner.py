"""Original decoder capture and PyTorch hook order on CPU, not mixer numerics."""
import ast
import os
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
from diagnose_qwen35_root_capture_inventory import NativeRootCaptureInventory


def original_decoder_capture():
    source = Path(os.environ['DT_ROOT_INVENTORY_DECODER_SOURCE'])
    node = next(n for n in ast.parse(source.read_bytes()).body
                if isinstance(n, ast.ClassDef) and n.name == 'NativeDecoderCapture')
    # This class only depends on torch. Execute its exact owner AST without
    # importing GPU finite kernels or creating a substitute capture class.
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['NativeDecoderCapture']


class Decoder(torch.nn.Module):
    block_type = 'cpu_hook_contract_only'

    def __init__(self):
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(4)
        self.post_attention_layernorm = torch.nn.LayerNorm(4)
        self.mlp = MLP()

    def forward(self, value, fail=False):
        value = self.input_layernorm(value)
        if fail:
            raise ValueError('original forward failure')
        return self.mlp(self.post_attention_layernorm(value))


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 8, bias=False)
        self.up_proj = torch.nn.Linear(4, 8, bias=False)
        self.act_fn = torch.nn.SiLU()
        self.down_proj = torch.nn.Linear(8, 4, bias=False)

    def forward(self, value):
        return self.down_proj(self.act_fn(self.gate_proj(value)) * self.up_proj(value))


class NoMixerObservation:
    """No mixer runs in this decoder hook test; no FA/FLA claim."""
    def __init__(self):
        self.values = {}
        self.calls = {}
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


def test_original_capture_post_hook_order_release_and_default_output():
    layer = Decoder().eval()
    value = torch.arange(48, dtype=torch.float32).reshape(4, 3, 4)
    with torch.no_grad():
        expected = layer(value)
    capture = original_decoder_capture()
    created = []
    def factory(index, module, args, kwargs):
        dc = capture(module, destination='cpu', copy_tensors=False,
                     retained_names={'input_norm_input', 'post_norm_input',
                                     'gate_output', 'up_output', 'silu_output'})
        mc = NoMixerObservation()
        created.append((dc, mc))
        return dc, mc, {'layer_input': args[0]}
    with torch.no_grad(), NativeRootCaptureInventory([layer], factory) as audit:
        actual = layer(value)
        repeated = layer(value)
    assert torch.equal(actual, expected) and torch.equal(repeated, expected)
    report = audit.report()
    assert report['complete_layers'] == report['recorded_layers'] == 1
    assert report['seen_layer_calls'] == [2]
    assert report['original_forward_calls_added'] == 0
    assert report['hooks_remaining'] == 0 and not report['active_capture_remaining']
    row = report['rows'][0]
    assert row['decoder_calls'] == {name: 1 for name in
        ('input_norm', 'post_norm', 'gate', 'up', 'silu', 'down', 'mlp', 'decoder')}
    assert {f['field'] for f in row['fields']} == {
        'decoder.' + name for name in ('input_norm_input', 'post_norm_input',
                                      'gate_output', 'up_output', 'silu_output')}
    assert row['external_alias_storage_bytes'] == value.untyped_storage().nbytes()
    assert not row['diagnostic_errors']
    assert all(not dc.values and not dc.handles and not mc.values for dc, mc in created)
    assert not layer._forward_hooks and not layer._forward_pre_hooks


def test_original_forward_exception_preserved_and_capture_hooks_removed():
    layer = Decoder().eval()
    capture = original_decoder_capture()
    def factory(index, module, args, kwargs):
        return capture(module, destination='cpu', copy_tensors=False), NoMixerObservation()
    with pytest.raises(ValueError, match='original forward failure'):
        with NativeRootCaptureInventory([layer], factory) as audit:
            layer(torch.zeros(4, 3, 4), fail=True)
    report = audit.report()
    assert report['recorded_layers'] == 1
    assert report['rows'][0]['status'] == 'original_call_raised'
    assert report['hooks_remaining'] == 0 and not report['active_capture_remaining']
    assert not layer._forward_hooks and not layer._forward_pre_hooks
