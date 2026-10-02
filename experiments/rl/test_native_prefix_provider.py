"""Control-flow checks for an optional owner cache input, not DT numerics."""
import ast
import copy
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
OWNER = 'deltatrace/clean/qwen35/qwen35_dense_finite_runner.py'
BASELINE = '0936d34'
OWNER_PATH = Path(os.environ.get('DT_PREFIX_OWNER_SOURCE', ROOT / OWNER))


def attribute_node(source):
    owner = next(n for n in ast.parse(source).body
                 if isinstance(n, ast.ClassDef) and n.name == 'Qwen35DenseFiniteRunner')
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == 'attribute')


def test_default_body_preserves_the_recorded_owner():
    original = attribute_node(subprocess.check_output(
        ['git', 'show', f'{BASELINE}:{OWNER}'], cwd=ROOT, text=True, encoding='utf8'))
    candidate = attribute_node(OWNER_PATH.read_text(encoding='utf8'))
    # Reduce the optional branch to its default before comparing the complete
    # owner body; this includes all existing capture/replay/finite math.
    class Default(ast.NodeTransformer):
        def visit_If(self, node):
            if (isinstance(node.test, ast.Compare)
                    and isinstance(node.test.left, ast.Name)
                    and node.test.left.id == 'prefix_cache_provider'):
                return node.body if isinstance(node.test.ops[0], ast.Is) else node.orelse
            return self.generic_visit(node)
    candidate = Default().visit(candidate)
    candidate.body.pop(0)  # New API documentation only.
    candidate.args.kwonlyargs.clear()
    candidate.args.kw_defaults.clear()
    assert ast.dump(candidate) == ast.dump(original)


class ReachedFiniteOwner(Exception):
    pass


@pytest.mark.parametrize('supplied', [False, True])
@pytest.mark.parametrize('prepared_length', [None, 64])
def test_provider_gets_synchronized_factual_ids_and_preserves_owner_forks(monkeypatch, supplied, prepared_length):
    # Execute the real owner method up to its finite capture boundary. The
    # fixtures record calls only; they do not compute a model or a cache.
    torch = pytest.importorskip('torch')
    node = attribute_node(OWNER_PATH.read_text(encoding='utf8'))
    namespace = {'torch': torch, 'copy': copy, 'time': __import__('time')}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(OWNER_PATH), 'exec'), namespace)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda: 0)
    calls = []

    class CacheCalls:
        def reorder_cache(self, order):
            calls.append(('owner_reorder', order.tolist()))

        def __deepcopy__(self, memo):
            calls.append(('owner_fork',))
            return CacheCalls()

    class LayerCalls:
        block_type = 'linear_attention'

        def register_forward_pre_hook(self, hook, **kwargs):
            raise ReachedFiniteOwner

    def forward_root(**kwargs):
        calls.append(('native_forward', kwargs))
        return SimpleNamespace(past_key_values=CacheCalls())

    def synchronize_prefix_start(length):
        calls.append(('synchronize', length))
        return 64

    model = SimpleNamespace(
        model=SimpleNamespace(language_model=SimpleNamespace(
            layers=[LayerCalls() for _ in range(32)], norm=None)),
        forward_root=forward_root, synchronize_prefix_start=synchronize_prefix_start)
    runner = SimpleNamespace(model=model, defer_diagnostics=False,
        reuse_native_prefix=True, fa_coefficient_suffix=False, gdn_coefficient_suffix=False)
    factual = torch.arange(4 * 192).reshape(4, 192)
    reference = factual.clone()
    reference[:, 128:160] = -1
    paired = torch.stack((reference, factual), 1).flatten(0, 1)
    selection = SimpleNamespace(batch=4, length=192, positions=torch.tensor([180]))
    selection.suffix = lambda length: calls.append(('owner_suffix', length))

    def provider(ids):
        calls.append(('provider', ids.clone()))
        return CacheCalls()
    if prepared_length is not None:
        provider.prefix_length = prepared_length

    with pytest.raises(ReachedFiniteOwner):
        namespace['attribute'](runner, paired, torch.ones_like(paired), selection,
                               prefix_cache_provider=provider if supplied else None)
    assert calls[0] == ('synchronize', prepared_length if supplied and prepared_length is not None else 128)
    assert calls[-3:] == [('owner_reorder', [0, 0, 1, 1, 2, 2, 3, 3]),
                         ('owner_fork',), ('owner_suffix', 64)]
    if supplied:
        assert calls[1][0] == 'provider'
        assert torch.equal(calls[1][1], factual[:, :64])
        assert not any(c[0] == 'native_forward' for c in calls)
    else:
        assert calls[1][0] == 'native_forward'
        kwargs = calls[1][1]
        assert set(kwargs) == {'input_ids', 'use_cache', 'logits_to_keep'}
        assert torch.equal(kwargs['input_ids'], factual[:, :64])
        assert kwargs['use_cache'] is True and kwargs['logits_to_keep'] == 1


@pytest.mark.parametrize('supplied', [False, True])
def test_rl_boundary_forwards_only_explicit_provider_and_keeps_release(supplied):
    torch = pytest.importorskip('torch')
    from deltatrace_credit import trace_token_attribution
    calls = []
    provider = object()

    def attribute(pair, mask, selection, **kwargs):
        calls.append(kwargs)
        raise ReachedFiniteOwner

    runner = SimpleNamespace(attribute=attribute, model=SimpleNamespace(
        release_owner_params=lambda: calls.append('released')))
    ids = torch.ones((4, 8), dtype=torch.long)
    with pytest.raises(ReachedFiniteOwner):
        trace_token_attribution(runner, ids, ids, [{} for _ in range(4)], [[0]] * 4,
            packed_answer_targets=lambda *a, **k: object(),
            prefix_cache_provider=provider if supplied else None)
    expected = {'select_output_rows': True, 'observer': None}
    if supplied:
        expected['prefix_cache_provider'] = provider
    assert calls == [expected, 'released']
