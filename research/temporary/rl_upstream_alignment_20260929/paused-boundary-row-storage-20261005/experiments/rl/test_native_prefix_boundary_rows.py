"""GDN row-storage interfaces; no FLA/DT numerical or speed acceptance claim."""
import ast
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
OWNER_PATH = Path(os.environ.get('DT_PREFIX_ARTIFACT_SOURCE',
    ROOT/'deltatrace/clean/qwen35/qwen35_native_prefix_artifacts.py'))
BASELINE_COMMIT = '8e7dd71'
BASELINE_SHA = '06fda7843c4120cb1406b671f06cb19f29921b61b5d524dc28539550446aef16'


def _baseline_source():
    path = os.environ.get('DT_PREFIX_BOUNDARY_BASELINE_SOURCE')
    data = Path(path).read_bytes() if path else subprocess.check_output([
        'git', 'show', BASELINE_COMMIT+
        ':deltatrace/clean/qwen35/qwen35_native_prefix_artifacts.py'], cwd=ROOT)
    assert hashlib.sha256(data).hexdigest() == BASELINE_SHA
    return data.decode('utf8')


def _function(nodes, name):
    return next(node for node in nodes if isinstance(node, ast.FunctionDef)
                and node.name == name)


def _is_selected_condition(node):
    return isinstance(node, ast.If) and isinstance(node.test, ast.Compare) \
        and isinstance(node.test.left, ast.Name) \
        and node.test.left.id == 'selected_rows'


def _default_projection(module):
    """Remove only the explicit new storage seam, retaining all owner math."""
    module = copy.deepcopy(module)
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef)
               and node.name == 'NativePrefixArtifacts')
    cls.body = [node for node in cls.body if not (
        isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        and node.target.id == 'boundary_rows')]
    capture = _function(cls.body, 'capture')
    index = [node.arg for node in capture.args.kwonlyargs].index('boundary_rows')
    capture.args.kwonlyargs.pop(index)
    capture.args.kw_defaults.pop(index)
    capture.body = [node for node in capture.body if not (
        (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == 'selected_rows' for target in node.targets))
        or (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == 'boundary_rows')
        or _is_selected_condition(node))]
    projection = _function(capture.body, 'projection')
    assert len(projection.body) == 1 and _is_selected_condition(projection.body[0])
    projection.body = projection.body[0].body
    end = _function(capture.body, 'end')
    native_branch = end.body[1]
    assert isinstance(native_branch, ast.If)
    loop = next(node for node in native_branch.body if isinstance(node, ast.For))
    replacement = []
    for node in loop.body:
        replacement.extend(node.body if _is_selected_condition(node) else [node])
    loop.body = replacement
    compose = _function(module.body, 'compose_native_prefix_cache')
    loop = next(node for node in compose.body if isinstance(node, ast.For))
    branch = loop.body[0]
    assert isinstance(branch, ast.If)
    replacement = []
    for node in branch.body:
        if isinstance(node, ast.If):
            assert isinstance(node.test, ast.Call) and node.test.func.id == 'all'
            replacement.extend(node.body)
        else:
            replacement.append(node)
    branch.body = replacement
    return module


def test_disabled_storage_seam_preserves_complete_recorded_owner_ast():
    baseline = ast.parse(_baseline_source())
    candidate = _default_projection(ast.parse(OWNER_PATH.read_text(encoding='utf8')))
    assert ast.dump(candidate, include_attributes=False) == ast.dump(
        baseline, include_attributes=False)


def _load_owner(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _owner_pair(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    baseline_path = tmp_path/'baseline_artifacts.py'
    baseline_path.write_text(_baseline_source(), encoding='utf8')
    return (_load_owner('_boundary_original_owner', baseline_path, monkeypatch),
            _load_owner('_boundary_candidate_owner', OWNER_PATH, monkeypatch))


def _fields(cache):
    return (cache.layers[0].conv_states, cache.layers[0].recurrent_states,
            cache.layers[1].keys, cache.layers[1].values)


def _artifacts(owner, config, torch, *, row_map=None, offset=0):
    ids = torch.arange(4*128).reshape(4, 128)+offset
    conv = torch.arange(4*8*4, dtype=torch.float32).reshape(4, 8, 4).to(torch.bfloat16)
    state = torch.arange(4*2*2*2, dtype=torch.float32).reshape(4, 2, 2, 2)
    keys = torch.arange(4*1*128*4, dtype=torch.float32).reshape(4, 1, 128, 4).to(torch.bfloat16)
    values = keys+1000
    boundaries = {64: (conv, state), 128: (conv+200, state+300)}
    kwargs = {}
    if row_map is not None:
        boundaries = {n: (boundaries[n][0][rows], boundaries[n][1][rows])
                      for n, rows in row_map.items() if rows}
        kwargs['boundary_rows'] = {n: {row: index for index, row in enumerate(rows)}
                                   for n, rows in row_map.items()}
    return owner.NativePrefixArtifacts(config, ids,
        [{'boundaries': boundaries}, {'keys': keys, 'values': values}], **kwargs)


@pytest.mark.parametrize('boundary, rows', [(64, [3, 1]), (128, [0, 2, 3])])
def test_real_hf_cache_compact_gdn_rows_equal_original_and_keep_fa_rows(
        tmp_path, monkeypatch, boundary, rows):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    original, candidate = _owner_pair(tmp_path, monkeypatch)
    config = transformers.Qwen3_5TextConfig(num_hidden_layers=2,
        layer_types=['linear_attention', 'full_attention'])
    full = _artifacts(original, config, torch)
    compact = _artifacts(candidate, config, torch,
        row_map={64: [1, 3], 128: [3, 0, 2]})
    expected = original.compose_native_prefix_cache(config,
        [(full, row) for row in rows], boundary, device='cpu')
    actual = candidate.compose_native_prefix_cache(config,
        [(compact, row) for row in rows], boundary, device='cpu')
    for left, right in zip(_fields(actual), _fields(expected)):
        assert left.dtype == right.dtype
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert compact.input_ids.shape == full.input_ids.shape
    assert compact.layers[1]['keys'].shape == full.layers[1]['keys'].shape
    assert compact.layers[0]['boundaries'][boundary][0].shape[0] < 4


def test_real_hf_cache_mixed_sources_repeated_lease_is_fresh_and_bank_unchanged(
        tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    original, candidate = _owner_pair(tmp_path, monkeypatch)
    config = transformers.Qwen3_5TextConfig(num_hidden_layers=2,
        layer_types=['linear_attention', 'full_attention'])
    original_a = _artifacts(original, config, torch)
    original_b = _artifacts(original, config, torch, offset=10000)
    compact = _artifacts(candidate, config, torch, row_map={64: [2, 0], 128: []})
    plain = _artifacts(candidate, config, torch, offset=10000)
    sources = [(compact, 0), (plain, 3), (compact, 2), (plain, 1)]
    lease = candidate.NativePrefixLease(sources, 64)
    ids = torch.stack([source.input_ids[row, :64] for source, row in sources])
    expected = original.compose_native_prefix_cache(config,
        [(original_a, 0), (original_b, 3), (original_a, 2), (original_b, 1)],
        64, device='cpu')
    first = lease(ids)
    for tensor in _fields(first):
        tensor.add_(100)
    second = lease(ids)
    assert second is not first
    for actual, wanted in zip(_fields(second), _fields(expected)):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    for source in (compact, plain):
        for index, layer in enumerate(source.layers):
            original_layer = (original_a if source is compact else original_b).layers[index]
            if index == 0:
                for n, pair in layer['boundaries'].items():
                    rows = [2, 0] if source is compact else list(range(4))
                    for actual, wanted in zip(pair, original_layer['boundaries'][n]):
                        torch.testing.assert_close(actual, wanted[rows], rtol=0, atol=0)
            else:
                for field in ('keys', 'values'):
                    torch.testing.assert_close(layer[field], original_layer[field], rtol=0, atol=0)
    with pytest.raises(KeyError):
        candidate.compose_native_prefix_cache(config, [(compact, 1)], 64, device='cpu')


@pytest.mark.parametrize('distributed_capture_rows', [None, 12])
def test_original_verl_padding_saves_only_final_uid_slot_and_needed_boundaries(
        tmp_path, monkeypatch, distributed_capture_rows):
    """Producer metadata contract with original VERL padding; no model numerics."""
    torch = pytest.importorskip('torch')
    pytest.importorskip('verl')
    candidate = _load_owner('qwen35_native_prefix_artifacts', OWNER_PATH, monkeypatch)
    from native_prefix_leases import prepare_native_prefix_leases
    captures = []
    def observe_capture(model, ids, lengths, **kwargs):
        artifact = SimpleNamespace(input_ids=ids.clone(), config=kwargs['config'])
        captures.append((artifact, list(lengths), kwargs['boundary_rows']))
        return artifact
    monkeypatch.setattr(candidate.NativePrefixArtifacts, 'capture', observe_capture)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    group = object()
    mesh = SimpleNamespace(ndim=1, size=lambda: 2, get_group=lambda: group)
    weight = SimpleNamespace(device_mesh=mesh) if distributed_capture_rows else torch.empty(1)
    if distributed_capture_rows:
        def all_reduce(value, *, op, group):
            assert op is torch.distributed.ReduceOp.MAX
            value.fill_(distributed_capture_rows)
        monkeypatch.setattr(torch.distributed, 'all_reduce', all_reduce)
    runner = SimpleNamespace(model=SimpleNamespace(lm_head=SimpleNamespace(weight=weight),
        execution_device=torch.device('cpu'), _conditional=SimpleNamespace(config=object()),
        synchronize_prefix_start=lambda n: n))
    requests = []
    for uid in range(6):
        for n in ((64, 192) if uid % 2 else (128, 192)):
            requests.extend(dict(traj_uid=str(uid), start=n,
                prompt=torch.arange(256)+uid*10000) for _ in range(4))
    leases, report = prepare_native_prefix_leases(runner, requests,
        minibatch_size=4, eos_token_id=99)
    assert len(captures) == report['capture_rounds'] == (3 if distributed_capture_rows else 2)
    assert report['local_unique_histories'] == 6
    consumed = {(id(source), lease.prefix_length, row)
                for lease in leases for source, row in lease.sources}
    selected = {(id(artifact), n, row) for artifact, _, mapping in captures
                for n, rows in mapping.items() for row in rows}
    assert selected == consumed
    assert len(selected) == 12  # Each of the six original histories needs two boundaries.
    assert all(artifact.input_ids.shape == (4, 192) for artifact, _, _ in captures)
    assert report['capture_token_slots'] == len(captures)*4*192
    for request, lease in zip(requests[::4], leases):
        assert lease.prefix_length == request['start']
        assert all(torch.equal(source.input_ids[row, :lease.prefix_length],
                               request['prompt'][:lease.prefix_length])
                   for source, row in lease.sources)
