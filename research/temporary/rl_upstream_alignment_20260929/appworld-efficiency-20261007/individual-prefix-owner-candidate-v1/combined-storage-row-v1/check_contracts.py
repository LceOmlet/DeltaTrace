"""Reuse frozen source/HF Cache contracts; optional CPU actual-ID dispatch spy.

No model, attention, FLA, DT target, backward, optimizer or GPU execution.
Actual 176 requests test owner grouping/identity only; small Cache data test
original-row mapping only. Neither substitutes for real B8 numerical checks.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent


def identity(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def storage_default_projection(tree, test_source):
    # Execute only the existing stdlib AST projection functions, not pytest,
    # Torch, fixture construction or any original tests at import time.
    nodes = ast.parse(test_source.read_bytes()).body
    names = {'_function', '_is_selected_condition', '_default_projection'}
    namespace = dict(ast=ast, copy=copy)
    code = ast.Module(body=[n for n in nodes if isinstance(n, ast.FunctionDef)
                           and n.name in names], type_ignores=[])
    exec(compile(code, str(test_source), 'exec'), namespace)
    result = namespace['_default_projection'](tree)
    helper = next(n for n in result.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_compose_row_prefix_cache')
    loop = next(n for n in helper.body if isinstance(n, ast.For))
    branch = loop.body[0]
    assert isinstance(branch, ast.If)
    body = []
    for node in branch.body:
        if isinstance(node, ast.If):
            assert isinstance(node.test, ast.Call) and node.test.func.id == 'all'
            body.extend(node.body)
        else:
            body.append(node)
    branch.body = body
    return result


def lease_storage_default_projection(tree):
    result = copy.deepcopy(tree)
    function = next(n for n in result.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'prepare_native_prefix_leases')
    for name in ('boundary_row_storage', 'observe_boundary_rows'):
        index = [n.arg for n in function.args.kwonlyargs].index(name)
        function.args.kwonlyargs.pop(index)
        function.args.kw_defaults.pop(index)
    class Disabled(ast.NodeTransformer):
        def visit_If(self, node):
            if isinstance(node.test, ast.Name) and node.test.id == 'boundary_row_storage':
                return [self.visit(child) for child in node.orelse]
            return self.generic_visit(node)
    return Disabled().visit(result)


def compact_cache_contracts(original):
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    owner = original.module(ROOT/'candidate/qwen35_native_prefix_artifacts.py', '_combined_compact_cache_owner')
    config = Qwen3_5TextConfig(num_hidden_layers=2, layer_types=['linear_attention', 'full_attention'])
    ids = torch.arange(4*256, dtype=torch.long).reshape(4, 256)
    keys = torch.arange(4*2*256*8, dtype=torch.float32).reshape(4, 2, 256, 8).to(torch.bfloat16)
    boundaries = {}
    for n in (64, 128, 192, 256):
        conv = (torch.arange(4*6*4, dtype=torch.float32).reshape(4, 6, 4)+n).to(torch.bfloat16)
        state = torch.arange(4*2*8*8, dtype=torch.float32).reshape(4, 2, 8, 8)+n
        boundaries[n] = (conv, state)
    full = owner.NativePrefixArtifacts(config, ids, [{'boundaries': boundaries}, {'keys': keys, 'values': -keys}])
    selected = {64: (3, 0), 128: (1,), 192: (3,), 256: (0,)}
    packed = {n: tuple(x[list(rows)].clone() for x in boundaries[n]) for n, rows in selected.items()}
    mapping = {n: {row: index for index, row in enumerate(rows)} for n, rows in selected.items()}
    compact = owner.NativePrefixArtifacts(config, ids, [{'boundaries': packed}, full.layers[1]], boundary_rows=mapping)
    rows, lengths = (3, 1, 3, 0), (64, 128, 192, 256)
    # Mixed full/compact sources and nonidentity stored-row indices.
    sources = [(compact, 3), (compact, 1), (full, 3), (compact, 0)]
    lease = owner.NativePrefixLease(sources, 256, prefix_lengths=lengths, context_lengths=(280, 290, 300, 310))
    result = lease(ids[list(rows)])
    fields = ((0, 'conv_states'), (0, 'recurrent_states'), (1, 'keys'), (1, 'values'))
    checks = []
    def equal_bytes(a, b):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    for index, (row, n) in enumerate(zip(rows, lengths)):
        expected = full.materialize(n, device='cpu', rows=[row])
        for layer, field in fields:
            actual = getattr(result.layers[layer], field)[index:index+1]
            wanted = getattr(expected.layers[layer], field)
            if layer == 1:
                assert equal_bytes(actual[:, :, :n], wanted)
                assert actual[:, :, n:].eq(0).all()
            else:
                assert equal_bytes(actual, wanted)
            checks.append(dict(consumer_row=index, original_row=row, boundary=n, field=field, bytewise_equal=True))
    second = lease(ids[list(rows)])
    assert second is not result
    result.update_conv_state(torch.zeros_like(result.layers[0].conv_states), 0)
    assert not equal_bytes(result.layers[0].conv_states, second.layers[0].conv_states)
    return dict(scope='Small CPU original HF DynamicCache field interface, no model/FA/FLA/DT acceptance.',
        checks=checks, fresh_mutable_cache=True, context_lengths=lease.context_lengths)


def actual_dispatch_contracts(args, original):
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    geometry = json.loads(args.geometry.read_bytes())
    assert identity(args.geometry)['sha256'] == 'a61c185751434edb4d2fbbac4305e3dbbcaf53c81f705693cacc9a75d42f97a0'
    raw_owner = original.module(args.row_contracts.parent/'candidate/qwen35_native_prefix_artifacts.py', '_raw_row_artifact')
    raw_lease = load(args.row_contracts.parent/'candidate/native_prefix_leases.py', '_raw_row_lease')
    combined_owner = original.module(ROOT/'candidate/qwen35_native_prefix_artifacts.py', '_combined_row_artifact')
    combined_lease = load(ROOT/'candidate/native_prefix_leases.py', '_combined_row_lease')
    config = Qwen3_5TextConfig(num_hidden_layers=2, layer_types=['linear_attention', 'full_attention'])
    runner = SimpleNamespace(model=SimpleNamespace(
        synchronize_prefix_start=lambda n: n, lm_head=SimpleNamespace(weight=torch.empty(1)),
        execution_device=torch.device('cpu'), _conditional=SimpleNamespace(config=config)))
    output = []
    for rank, meta in enumerate(geometry['ranks']):
        path = args.requests[rank] if args.requests else Path(meta['path'])
        assert identity(path)['sha256'] == meta['sha256'], path
        payload = torch.load(path, map_location='cpu', weights_only=False)
        requests = sorted(payload['requests'], key=lambda r: r['context_tokens'])
        assert len(requests) == 88
        for request, row in zip(requests, meta['rows']):
            assert (request['traj_uid'], request['source_step'], request['start'], request['end'],
                    request['context_tokens']) == (row['traj_uid'], row['source_step'], row['source_start'],
                    row['source_end'], row['context_tokens'])
            assert request['prompt'].numel() == row['prompt_tokens']
        def dispatch(owner, lease_module, *, storage):
            captures = []
            def capture_spy(model, ids, lengths, **kwargs):
                # Dispatch-only observation: this deliberately does not create
                # native activations, GDN states, fake target results or credit.
                artifact = SimpleNamespace(input_ids=ids.detach().clone(), config=kwargs['config'])
                artifact.capture_index = len(captures)
                mapping = kwargs.get('boundary_rows')
                artifact.boundary_rows = None if mapping is None else {
                    n: {row:index for index, row in enumerate(rows)} for n, rows in mapping.items()}
                captures.append(dict(artifact=artifact, lengths=tuple(lengths), rows=mapping,
                    ids_sha256=hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest(),
                    shape=list(ids.shape), keywords=tuple(kwargs)))
                return artifact
            keywords = dict(minibatch_size=4, eos_token_id=int(payload['eos_token_id']), individual_prefixes=True)
            if storage is not None:
                keywords['boundary_row_storage'] = storage
            with patch.dict(sys.modules, {'qwen35_native_prefix_artifacts': owner}), \
                    patch.object(owner.NativePrefixArtifacts, 'capture', capture_spy), \
                    patch.object(torch.cuda, 'synchronize', lambda: None):
                leases, report = lease_module.prepare_native_prefix_leases(runner, requests, **keywords)
            return captures, leases, report
        raw, raw_leases, raw_report = dispatch(raw_owner, raw_lease, storage=None)
        off, off_leases, off_report = dispatch(combined_owner, combined_lease, storage=False)
        on, on_leases, on_report = dispatch(combined_owner, combined_lease, storage=True)
        assert len(raw) == len(off) == len(on) == 5
        for a, b, c in zip(raw, off, on):
            assert (a['ids_sha256'], a['shape'], a['lengths']) == (b['ids_sha256'], b['shape'], b['lengths'])
            assert (a['ids_sha256'], a['shape'], a['lengths']) == (c['ids_sha256'], c['shape'], c['lengths'])
            assert b['keywords'] == ('config',)
        consumed = set()
        for index, (a, b, c) in enumerate(zip(raw_leases, off_leases, on_leases)):
            assert a.prefix_lengths == b.prefix_lengths == c.prefix_lengths
            assert a.context_lengths == b.context_lengths == c.context_lengths
            for row_index, ((source, origrow), n) in enumerate(zip(c.sources, c.prefix_lengths)):
                request = requests[index*4+row_index]
                assert torch.equal(source.input_ids[origrow, :n], request['prompt'][:n])
                assert source.boundary_rows[n][origrow] >= 0
                consumed.add((source.capture_index, n, origrow))
                for lease in (a, b):
                    src, row = lease.sources[row_index]
                    assert (src.capture_index, row) == (source.capture_index, origrow)
        selected = {(index, n, row) for index, capture in enumerate(on)
                    for n, rows in capture['rows'].items() for row in rows}
        assert selected == consumed
        assert len(consumed) == 88
        for key in raw_report:
            if key != 'capture_and_preparation_seconds':
                assert raw_report[key] == off_report[key] == on_report[key]
        output.append(dict(rank=rank, input=identity(path), actual_requests=len(requests),
            capture_rounds=len(on), capture_token_slots=on_report['capture_token_slots'],
            consumed_boundary_rows_per_gdn_layer=len(consumed), selected_rows_exactly_consumed=True,
            capture_literal_ids_boundaries_and_source_last_write_unchanged=True,
            original_owner_reports={k:v for k,v in on_report.items() if k != 'capture_and_preparation_seconds'}))
    return dict(scope='Actual 176 literal request CPU owner-dispatch spy. No native capture, model, FA/FLA, DT or numerical acceptance.', ranks=output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--row-contracts', type=Path, default=ROOT.parent/'native-representation-candidate/check_contracts.py')
    parser.add_argument('--storage-tests', type=Path, default=next(
        (p/'experiments/rl/test_native_prefix_boundary_rows.py' for p in ROOT.parents
         if (p/'experiments/rl/test_native_prefix_boundary_rows.py').exists()),
        ROOT/'test_native_prefix_boundary_rows.py'))
    parser.add_argument('--geometry', type=Path, default=next(
        (p/'experiments/rl/results_actual_prefix_request_geometry_20261004.json' for p in ROOT.parents
         if (p/'experiments/rl/results_actual_prefix_request_geometry_20261004.json').exists()),
        ROOT/'results_actual_prefix_request_geometry_20261004.json'))
    parser.add_argument('--requests', type=Path, nargs=2)
    parser.add_argument('--torch-interface', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'source-contracts.json')
    args = parser.parse_args()
    assert identity(args.row_contracts)['sha256'] == '3bf4fac86eb4657f427e1fdcb210926501fe7073607489e26d95b7eb129cfc93'
    original = load(args.row_contracts, '_frozen_row_contracts')
    original.ROOT = ROOT
    changes = []
    for name, artifact in [('qwen35_native_prefix_artifacts.py', True), ('native_prefix_leases.py', False)]:
        old = ast.parse((ROOT/'baseline'/name).read_bytes())
        new = ast.parse((ROOT/'candidate'/name).read_bytes())
        compile(new, name, 'exec')
        assert original.dump(old) == original.dump(original.project_default(new, artifact)), name
        storage_off = storage_default_projection(new, args.storage_tests) if artifact else lease_storage_default_projection(new)
        row_old = ast.parse((args.row_contracts.parent/'candidate'/name).read_bytes())
        assert original.dump(storage_off) == original.dump(row_old), name
        changes.append(dict(name=name, baseline=identity(ROOT/'baseline'/name), candidate=identity(ROOT/'candidate'/name),
            row_default_complete_verified_storage_AST_equal=True,
            storage_default_complete_uncompressed_row_AST_equal=True))
    result = dict(status='prepared_only_not_deployed', script=identity(Path(__file__)),
        reused_row_contracts=identity(args.row_contracts), reused_storage_AST_tests=identity(args.storage_tests),
        source_checks=changes, torch_CPU={'status':'not_run'}, operations=dict(remote=False, GPU=False,
            model=False, FA_FLA=False, DT=False, backward=False, optimizer=False),
        limitations=['AST projections are not numerical acceptance.',
            'CPU literal-ID dispatch spy does not capture activations or validate actual row runner/target kernels.',
            'Small HF DynamicCache byte checks do not replace actual same-capture B8 state transport and official operator assertions.'])
    if args.torch_interface:
        import os
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('MACA_VISIBLE_DEVICES') != '':
            raise ValueError('CPU contract invocation requires both device visibility variables empty')
        result['torch_CPU'] = dict(original_row_interface=original.real_cache_contracts(),
            storage_mapping_interface=compact_cache_contracts(original),
            actual_176_dispatch=actual_dispatch_contracts(args, original))
        import torch
        result['torch_CPU']['CUDA_initialized'] = torch.cuda.is_initialized()
        assert not torch.cuda.is_initialized()
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(output=str(args.output), sha256=identity(args.output)['sha256'], source_checks=changes)))


if __name__ == '__main__':
    main()
