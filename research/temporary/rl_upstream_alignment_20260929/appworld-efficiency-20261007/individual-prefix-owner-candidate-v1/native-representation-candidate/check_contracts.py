"""Source contracts locally; optional real Torch/HF Cache CPU interface check.

No attention, FLA, model forward or numerical acceptance is performed here.
"""
import argparse
import ast
import copy
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def dump(tree):
    return ast.dump(tree, include_attributes=False)


def project_default(tree, artifact):
    tree = copy.deepcopy(tree)
    if artifact:
        tree.body = [n for n in tree.body if not (
            isinstance(n, ast.FunctionDef) and n.name == '_compose_row_prefix_cache')]
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'compose_native_prefix_cache':
                node.args.kw_defaults.pop()
                node.args.kwonlyargs.pop()
                node.body.pop(1)  # docstring then new opt-in dispatch
            if isinstance(node, ast.ClassDef) and node.name == 'NativePrefixLease':
                node.body = [n for n in node.body if not (
                    isinstance(n, ast.AnnAssign) and n.target.id in ('prefix_lengths', 'context_lengths'))]
                method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__call__')
                method.body.pop(0)
    else:
        method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_native_prefix_leases')
        method.args.kwonlyargs.pop()
        method.args.kw_defaults.pop()
        body = []
        for node in method.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'row_prefix_lengths':
                continue
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'individual_prefixes':
                body.extend(node.orelse)
            else:
                body.append(node)
        method.body = body
    return tree


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def real_cache_contracts():
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.cache_utils import DynamicCache
    baseline = module(ROOT/'baseline/qwen35_native_prefix_artifacts.py', '_row_prefix_original_artifact')
    candidate = module(ROOT/'candidate/qwen35_native_prefix_artifacts.py', '_row_prefix_candidate_artifact')
    config = Qwen3_5TextConfig(num_hidden_layers=2, layer_types=['linear_attention', 'full_attention'])
    ids = torch.arange(4*256, dtype=torch.long).reshape(4, 256)
    keys = torch.arange(4*2*256*8, dtype=torch.float32).reshape(4, 2, 256, 8).to(torch.bfloat16)
    values = keys.neg()
    boundaries = {}
    for n in (64, 128, 192, 256):
        conv = (torch.arange(4*6*4, dtype=torch.float32).reshape(4, 6, 4)+n).to(torch.bfloat16)
        state = torch.arange(4*2*8*8, dtype=torch.float32).reshape(4, 2, 8, 8)+n
        boundaries[n] = (conv, state)
    original = baseline.NativePrefixArtifacts(config, ids, [{'boundaries': boundaries}, {'keys': keys, 'values': values}])
    compact = candidate.NativePrefixArtifacts(config, ids, original.layers)
    def equal_bytes(left, right):
        return (left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)))
    fields = ((0, 'conv_states'), (0, 'recurrent_states'), (1, 'keys'), (1, 'values'))
    shared_old = baseline.NativePrefixLease([(original, i) for i in range(4)], 128)(ids[:, :128])
    shared_new = candidate.NativePrefixLease([(compact, i) for i in range(4)], 128)(ids[:, :128])
    assert all(equal_bytes(getattr(shared_old.layers[i], field), getattr(shared_new.layers[i], field)) for i, field in fields)
    rows, lengths = (3, 1, 3, 0), (64, 128, 192, 256)
    lease = candidate.NativePrefixLease([(compact, row) for row in rows], 256, prefix_lengths=lengths)
    cache = lease(ids[list(rows)])
    assert isinstance(cache, DynamicCache)
    checks = []
    for index, (row, n) in enumerate(zip(rows, lengths)):
        expected = original.materialize(n, device='cpu', rows=[row])
        for layer_index, field in fields:
            actual = getattr(cache.layers[layer_index], field)[index:index+1]
            value = getattr(expected.layers[layer_index], field)
            if layer_index == 1:
                assert equal_bytes(actual[:, :, :n], value)
                assert actual[:, :, n:].eq(0).all()
            else:
                assert equal_bytes(actual, value)
            checks.append(dict(row=index, original_row=row, prefix_length=n, layer=layer_index, field=field))
    second = lease(ids[list(rows)])
    assert cache is not second
    cache.update_conv_state(torch.zeros_like(cache.layers[0].conv_states), 0)
    assert not equal_bytes(cache.layers[0].conv_states, second.layers[0].conv_states)
    bad = ids[list(rows)].clone()
    bad[0, 0] += 1
    try:
        lease(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched factual prefix IDs were accepted')
    return dict(status='CPU_cache_interface_checked', torch_version=torch.__version__,
                cache_source=str(Path(sys.modules[DynamicCache.__module__].__file__).resolve()),
                cache_sha256=hashlib.sha256(Path(sys.modules[DynamicCache.__module__].__file__).read_bytes()).hexdigest(),
                default_cache_bytewise_equal=True, original_boundary_row_checks=checks,
                fresh_mutable_cache_per_call=True, mismatched_literal_ID_rejected=True,
                scope='Small CPU interface data; no FA/FLA/DT, real B8, 32k or speed acceptance.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch-interface', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'source-contracts-context-lengths.json')
    args = parser.parse_args()
    changes = []
    for name, artifact in [('qwen35_native_prefix_artifacts.py', True), ('native_prefix_leases.py', False)]:
        old_path, new_path = ROOT/'baseline'/name, ROOT/'candidate'/name
        old, new = old_path.read_bytes(), new_path.read_bytes()
        old_ast, new_ast = ast.parse(old), ast.parse(new)
        compile(new_ast, str(new_path), 'exec')
        assert dump(old_ast) == dump(project_default(new_ast, artifact)), name
        changes.append(dict(name=name, baseline_sha256=hashlib.sha256(old).hexdigest(),
                            candidate_sha256=hashlib.sha256(new).hexdigest(), default_module_AST_equal=True))
        patch = ''.join(difflib.unified_diff(old.decode().splitlines(True), new.decode().splitlines(True),
                         fromfile='baseline/'+name, tofile='candidate/'+name))
        (ROOT/(name+'.patch')).write_text(patch, encoding='utf-8', newline='\n')
    data = dict(status='prepared_only', source_checks=changes,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                local_operations=dict(torch_import=False, CUDA=False, model_forward=False, FA_FLA=False),
                CPU_cache_interface={'status': 'not_run', 'reason': 'Local Python has no Torch; actual-owner CPU command prepared.'},
                contracts={'consumer_order': 'unchanged', 'B4_capture_and_MAX_count_collectives': 'unchanged source body',
                           'zero_common_prefix': 'same original no-cache branch',
                           'duplicate_padded_UID': 'same sources[uid] last-write code unchanged',
                           'target_and_runner': 'not modified',
                           'original_shared_prefix_token_slots': 'keeps scalar common-boundary denominator'},
                limits=['Source projection is not FA/FLA numerical acceptance.',
                        'Row KV holes require original HF mask/position and runner/target representation integration.',
                        'Native physical KV width may exceed logical 32k; geometry is separately recorded.'])
    if args.torch_interface:
        data['CPU_cache_interface'] = real_cache_contracts()
        data['local_operations']['torch_import'] = True
    args.output.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8', newline='\n')
    manifest = ROOT/'source-manifest.json'
    if not args.torch_interface:
        value = json.loads(manifest.read_text(encoding='utf-8'))
        value['source_changes'] = [{k: c[k] for k in ('name', 'baseline_sha256', 'candidate_sha256')} for c in changes]
        manifest.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(output=str(args.output), sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(), source_checks=changes)))


if __name__ == '__main__':
    main()
