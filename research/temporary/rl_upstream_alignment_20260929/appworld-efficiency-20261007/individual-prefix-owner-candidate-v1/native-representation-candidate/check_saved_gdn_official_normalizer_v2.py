"""Original saved GDN checker with only its normalization input boundary fixed.

The installed FLA l2norm_fwd owns normalization, exactly as in its production
chunk forward. Original recurrent reference and original o/ht assertion AST
are unchanged. This isolated diagnostic never changes a model or DT owner.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback


OLD_CHECKER_SHA = '5c1924ce50d797acf2b9f7ac1bb79a7adfd752c6c383dac160fff6fa695cd53a'
L2NORM_SHA = 'fca8a850419b44d60aa2cd7c84899c9de1a250896a3098bfe8e937854aa1bbc8'
CHUNK_SHA = 'fe10b455f8c71f4f845415f60a4385ff4f630336e5e46c592611765dfd8cce97'
EXPECTED_ASSIGNMENT = 'q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)'
OWNER_ASSIGNMENT = 'q, k = _official_l2norm_fwd(q)[0], _official_l2norm_fwd(k)[0]'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepared_function(original_checker, normalizer):
    path = Path(original_checker)
    if sha(path) != OLD_CHECKER_SHA:
        raise ValueError('Original GDN checker changed.')
    spec = importlib.util.spec_from_file_location('_preserved_saved_gdn_checker', path)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    parsed = ast.parse(path.read_bytes(), filename=str(path))
    source = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'check_saved_fla')
    candidate = copy.deepcopy(source)
    expected = ast.dump(ast.parse(EXPECTED_ASSIGNMENT).body[0], include_attributes=False)
    replacements = []

    class InputBoundary(ast.NodeTransformer):
        def visit_Assign(self, node):
            if ast.dump(node, include_attributes=False) == expected:
                replacements.append(node.lineno)
                return ast.copy_location(ast.parse(OWNER_ASSIGNMENT).body[0], node)
            return self.generic_visit(node)

    candidate = InputBoundary().visit(candidate)
    if len(replacements) != 1:
        raise ValueError('Original normalization boundary did not match exactly once.')
    namespace = dict(owner.__dict__, _official_l2norm_fwd=normalizer)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[candidate], type_ignores=[])),
                 str(path) + '::official-normalizer-input-boundary', 'exec'), namespace)
    provenance = dict(original_checker=dict(path=str(path), sha256=OLD_CHECKER_SHA),
        changed_statement_lines=replacements, original_statement=EXPECTED_ASSIGNMENT,
        replacement_statement=OWNER_ASSIGNMENT,
        original_function_ast_sha256=hashlib.sha256(ast.dump(source, include_attributes=False).encode()).hexdigest(),
        checked_function_ast_sha256=hashlib.sha256(ast.dump(candidate, include_attributes=False).encode()).hexdigest(),
        recurrent_reference_changed=False, original_o_ht_assertion_AST_changed=False)
    return namespace['check_saved_fla'], provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--official-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rank0', action='store_true')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '3':
        raise ValueError('Use the explicitly assigned physical GPU3 only.')
    physical = subprocess.check_output(['mx-smi'], text=True)
    if re.search(r'^\|\s+3\s+\d+\s+', physical.split('| Process:')[-1], re.M):
        raise RuntimeError('GPU3 is occupied; leave other processes unchanged.')
    if sha(args.parent / 'result.json') != 'f637f0e5c9747adae8ee85e5ea333d23e1d1bcadbc2183b197e49e33bcaf532e':
        raise ValueError('Completed real B8 parent result changed.')
    import torch
    import psutil
    from fla.modules.l2norm import l2norm_fwd
    import fla.ops.gated_delta_rule.chunk as chunk_owner
    normalizer_path = Path(inspect.getsourcefile(l2norm_fwd))
    chunk_path = Path(inspect.getsourcefile(chunk_owner))
    if sha(normalizer_path) != L2NORM_SHA or sha(chunk_path) != CHUNK_SHA:
        raise ValueError('Installed original FLA normalization/chunk owner changed.')
    checker, provenance = prepared_function(args.parent / 'observe_native_gdn0_operands.py', l2norm_fwd)
    report = dict(scope=__doc__, pid=os.getpid(), pid_birth=psutil.Process().create_time(),
        started_unix=time.time(), physical_device=3,
        helper_source=dict(path=__file__, sha256=sha(__file__)),
        normalization=dict(path=str(normalizer_path), sha256=L2NORM_SHA,
            original_signature=str(inspect.signature(l2norm_fwd)),
            production_chunk_source=dict(path=str(chunk_path), sha256=CHUNK_SHA)),
        reference_interface=provenance, ranks=[], status='running',
        numeric_correction=False, tolerance_changed=False, model=False, checkpoint=False,
        whole_DT_acceptance=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')

    save()
    try:
        for rank in ([0] if args.rank0 else [0, 1]):
            record = json.loads((args.parent / f'rank{rank}.json').read_bytes())
            operand = record['native_gdn0']
            if sha(operand['path']) != operand['sha256']:
                raise ValueError('Actual saved GDN operands changed.')
            saved = torch.load(operand['path'], map_location='cpu', weights_only=True, mmap=True)
            actual_finite = {name: bool(torch.isfinite(value).all())
                             for name, value in saved['tensors'].items() if value is not None}
            del saved
            result = checker(operand['path'], args.official_source, device='cuda')
            report['ranks'].append(dict(rank=rank, all_actual_saved_tensors_finite=all(actual_finite.values()),
                actual_finite=actual_finite, original_checker_result=result))
            save()
        report['status'] = 'passed'
    except BaseException:
        report['status'] = 'failed'
        path = args.output.with_suffix('.failure-traceback.txt')
        path.write_text(traceback.format_exc(), encoding='utf8')
        report['failure_traceback'] = dict(path=str(path), sha256=sha(path))
        raise
    finally:
        torch.cuda.synchronize()
        report.update(finished_unix=time.time(), torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      process_PSS_bytes=psutil.Process().memory_full_info().pss)
        save()
    print(json.dumps(dict(output=str(args.output), status=report['status'],
                         ranks=[row['rank'] for row in report['ranks']])), flush=True)


if __name__ == '__main__':
    main()
