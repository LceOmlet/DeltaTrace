"""Explicit scheduling-only FLA0.4.1 adaptation for a dedicated MetaX venv.

One original autotune list is changed to num_stages=[1]. All function bodies
remain byte-for-byte/AST identical. This is not a replacement state kernel.
"""
import ast
import hashlib
import json
from pathlib import Path

ORIGINAL_SHA256 = 'be25cba5e99073465c0653f50cd7c1c96466335e541a76cdfe449992d4146e01'


def apply_schedule(environment, source, receipt_path):
    environment, source = Path(environment).resolve(), Path(source).resolve()
    if not (environment / 'pyvenv.cfg').is_file() or not source.is_relative_to(environment):
        raise ValueError('Target must belong to the explicitly selected virtual environment')
    before = source.read_bytes()
    if hashlib.sha256(before).hexdigest() != ORIGINAL_SHA256:
        raise ValueError('Not the pinned original FLA0.4.1 chunk_delta_h.py')
    text = before.decode()
    boundary = text.index('def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(')
    prefix = text[:boundary]
    old, new = 'for num_stages in [2, 3, 4]', 'for num_stages in [1]'
    assert prefix.count(old) == 1
    after = (prefix.replace(old, new, 1) + text[boundary:]).encode()
    assert before[boundary:] == after[after.index(b'def chunk_gated_delta_rule_fwd_kernel_h_blockdim64('):]
    a, b = ast.parse(before), ast.parse(after)
    def bodies(tree):
        return {n.name: ast.dump(ast.Module(body=n.body, type_ignores=[])) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert bodies(a) == bodies(b)
    backup = source.with_name(source.name + '.before_deltatrace_maca_schedule')
    receipt_path = Path(receipt_path)
    if backup.exists() or receipt_path.exists():
        raise FileExistsError('Existing scheduling receipt or backup; inspect before retrying')
    receipt = {'kind': 'explicit_one_autotune_list_change', 'kernel': 'chunk_gated_delta_rule_fwd_kernel_h_blockdim64',
        'before_sha256': ORIGINAL_SHA256, 'after_sha256': hashlib.sha256(after).hexdigest(),
        'num_stages_before': [2, 3, 4], 'num_stages_after': [1], 'function_bodies_identical': True,
        'model_changes': 0, 'backward_changes': 0, 'runtime_verified': False,
        'related_report': 'https://github.com/flagos-ai/vllm-plugin-FL/issues/93',
        'related_report_limit': 'Different vLLM stack; motivates the scheduling trial but does not validate this FLA package.'}
    backup.write_bytes(before); source.write_bytes(after)
    receipt_path.write_text(json.dumps(receipt, indent=2))
    return receipt
