"""Reapply the already verified DT storage options to this new actor lifetime.

The existing resource owners check their saved numerical/capacity evidence.
Only metadata-receipt identity changes; no new model/DT call is introduced.
"""
import ast
from pathlib import Path
import sys

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OWNERS = ROOT/'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4/source'


def rebind_metadata(source):
    tree = ast.parse(source)
    last = tree.body[-1]
    assert isinstance(last, ast.Expr) and isinstance(last.value, ast.Call)
    assert isinstance(last.value.func, ast.Name) and last.value.func.id == 'exec'
    lines = source.splitlines()
    lines.insert(last.lineno-1,
        "code = code.replace('textcraft-eight-gpu-v9-negative-clip2-20261019-v4', 'textcraft-v5-credit-integration-20261020-v1')")
    return '\n'.join(lines)+'\n'


for name in ['apply_root_pageable_resource.py', 'apply_small_gpu_resident_resource.py',
             'apply_validated_16384_gpu_resource.py']:
    owner = OWNERS/name
    code = owner.read_text()
    if name == 'apply_validated_16384_gpu_resource.py':
        # This owner composes the unchanged small-storage owner. Rebind only
        # that owner's metadata adapter, keeping all evidence paths intact.
        tree = ast.parse(code)
        lines = code.splitlines()
        lines.insert(tree.body[-1].lineno-1, 'code = rebind_metadata(code)')
        code = '\n'.join(lines)+'\n'
    else:
        code = rebind_metadata(code)
    sys.argv = [str(owner), 'dt', '--client']
    namespace = dict(__name__='__main__', __file__=str(owner), rebind_metadata=rebind_metadata)
    exec(compile(code, str(owner), 'exec'), namespace)
