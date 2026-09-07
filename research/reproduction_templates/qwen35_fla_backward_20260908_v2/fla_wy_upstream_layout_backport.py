"""Backport two existing upstream WY expressions into pinned FLA0.4.1.

This is an explicit compiler-layout compatibility change, not a replacement
backward. No forward formula, accumulation dtype or reduction order is changed.
"""
import hashlib
import json
from pathlib import Path

ORIGINAL_SHA256='cc6c2bfdbf3b1a2c90bc0ef0c7325c53755a2a4fa440264cb39f4627d4c8de29'
UPSTREAM_COMMIT='9d981ffef3b361ba931102b633ae2a9fd91ca6c3'
UPSTREAM_SHA256='e11df00e9f3a996abc6d1d2ef4e03d3fb0d5a33b236e2da783ba24fe55e6fbc9'
REPLACEMENTS=[
    ('        b_ktb = b_kt * b_b[None, :]','        b_kb = b_k * b_b[:, None]'),
    ('        b_dk = b_dkb * b_b[:, None] + tl.trans(tl.dot(b_ktb.to(b_dA.dtype), b_dA))',
     '        b_dk = b_dkb * b_b[:, None] + tl.trans(tl.dot(tl.trans(b_kb).to(b_dA.dtype), b_dA))')]


def transformed_source(before,upstream):
    assert hashlib.sha256(before).hexdigest()==ORIGINAL_SHA256
    assert hashlib.sha256(upstream).hexdigest()==UPSTREAM_SHA256
    text=before.decode('utf-8');new=text
    for old,replacement in REPLACEMENTS:
        assert new.count(old)==1 and replacement.strip() in upstream.decode('utf-8')
        new=new.replace(old,replacement,1)
    boundary=text.index('def prepare_wy_repr_bwd_kernel(')
    assert text[:boundary]==new[:boundary]  # Forward code and decorators unchanged.
    restored_tail=new[boundary:]
    for old,replacement in reversed(REPLACEMENTS):restored_tail=restored_tail.replace(replacement,old,1)
    assert new[:boundary]+restored_tail==text
    import ast
    ast.parse(new)
    return new.encode('utf-8')


def apply_backport(environment,source,upstream_source,receipt_path):
    environment=Path(environment).resolve();source=Path(source).resolve()
    if not (environment/'pyvenv.cfg').is_file() or not source.is_relative_to(environment):
        raise ValueError('Use the explicitly selected isolated virtual environment.')
    before=source.read_bytes();upstream=Path(upstream_source).read_bytes()
    after=transformed_source(before,upstream)
    backup=source.with_name(source.name+'.before_deltatrace_wy_backport');receipt_path=Path(receipt_path)
    if backup.exists() or receipt_path.exists():raise FileExistsError('Backport already recorded; inspect state before retrying.')
    receipt={'kind':'two_upstream_backward_expression_backport','upstream_commit':UPSTREAM_COMMIT,
        'upstream_source_sha256':UPSTREAM_SHA256,'before_sha256':ORIGINAL_SHA256,
        'after_sha256':hashlib.sha256(after).hexdigest(),'changed_lines':2,
        'forward_and_decorators_unchanged':True,'runtime_verified':False,
        'identity':'Transpose commutes with row-wise scalar multiplication; same single multiply per element and same casts/dot operands.',
        'source_url':'https://github.com/fla-org/flash-linear-attention/blob/'+UPSTREAM_COMMIT+'/fla/ops/gated_delta_rule/wy_fast.py'}
    backup.write_bytes(before);source.write_bytes(after)
    receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt
