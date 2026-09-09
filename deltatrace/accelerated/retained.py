"""Explicit Qwen3 replay tensor retention on the pinned native implementation.

Only DT's temporary replay operand copies are removed. Original root checkpoint
copies and compact logit storage remain. No native callable is replaced.
"""
import hashlib
import json
from pathlib import Path
import sys


def make_retained_qwen3(root):
    """Return the measured B1/E2 entry point and both provenance receipts."""
    root = Path(root)
    from deferred import make_deferred_qwen3
    _, base_receipt = make_deferred_qwen3(root)
    manifest_path = root / 'deltatrace/accelerated/retained_sources.json'
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    assert base_receipt['manifest_sha256'] == manifest['base_deferred_manifest_sha256']
    for name, digest in manifest['files'].items():
        assert sha((root / name).read_bytes()) == digest, name
    from qwen3_retained_pair import propagate_paired_secant
    for name in ('qwen3_retained_pair', 'qwen3_retained_replay'):
        expected = (root / 'deltatrace/accelerated/qwen3' / (name + '.py')).resolve()
        assert Path(sys.modules[name].__file__).resolve() == expected, name
    assert Path(__file__).resolve() == (root / 'deltatrace/accelerated/retained.py').resolve()
    return propagate_paired_secant, {'manifest_sha256': sha(raw), 'files': manifest['files'],
                                   'base_deferred': base_receipt, 'family': 'qwen3'}
