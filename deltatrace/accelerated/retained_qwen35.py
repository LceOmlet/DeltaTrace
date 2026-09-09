"""Pinned Qwen3.5 native replay operand retention on the existing B2 runner."""
import hashlib
import json
from pathlib import Path
import sys


def make_retained_qwen35(root, model, finite_fa, finite_fla):
    root = Path(root)
    from deferred import _verify
    base = _verify(root, 'qwen35')
    raw = (root / 'deltatrace/accelerated/retained_qwen35_sources.json').read_bytes()
    manifest = json.loads(raw)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    assert base['manifest_sha256'] == manifest['base_deferred_manifest_sha256']
    for name, digest in manifest['files'].items():
        assert sha((root / name).read_bytes()) == digest, name
    folder = root / 'deltatrace/accelerated/qwen35'
    sys.path.insert(0, str(folder))
    from qwen35_retained_controller import Qwen35DenseFiniteRunner
    import qwen35_retained_capture as capture
    from dynamic_finite import configure_dynamic_finite
    for name in ('qwen35_retained_controller', 'qwen35_retained_capture', 'dynamic_finite'):
        assert Path(sys.modules[name].__file__).resolve() == (folder / (name + '.py')).resolve(), name
    assert not capture.AUDIT, 'Mutation auditing is a separate charged research run.'
    assert Path(__file__).resolve() == (root / 'deltatrace/accelerated/retained_qwen35.py').resolve()
    runner = configure_dynamic_finite(Qwen35DenseFiniteRunner(model, finite_fa, finite_fla, checkpoint_device='cuda'))
    return runner, {'manifest_sha256': sha(raw), 'files': manifest['files'], 'base_deferred': base, 'family': 'qwen35'}
