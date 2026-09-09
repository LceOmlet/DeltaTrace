"""Explicit, source-pinned diagnostic scheduling acceleration.

The clean release and previous acceleration backend remain separately usable.
Only DeltaTrace's own validation/diagnostic scheduling changes; model, FA, FLA,
finite operators, target, precision and public native calls retain provenance.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def _verify(root, family):
    path = root / 'deltatrace/accelerated/deferred_sources.json'
    raw = path.read_bytes()
    manifest = json.loads(raw)
    sha = lambda data: hashlib.sha256(data).hexdigest()
    assert sha((root / 'deltatrace/clean/sources.json').read_bytes()) == manifest['base_clean_sources_sha256']
    clean = json.loads((root / 'deltatrace/clean/sources.json').read_bytes())
    for name, record in clean['models'][family]['files'].items():
        assert sha((root / name).read_bytes()) == record['sha256'], name
    for name, digest in manifest['files'].items():
        assert sha((root / name).read_bytes()) == digest, name
    return {'manifest_sha256': sha(raw), 'files': manifest['files'], 'family': family}


def make_deferred_qwen3(root):
    """Return the measured B1/E2 finite propagation entry point and its identity."""
    receipt = _verify(root, 'qwen3')
    sys.path.insert(0, str(root / 'deltatrace/accelerated/qwen3'))
    from qwen3_deferred_pair import propagate_paired_secant
    for name in ('deferred_validation','qwen3_deferred_pair','qwen3_deferred_finite','qwen3_deferred_replay'):
        expected = (root / 'deltatrace/accelerated/qwen3' / (name + '.py')).resolve()
        assert Path(sys.modules[name].__file__).resolve() == expected, name
    return propagate_paired_secant, receipt


def make_deferred_qwen35(root, model, finite_fa, finite_fla):
    """Return the measured native padded B2 runner with deferred diagnostics."""
    receipt = _verify(root, 'qwen35')
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, root / 'deltatrace/accelerated/qwen35' / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    controller = load('dt_pinned_deferred_controller', 'controller_deferred.py')
    dynamic = load('dt_pinned_deferred_dynamic', 'dynamic_finite.py')
    runner = dynamic.configure_dynamic_finite(controller.Qwen35DenseFiniteRunner(
        model, finite_fa, finite_fla, checkpoint_device='cuda'))
    return runner, receipt
