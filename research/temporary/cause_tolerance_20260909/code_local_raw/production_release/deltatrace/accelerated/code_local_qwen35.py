"""Explicit Qwen3.5 runner with code-local passive capture (CPython >=3.12)."""
import hashlib
import json
from pathlib import Path
import sys


def make_code_local_qwen35(root,model,finite_fa,finite_fla):
    root=Path(root)
    from retained_qwen35 import make_retained_qwen35
    retained,base=make_retained_qwen35(root,model,finite_fa,finite_fla)
    del retained
    raw=(root/'deltatrace/accelerated/code_local_qwen35_sources.json').read_bytes()
    manifest=json.loads(raw)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    assert base['manifest_sha256']==manifest['base_retained_manifest_sha256']
    for name,digest in manifest['files'].items():
        assert sha((root/name).read_bytes())==digest,name
    folder=root/'deltatrace/accelerated/qwen35'
    from qwen35_code_local_controller import Qwen35DenseFiniteRunner
    from dynamic_finite import configure_dynamic_finite
    from native_capture_events import LocalCaptureEvents
    assert hasattr(sys,'monitoring'), 'Code-local capture requires CPython 3.12 or newer.'
    for name in ('qwen35_code_local_controller','qwen35_code_local_capture'):
        assert Path(sys.modules[name].__file__).resolve()==(folder/(name+'.py')).resolve(),name
    assert Path(sys.modules['native_capture_events'].__file__).resolve()==(folder.parent/'native_capture_events.py').resolve()
    assert Path(__file__).resolve()==(folder.parent/'code_local_qwen35.py').resolve()
    runner=configure_dynamic_finite(Qwen35DenseFiniteRunner(model,finite_fa,finite_fla,checkpoint_device='cuda'))
    return runner,{'manifest_sha256':sha(raw),'files':manifest['files'],'base_retained':base,'family':'qwen35'}
