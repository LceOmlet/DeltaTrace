"""Source-pinned streaming input storage for original Qwen3 finite graphs."""
import hashlib,json,sys
from pathlib import Path


def make_streamed_qwen3(root,model,library,library_sha256):
    """Return a fixed-eval-model B1/E2 controller with one cached geometry.

    Every invocation runs the original model root. Native operand storages
    are copied once as each decoder finishes, retaining only graph-owned
    storage between layers. Cold graph building adopts these same storages.
    Complete finite math, precision, public FA and strict checks are inherited.
    """
    root=Path(root).resolve()
    sys.path.insert(0,str(root/'deltatrace/accelerated'))
    import native_capture_events
    assert Path(native_capture_events.__file__).resolve()==root/'deltatrace/accelerated/native_capture_events.py'
    from graphed_qwen3 import make_graphed_qwen3
    original,base=make_graphed_qwen3(root,model,library,library_sha256)
    raw=(root/'deltatrace/accelerated/graphed_qwen3_streamed_sources.json').read_bytes()
    manifest=json.loads(raw);sha=lambda data:hashlib.sha256(data).hexdigest()
    assert base['manifest_sha256']==manifest['base_graphed_manifest_sha256']
    for name,digest in manifest['files'].items():
        assert sha((root/name).read_bytes())==digest,name
    from qwen3_streamed_build_graph import StreamedBuildFiniteGraphQwen3
    for name in manifest['modules']:
        expected=root/'deltatrace/accelerated/qwen3'/(name+'.py')
        assert Path(sys.modules[name].__file__).resolve()==expected,name
    assert Path(__file__).resolve()==root/'deltatrace/accelerated/graphed_qwen3_streamed.py'
    return StreamedBuildFiniteGraphQwen3(model,original.finite),{
        'manifest_sha256':sha(raw),'files':manifest['files'],'base_graphed':base,
        'family':'qwen3','finite_library_sha256':library_sha256,
        'scope':'Fresh original B2 root; one copy per native storage as layers finish; cold graph adopts those buffers. Original finite graph and full checks, one geometry, no result reuse.'}
