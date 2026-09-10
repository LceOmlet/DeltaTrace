"""Source-pinned complete Qwen3 DT with native finite-program graph replay."""
import hashlib,json,sys
from pathlib import Path


def make_graphed_qwen3(root,model,library,library_sha256):
    """Return a B1/E2 controller caching one geometry for a fixed eval model.

    Every call executes a fresh original model root, refreshes all static graph
    inputs, replays the original finite program and resolves strict checks.
    First-call graph construction is inside attribute. close() releases state.
    """
    root=Path(root).resolve()
    from root_retained_qwen3 import make_root_retained_qwen3
    _,base=make_root_retained_qwen3(root,model,library,library_sha256)
    raw=(root/'deltatrace/accelerated/graphed_qwen3_sources.json').read_bytes()
    manifest=json.loads(raw);sha=lambda b:hashlib.sha256(b).hexdigest()
    assert base['manifest_sha256']==manifest['base_root_retained_manifest_sha256']
    for name,digest in manifest['files'].items():assert sha((root/name).read_bytes())==digest,name
    from qwen3_finite_graph import FiniteGraphQwen3
    from vendor_fa_finite_runtime import VendorFAFiniteP1
    for name in manifest['modules']:
        expected=(root/'deltatrace/accelerated/qwen3'/(name+'.py')).resolve()
        assert Path(sys.modules[name].__file__).resolve()==expected,name
    assert Path(__file__).resolve()==root/'deltatrace/accelerated/graphed_qwen3.py'
    assert not model.training and model.config._attn_implementation=='flash_attention_2'
    finite=VendorFAFiniteP1(library,library_sha256)
    return FiniteGraphQwen3(model,finite),{'manifest_sha256':sha(raw),'files':manifest['files'],
        'base_root_retained':base,'family':'qwen3','finite_library_sha256':library_sha256,
        'scope':'Fresh original B2 root on every call; 883 logical graph inputs refreshed through 658 native storages for the pinned model; one graph geometry, no result reuse. Original finite math, MM and FA unchanged.'}
