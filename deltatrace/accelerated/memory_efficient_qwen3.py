"""Pinned Qwen3-8B attribution with one whole native graph and bounded reuse."""
import hashlib,importlib,json,sys
from pathlib import Path


def make_memory_efficient_qwen3(root,model,library,library_sha256):
    """Create a fixed-eval-model B1/E2 controller; first-call build is charged.

    library is the separately named shared-mean finite FA extension. The
    original model, full vocabulary and native FA implementation stay intact.
    Every call copies fresh token IDs, runs a complete root/finite graph and
    resolves the full signed vector and every strict diagnostic before return.
    close() releases its graph; the model and native allocator caches remain.
    """
    root=Path(root).resolve();sha=lambda data:hashlib.sha256(data).hexdigest()
    from retained import make_retained_qwen3
    _,base=make_retained_qwen3(root)
    raw=(root/'deltatrace/accelerated/memory_efficient_qwen3_sources.json').read_bytes()
    manifest=json.loads(raw)
    previous=base['manifest_sha256'];base_receipts=[]
    for entry in manifest['base_chain']:
        data=(root/entry['path']).read_bytes();assert sha(data)==entry['sha256']
        item=json.loads(data);assert item[entry['parent_field']]==previous
        for name,digest in item['files'].items():assert sha((root/name).read_bytes())==digest,name
        for name in item['modules']:
            module=importlib.import_module(name)
            assert Path(module.__file__).resolve()==root/'deltatrace/accelerated/qwen3'/(name+'.py'),name
        previous=sha(data);base_receipts.append({'manifest_sha256':previous,'files':item['files']})
    for name,digest in manifest['files'].items():assert sha((root/name).read_bytes())==digest,name
    from native_capture_events import check_runtime
    assert Path(sys.modules['native_capture_events'].__file__).resolve()==root/'deltatrace/accelerated/native_capture_events.py'
    monitor_check=check_runtime()
    from qwen3_memory_efficient_graph import SACFiniteGraphQwen3
    from qwen3_shared_mean_fa import VendorFAFiniteP1SharedMeanReuse
    for name in manifest['modules']:
        module=importlib.import_module(name)
        assert Path(module.__file__).resolve()==root/'deltatrace/accelerated/qwen3'/(name+'.py'),name
    assert Path(__file__).resolve()==root/'deltatrace/accelerated/memory_efficient_qwen3.py'
    assert library_sha256==manifest['finite_library_sha256']
    assert len(model.model.layers)==36 and model.config.hidden_size==4096
    assert model.config.num_attention_heads==32 and model.config.num_key_value_heads==8
    finite=VendorFAFiniteP1SharedMeanReuse(library,library_sha256)
    return SACFiniteGraphQwen3(model,finite),{
        'manifest_sha256':sha(raw),'files':manifest['files'],'base_retained':base,'base_chain':base_receipts,
        'family':'qwen3','finite_library_sha256':library_sha256,'capture_runtime':monitor_check,
        'scope':'Fixed unmodified eval Qwen3-8B, actual B2 endpoints, one geometry. Fresh original native root kernels on every GPU graph execution; original native decoder rematerialization with public same-root SAC; exact finite rules and complete CPU return. Original829 predicates plus36 same-input SAC predicates where reuse is active.',
        'graph_memory_accounting':'Warm allocated counters omit temporaries allocated in the captured pool. Use measured reserved footprint and separately reported cold/capture allocated peaks together; all build, copies and checks are charged.'}
