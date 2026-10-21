"""Compose the three previously accepted DT storage mutations at initialization.

The original resource owners validate their saved evidence and supply the
mutation bodies. This bridge adds no attribution, model, allocator or selector
implementation. Call once after the native producer constructor and before its
first DT call; keep the worker's default path unchanged when the mode is absent.
"""
import ast
from functools import lru_cache
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

MODE = 'root_pageable_replay_pinned_resident16384_v1'
ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OWNERS = ROOT / 'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4/source'
BINDINGS = (
    ('apply_root_pageable_resource.py',
     'fa44985a1a78b54e00004a52f32a5a7713c3530146999c87c0dcd486f60df3a1'),
    ('apply_small_gpu_resident_resource.py',
     '8125b22b248cf200303d12b4bdfb449beefd1c54841378794d7fb102f542858a'),
    ('apply_validated_16384_gpu_resource.py',
     'f29bc0e7081b2b85273eb2d750d9925b19795de5820152783364914e0c1b2ad9'),
)


def _source(path, digest):
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == digest, str(path)
    return data.decode()


def _owner_prefix(source, path):
    """Run the original receipt checks/composition, excluding its final RPC."""
    tree = ast.parse(source)
    last = tree.body[-1]
    assert isinstance(last, ast.Expr) and isinstance(last.value, ast.Call)
    assert isinstance(last.value.func, ast.Name) and last.value.func.id == 'exec'
    namespace = dict(__name__='accepted_DT_storage_owner_composition', __file__=str(path))
    prefix = ast.Module(body=tree.body[:-1], type_ignores=[])
    exec(compile(prefix, str(path), 'exec'), namespace)
    return namespace


@lru_cache(maxsize=1)
def compile_original_mutations():
    callbacks, provenance = [], []
    for name, digest in BINDINGS:
        path = OWNERS / name
        namespace = _owner_prefix(_source(path, digest), path)
        if name == 'apply_validated_16384_gpu_resource.py':
            # This original owner composes the small-storage owner first.
            namespace = _owner_prefix(namespace['code'], OWNERS / BINDINGS[1][0])
        body = namespace['override']
        module = ast.parse('def apply_original_storage(actor, result):\n' + body)
        exec(compile(module, str(path), 'exec'), namespace)
        callbacks.append(namespace['apply_original_storage'])
        provenance.append(dict(path=str(path), SHA256=digest,
                               original_mutation_body_SHA256=hashlib.sha256(body.encode()).hexdigest()))
    return tuple(callbacks), provenance


def configure_existing_producer(actor):
    """Select only the accepted existing options on this actual native worker."""
    assert os.environ.get('DT_ACCEPTED_STORAGE_PROFILE') == MODE
    producer = actor._deltatrace_producer
    assert not hasattr(producer, '_accepted_storage_profile_receipt')
    callbacks, provenance = compile_original_mutations()
    result = {}
    for callback in callbacks:
        result = callback(actor, result)
    import psutil
    worker_path = Path(inspect.getfile(type(actor)))
    result.update(status='completed_accepted_DT_storage_initialization_before_first_DT',
                  storage_profile=MODE, applied_unix=time.time(),
                  pid=os.getpid(), birth=psutil.Process().create_time(), rank=actor.rank,
                  credit_mode=os.environ.get('DT_CREDIT_MODE'),
                  worker_source=str(worker_path),
                  worker_source_SHA256=hashlib.sha256(worker_path.read_bytes()).hexdigest(),
                  bridge_source=str(Path(__file__).resolve()),
                  bridge_source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  original_resource_owners=provenance,
                  model_calls=0, DT_calls=0, optimizer_updates=0,
                  scope='Existing storage options only; no new numerical or capacity claim.')
    receipt = Path(os.environ['DT_ACCEPTED_STORAGE_RECEIPTS']) / f'rank{actor.rank}.json'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    os.replace(temporary, receipt)
    producer._accepted_storage_profile_receipt = str(receipt)
    return result
