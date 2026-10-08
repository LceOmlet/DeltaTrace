"""Read exact installed convolution owners and saved artifact structure only."""
import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import time
from pathlib import Path

import torch


def identity(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def structure(value, depth=0):
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype), stride=list(value.stride()))
    if depth < 4 and isinstance(value, dict):
        return {str(key):structure(item, depth+1) for key,item in value.items()}
    if depth < 4 and isinstance(value, (tuple,list)):
        return [structure(item,depth+1) for item in value]
    return str(type(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = dict(scope=__doc__, sources={}, artifact_structures={},
                  model_calls=0, DT_calls=0, optimizer=0, production_modified=False)
    def save(phase):
        result.update(phase=phase, observed_unix=time.time(), pid=os.getpid())
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    save('importing_original_owners')
    modeling = importlib.import_module('transformers.models.qwen3_5.modeling_qwen3_5')
    result['modeling_file'] = identity(modeling.__file__)
    result['import_availability'] = dict(causal_conv1d=modeling.causal_conv1d_fn is not None,
        chunk_gated_delta_rule=modeling.chunk_gated_delta_rule is not None,
        torch_cuda_available=torch.cuda.is_available())
    for label, function in [('actual_model_causal_conv',modeling.causal_conv1d_fn),
                            ('model_l2norm',modeling.chunk_gated_delta_rule)]:
        if function is None:
            continue
        path = inspect.getfile(inspect.unwrap(function))
        result['sources'][label] = dict(file=identity(path), qualname=function.__qualname__,
            source=inspect.getsource(inspect.unwrap(function)))
    conv = importlib.import_module(modeling.causal_conv1d_fn.__module__ if modeling.causal_conv1d_fn
                                   else 'causal_conv1d.causal_conv1d_interface')
    result['package_versions'] = {}
    for name in ('causal-conv1d','flash-linear-attention','transformers'):
        try:
            result['package_versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['package_versions'][name] = None
    result['native_extensions'] = {name:identity(getattr(conv,name).__file__)
        for name in ('causal_conv1d_cuda',) if hasattr(conv,name)}
    for name in ('causal_conv1d_fn','causal_conv1d_ref','CausalConv1dFn'):
        if hasattr(conv,name):
            function = getattr(conv,name)
            result['sources'][name] = dict(file=identity(inspect.getfile(function)),
                source=inspect.getsource(function))
    fla = importlib.import_module('fla.ops.gated_delta_rule.chunk')
    for name in ('l2norm_fwd','l2norm_bwd'):
        if hasattr(fla,name):
            function = getattr(fla,name)
            result['sources'][name] = dict(file=identity(inspect.getfile(function)),
                source=inspect.getsource(function))
    # The preceding read-only attempt stalled in a recursive NFS glob. Only
    # inspect the installed package's own test directory; no receipt-tree scan.
    tests = Path(conv.__file__).parent/'tests'
    result['installed_or_saved_tests'] = ([identity(path) for path in tests.glob('test*causal*conv*.py')]
        if tests.is_dir() else [])
    save('owner_interfaces_saved')
    for name in ('credit-single-background-nonfinite-appworld-20261009-v1',
                 'credit-single-background-nonfinite-appworld-20261009-v2'):
        folder = args.root/'receipts'/name/'results'
        for path in folder.glob('rank0-*.pt'):
            result['current_artifact'] = str(path)
            save('reading_saved_artifact_structure')
            saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            result['artifact_structures'][str(path)] = structure(saved)
            del saved
    result.pop('current_artifact', None)
    save('complete')
    print(json.dumps(dict(output=str(args.output), owners=list(result['sources']),
                         artifacts=list(result['artifact_structures']))))


if __name__ == '__main__':
    main()
