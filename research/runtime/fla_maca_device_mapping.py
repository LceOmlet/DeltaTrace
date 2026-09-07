"""Explicit, source-pinned FLA0.4.1 device mapping for a dedicated MetaX environment.

This changes device discovery only. The real Triton backend remains ``maca``;
it does not alias torch.maca, replace kernels, or mark MetaX as NVIDIA.
Successful application does not establish kernel or model compatibility.
"""
import argparse
import hashlib
import json
from pathlib import Path

ORIGINAL_SHA256 = '03e3e41a62741d45913a2d2d35de630aaba13dcbb0fb1e8c1f7904446a688cf1'
REPLACEMENTS = (
    ("return {'cuda': 'cuda', 'hip': 'cuda', 'xpu': 'xpu'}.get(backend, backend)",
     "return {'cuda': 'cuda', 'hip': 'cuda', 'xpu': 'xpu', 'maca': 'cuda'}.get(backend, backend)"),
    ("device = get_available_device() if get_available_device() != 'hip' else 'cuda'",
     'device = map_triton_backend_to_torch_device()'),
)


def apply_mapping(environment, source, receipt_path):
    environment, source = Path(environment).resolve(), Path(source).resolve()
    if not (environment / 'pyvenv.cfg').is_file() or not source.is_relative_to(environment):
        raise ValueError('Target must belong to the explicitly selected virtual environment')
    before = source.read_bytes()
    if hashlib.sha256(before).hexdigest() != ORIGINAL_SHA256:
        raise ValueError('Not the pinned unmodified FLA0.4.1 utils.py; no changes applied')
    text = before.decode('utf-8')
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise ValueError('Unexpected source structure; no changes applied')
        text = text.replace(old, new, 1)
    after = text.encode('utf-8')
    receipt = {'kind': 'explicit_device_name_mapping_only', 'upstream_version': 'fla-core0.4.1',
               'before_sha256': ORIGINAL_SHA256, 'after_sha256': hashlib.sha256(after).hexdigest(),
               'replacements': list(REPLACEMENTS), 'kernel_changes': 0, 'model_changes': 0,
               'runtime_verified': False, 'real_backend_identity': 'maca'}
    backup = source.with_name('utils.py.before_deltatrace_maca_mapping')
    if backup.exists() or Path(receipt_path).exists():
        raise FileExistsError('Existing mapping receipt or backup; inspect it instead of overwriting')
    backup.write_bytes(before)
    source.write_bytes(after)
    Path(receipt_path).write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    print(json.dumps(apply_mapping(args.environment, args.source, args.receipt)))
