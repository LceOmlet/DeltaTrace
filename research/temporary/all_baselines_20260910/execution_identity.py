"""Exact execution identities plus explicitly audited historical compatibility."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def current_identity(preflight_hash):
    return dict(protocol_sha256=sha(HERE/'protocol.json'),driver_sha256=sha(HERE/'evaluate_baselines.py'),
        adapter_sha256=sha(HERE/'baseline_adapters.py'),preflight_sha256=preflight_hash,inputs_sha256=sha(HERE/'inputs.json'),
        lrp_numeric_fix_sha256=sha(HERE/'lrp_numeric_fix.py'),numeric_amendment_sha256=sha(HERE/'NUMERIC_FIX.md'),
        storage_amendment_sha256=sha(HERE/'STORAGE_FIX.md'),compatibility_sha256=sha(HERE/'execution_compatibility.json'),
        identity_module_sha256=sha(Path(__file__)))

def verify_identity(identity,method,preflight_hash,receipt_sha256=None):
    expected=current_identity(preflight_hash)
    if identity==expected:return 'current_cpu_saved_tensors'
    compatibility=json.loads((HERE/'execution_compatibility.json').read_bytes())
    for item in compatibility['accepted']:
        if identity==item['identity'] and method in item['methods']:
            assert receipt_sha256 in item['result_sha256'],'Unregistered historical case receipt'
            for k in ('protocol_sha256','inputs_sha256','preflight_sha256','numeric_amendment_sha256'):
                assert identity[k]==expected[k]
            return item['name']
    raise ValueError('Unregistered execution identity')
