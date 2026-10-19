"""Read the accepted selector's counters with the existing native RPC."""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('inspect_small_gpu_resource_production.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '46c4aed0d95a6e3c4a18e7e617264d713df1cce0d26f6c14dd16b63ace5caf0f'
code = owner.read_text()
code = code.replace('small-GPU-resident-production-v1', 'validated-16384-GPU-production-v1')
code = code.replace('runtime-overrides/small-GPU-resident-resource-override-completed.json',
                    'runtime-overrides/validated-16384-GPU-resource-override-completed.json')
code = code.replace('small_GPU_resident_production', 'validated_GPU_resource_production')
exec(compile(code, str(owner), 'exec'), globals())
