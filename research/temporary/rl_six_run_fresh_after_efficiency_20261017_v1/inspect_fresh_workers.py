"""Reuse the accepted read-only worker metadata RPC for the fresh campaign."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT / 'receipts/textcraft-eight-gpu-sequential-20261017-v1/source/inspect_native_workers.py'
code = source.read_text()
code = code.replace('textcraft-eight-gpu-sequential-20261017-v1', 'textcraft-eight-gpu-sequential-20261017-v2')
code = code.replace("else 'eight-GPU-initial-v1')", "else 'fresh-eight-GPU-initial-v1')")
anchor = "    return result\n"
assert code.count(anchor) == 1
code = code.replace(anchor, '''    import inspect
    import torch
    runner = actor.rollout_sharding_manager.model_runner
    native = Path(inspect.getfile(type(runner)))
    result['native_generation_owner'] = str(native)
    result['native_generation_SHA256'] = hashlib.sha256(native.read_bytes()).hexdigest()
    result['native_allocator_settings'] = torch.cuda.memory._snapshot()['allocator_settings']
    result['read_only_generation_extension_model_calls'] = 0
    return result
''')
exec(compile(code, str(source), 'exec'), globals())
