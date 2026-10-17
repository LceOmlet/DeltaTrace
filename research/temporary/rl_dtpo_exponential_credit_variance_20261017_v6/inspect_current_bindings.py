"""Use the same native metadata RPC for current v6/control worker bindings."""
import ast
import hashlib
from pathlib import Path
import sys

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT / 'receipts/appworld-formal-training-20261017-v1/source/inspect_current_workers.py'
code = source.read_text()
task = sys.argv.pop(1)
assert task in ('TextCraft', 'AppWorld')
code = code.replace("AUDIT = ROOT / 'receipts/appworld-formal-training-20261017-v1'",
                    "AUDIT = ROOT / 'receipts/dtpo-exponential-credit-variance-20261017-v6'")
code = code.replace("j.get('task') == 'AppWorld'", 'j.get(\'task\') == ' + repr(task))
code = code.replace("task='AppWorld'", 'task=' + repr(task))
code = code.replace("__file__, method, '--client'", '__file__, ' + repr(task) + ", method, '--client'")
code = code.replace("('worker-imports-request-' + method", "('worker-imports-request-' + " + repr(task) + " + '-' + method")
anchor = "exec(compile(ast.Module(body=[function], type_ignores=[]), str(wrapper), 'exec'), globals())"
assert code.count(anchor) == 1
extension = '''
inspect_native_owner = inspect_worker
def inspect_worker(self):
    result = inspect_native_owner(self)
    import os,torch
    result['actual_method_resource_environment'] = {key:os.environ.get(key) for key in (
        'DT_CREDIT_MODE','DT_NATIVE_HOST_MEMORY_LEASE','DT_NATIVE_HOST_CACHE_BATCH','VERL_ROOT','DT_ENTRY_ROOT','DT_ROOT')}
    if torch.cuda.is_initialized():
        stats=torch.cuda.memory.host_memory_stats()
        result['native_host_pinned_cache_bytes']={key:stats[key] for key in (
            'allocated_bytes.current','reserved_bytes.current','allocated_bytes.peak','reserved_bytes.peak')}
    return result
'''
code = code.replace(anchor, anchor + '\n' + extension)
exec(compile(code, str(source), 'exec'), globals())
