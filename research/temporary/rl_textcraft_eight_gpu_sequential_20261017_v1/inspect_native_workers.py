"""Reuse the existing read-only owner RPC for the eight-rank campaign identity."""
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source = ROOT / 'receipts/appworld-formal-training-20261017-v1/source/inspect_current_workers.py'
code = source.read_text()
code = code.replace("AUDIT = ROOT / 'receipts/appworld-formal-training-20261017-v1'",
                    "AUDIT = ROOT / 'receipts/textcraft-eight-gpu-sequential-20261017-v1'")
code = code.replace("('dt', 'ppo')", "('dt', 'ppo', 'grpo')")
code = code.replace("j.get('task') == 'AppWorld'", "j.get('task') == 'TextCraft'")
code = code.replace("task='AppWorld'", "task='TextCraft'")
code = code.replace("else 'formal-v1')", "else 'eight-GPU-initial-v1')")
code = code.replace('assert len(named) == 2', "assert len(named) == len(job['devices']) == 8")
anchor = "exec(compile(ast.Module(body=[function], type_ignores=[]), str(wrapper), 'exec'), globals())"
assert code.count(anchor) == 1
code = code.replace(anchor, anchor + '''
inspect_owner = inspect_worker
def inspect_worker(worker):
    result = inspect_owner(worker)
    import os
    actor, = [item for item in worker.worker_dict.values() if getattr(item, '_is_actor', False)]
    result['DT_NATIVE_HOST_MEMORY_LEASE'] = os.environ.get('DT_NATIVE_HOST_MEMORY_LEASE')
    result['DT_NATIVE_HOST_CACHE_REPLAY_LAYER'] = os.environ.get('DT_NATIVE_HOST_CACHE_REPLAY_LAYER')
    result['actor_optimizer_entries'] = len(actor.actor_optimizer.state)
    result['actor_scheduler_epoch'] = actor.actor_lr_scheduler.last_epoch
    result['DP_world_size'] = actor.world_size
    return result
''')
exec(compile(code, str(source), 'exec'), globals())
