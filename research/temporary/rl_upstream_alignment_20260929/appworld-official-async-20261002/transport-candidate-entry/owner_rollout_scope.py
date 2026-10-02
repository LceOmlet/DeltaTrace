"""Keep the original VERL engine context around one complete owner rollout.

Environment loops, vLLM scheduling, LoRA synchronization and RNG restoration
remain in their owning implementations. This only matches their call scopes.
"""
from functools import wraps


def native_rollout_scope(method):
    @wraps(method)
    def collect(self, gen_batch, actor_rollout_wg, *args, **kwargs):
        manager = getattr(self, 'async_rollout_manager', None)
        begin = actor_rollout_wg.begin_rollout_context if manager is None else manager.wake_up
        end = actor_rollout_wg.end_rollout_context if manager is None else manager.sleep
        begin()
        try:
            return method(self, gen_batch, actor_rollout_wg, *args, **kwargs)
        finally:
            end()
    return collect
