"""Extend the pinned worker to compose its existing context across tool turns.

Single-call/default generation is unchanged. The owner collector explicitly
opens/closes the same sharding manager once per whole rollout; no lifecycle,
cache, sampler, weight loader or state-dictionary implementation is copied.
"""
from pathlib import Path
from patch_verl_environment_entry import replace_once


def patch_worker(source):
    anchor = ('    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)\n'
              '    def generate_sequences(self, prompts: DataProto):\n'
              '        # Support all hardwares\n')
    methods = '''    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def begin_rollout_context(self):
        if not getattr(self, "_owner_rollout_context_open", False):
            self.rollout_sharding_manager.__enter__()
            self._owner_rollout_context_open = True

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def end_rollout_context(self):
        if getattr(self, "_owner_rollout_context_open", False):
            try:
                self.rollout_sharding_manager.__exit__(None, None, None)
            finally:
                self._owner_rollout_context_open = False

'''
    source = replace_once(source, anchor, methods + anchor)
    source = replace_once(source,
        '        prompts.meta_info.update(meta_info)\n        with self.rollout_sharding_manager:\n',
        '        prompts.meta_info.update(meta_info)\n'
        '        from contextlib import nullcontext\n'
        '        context = (nullcontext() if getattr(self, "_owner_rollout_context_open", False)\n'
        '                   else self.rollout_sharding_manager)\n'
        '        with context:\n')
    return source


def patch_collector(source):
    source = replace_once(source, 'class TrajectoryCollector:',
        'from owner_rollout_scope import native_rollout_scope\n\n\nclass TrajectoryCollector:')
    return replace_once(source, '    def multi_turn_loop(\n',
                        '    @native_rollout_scope\n    def multi_turn_loop(\n')


def apply(root):
    for relative, patch in (
        ('verl/workers/fsdp_workers.py', patch_worker),
        ('agent_system/multi_turn_rollout/rollout_loop.py', patch_collector),
    ):
        path = Path(root) / relative
        source = patch(path.read_text())
        compile(source, str(path), 'exec')
        path.write_text(source, newline='\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    apply(parser.parse_args().root)
