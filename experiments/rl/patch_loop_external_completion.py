"""Inject an existing completion client into LOOP's original rollout worker.

The original sampler, runner pool, queue, collection thresholds and cancellation
are unchanged. The optional dependency bypasses only LOOP-owned GPU servers;
VERL owns the existing policy and its inference lifecycle.
"""
from pathlib import Path
from patch_verl_environment_entry import replace_once


def patch(source):
    source = replace_once(source,
        '        num_runners: int | None = None,\n',
        '        num_runners: int | None = None,\n'
        '        external_llm_factory=None,\n'
        '        external_cancellation_event=None,\n')
    source = replace_once(source,
        '        if not ray.is_initialized():\n',
        '        self._external_llm_factory = external_llm_factory\n'
        '        if external_llm_factory is None and not ray.is_initialized():\n')
    source = replace_once(source,
        '        self._cancellation_event = threading.Event()\n',
        '        self._cancellation_event = (threading.Event() if external_cancellation_event is None\n'
        '                                    else external_cancellation_event)\n')
    begin = '        # submit an asynchronous task for reach rollout\n'
    end = '        # mapping from rollout generation tasks to tuples (scenario_idx, scenario, runner_idx)\n'
    if '        if self._external_llm_factory is not None:\n' not in source:
        a, b = source.index(begin), source.index(end)
        original = source[a:b]
        replacement = ('        if self._external_llm_factory is not None:\n'
                       '            llms = self._external_llm_factory(self._cancellation_event)\n'
                       '        else:\n' + ''.join('    '+line if line.strip() else line
                                                   for line in original.splitlines(keepends=True)))
        source = source[:a] + replacement + source[b:]
    start = '        # make sure all processes finished all other work before we load new weights\n'
    end = '        # actually ask the worker thread to start working on new rollouts\n'
    marker = '        if self._external_llm_factory is None:\n' + '    ' + start
    if marker not in source:
        a, b = source.index(start), source.index(end)
        original = source[a:b]
        replacement = ('        if self._external_llm_factory is None:\n' +
                       ''.join('    '+line if line.strip() else line
                               for line in original.splitlines(keepends=True)))
        source = source[:a] + replacement + source[b:]
    compile(source, '<patched LOOP rollout worker>', 'exec')
    return source


def apply(root):
    path = Path(root)/'phi_agents/rl/vllm_rollout_worker.py'
    path.write_text(patch(path.read_text()), newline='\n')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    apply(p.parse_args().root)
