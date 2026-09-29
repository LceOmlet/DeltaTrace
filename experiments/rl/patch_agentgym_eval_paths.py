"""Repair the pinned author's demonstrated evaluation working-directory error.

After `cd AgentGym-RL`, the example enters `AgentGym-RL/scripts` a second time.
Keep the already selected project directory and call its original merger by
relative path. No evaluation parameter, merger, environment or trainer changes.
"""
from pathlib import Path
import argparse


def patch(text: str) -> str:
    old = 'cd AgentGym-RL/scripts\npython model_merger.py'
    new = 'python scripts/model_merger.py'
    if new in text and old not in text:
        return text
    if text.count(old) != 1:
        raise ValueError('Expected the pinned AgentGym-RL evaluation path defect once')
    return text.replace(old, new, 1)


def forward_native_overrides(text: str) -> str:
    """Expose the native Hydra CLI for model/path/resource overrides only.

    The pinned scripts otherwise discard all command-line arguments, leaving
    Qwen2.5's literal path in place. No default parameter is changed here.
    """
    if '    "$@"\nstatus=$?' in text:
        return text
    anchor = '\nstatus=$?'
    if text.count(anchor) != 1:
        raise ValueError('Expected one native trainer command status boundary')
    return text.replace(anchor, ' \\\n    "$@"\nstatus=$?', 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('owner_root', type=Path)
    args = parser.parse_args()
    for task in ('textcraft', 'webarena'):
        path = args.owner_root / f'examples/eval/{task}_eval.sh'
        path.write_text(forward_native_overrides(patch(path.read_text(encoding='utf-8'))), encoding='utf-8', newline='\n')
        train = args.owner_root / f'examples/train/AgentGym-RL/{task}_train.sh'
        train.write_text(forward_native_overrides(train.read_text(encoding='utf-8')), encoding='utf-8', newline='\n')
