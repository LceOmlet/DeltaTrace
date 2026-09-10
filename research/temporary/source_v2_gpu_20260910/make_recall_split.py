"""Freeze deterministic candidate-development and validation case identities."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    release = json.loads((ROOT / 'experiments/official/protocol.json').read_bytes())
    tasks = ['vt_h2_c3', 'vt_h4_c1', 'vt_h6_c1', 'vt_h10_c1', 'hotpotqa_long']
    split = {'version': 'recall-pilot-v1', 'plan_sha256': hashlib.sha256(
        (HERE / 'RECALL_PILOT.md').read_bytes()).hexdigest(), 'tasks': {}}
    for task in tasks:
        order = sorted(range(release['tasks'][task]['count']), key=lambda i:
            hashlib.sha256(f'recall-pilot-v1|{task}|{i}'.encode()).hexdigest())
        split['tasks'][task] = {'development': sorted(order[:8]) if task in
            ('vt_h4_c1', 'hotpotqa_long') else [], 'validation': sorted(order[8:24]),
            'cache_sha256': release['tasks'][task]['cache_sha256']}
    path = HERE / 'recall_split.json'
    if path.exists():
        assert json.loads(path.read_bytes()) == split, 'Do not alter an existing split'
    else:
        path.write_text(json.dumps(split, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(split, indent=2))


if __name__ == '__main__':
    main()
