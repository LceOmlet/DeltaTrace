"""Run frozen, paired source-v2 task shards and verify completed shards on resume."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OFFICIAL = ROOT / 'experiments/official'
sys.path.insert(0, str(OFFICIAL))
from evidence_protocol import SUPPORTED_TASKS
from summarize import summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['smoke', 'full'], required=True)
    parser.add_argument('--scope', choices=['all', 'vt_hotpot'], default='all')
    parser.add_argument('--task-order', nargs='+', choices=SUPPORTED_TASKS,
                        help='Change execution priority while retaining every predeclared full task.')
    args = parser.parse_args()
    tasks = ['niah_mq_q2', 'vt_h4_c1', 'hotpotqa_long'] if args.stage == 'smoke' else list(SUPPORTED_TASKS)
    if args.scope == 'vt_hotpot':
        assert args.stage == 'full'
        tasks = [t for t in SUPPORTED_TASKS if t.startswith('vt_') or t == 'hotpotqa_long']
    if args.task_order:
        assert args.stage == 'full' and len(args.task_order) == len(tasks)
        assert set(args.task_order) == set(tasks)
        tasks = args.task_order
    selection = 'smoke' if args.stage == 'smoke' else 'paper'
    args.output.mkdir(parents=True, exist_ok=True)
    fingerprint = {'protocol': digest(HERE / 'PROTOCOL.md'), 'environment': digest(args.environment),
        'code': {name: digest(OFFICIAL / name) for name in ['evaluate.py', 'evidence_protocol.py',
            'source_protocol.json', 'summarize.py', 'recovery_diagnostics.py']},
        'controller_sha256': digest(Path(__file__)), 'stage': args.stage, 'tasks': tasks,
        'scope': args.scope, 'focus_addendum_sha256': digest(HERE / 'FOCUS.md') if args.scope == 'vt_hotpot' else None}
    identity_path = args.output / 'plan_identity.json'
    if identity_path.exists():
        assert json.loads(identity_path.read_bytes()) == fingerprint, 'Refuse to mix code or experiment plans'
    else:
        identity_path.write_text(json.dumps(fingerprint, indent=2) + '\n', encoding='utf-8')
    state = {'status': 'running', 'plan': fingerprint, 'completed_tasks': [], 'current_task': None}
    state_path = args.output / 'suite_status.json'

    def save():
        temporary = state_path.with_suffix('.partial')
        temporary.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        temporary.replace(state_path)

    release = json.loads((OFFICIAL / 'protocol.json').read_bytes())
    environment = dict(os.environ, MACA_PATH='/opt/maca', TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS='0')
    try:
        for task in tasks:
            state['current_task'] = task
            save()
            target = args.output / task
            result = target / 'results.json'
            if not target.exists():
                command = [sys.executable, '-u', str(OFFICIAL / 'evaluate.py'), '--family', 'qwen3',
                    '--environment', str(args.environment.resolve()), '--selection', selection,
                    '--datasets', task, '--evaluation-protocol', 'source-v2', '--ft', 'live',
                    '--sentence-recovery', '--paired-reference-audit', '--output', str(target.resolve())]
                with (args.output / (task + '.log')).open('w', encoding='utf-8') as log:
                    process = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                if process.returncode != 0:
                    raise RuntimeError(f'{task} failed with exit code {process.returncode}; preserve and inspect the failed shard')
            report = json.loads(result.read_bytes())
            assert report['driver_sha256'] == fingerprint['code']['evaluate.py']
            assert report['evidence_protocol_sha256'] == fingerprint['code']['evidence_protocol.py']
            assert report['evaluation_protocol_sha256'] == fingerprint['code']['source_protocol.json']
            assert report['recovery_diagnostics_sha256'] == fingerprint['code']['recovery_diagnostics.py']
            assert report['selection'] == selection and report['paired_reference_audit']
            assert report['selected_counts'] == {task: 1 if selection == 'smoke' else release['tasks'][task]['count']}
            assert digest(target / 'vectors.npz') == report['vectors_sha256']
            summary = summarize(report, release)
            (target / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            state['completed_tasks'].append({'dataset': task, 'results_sha256': digest(result),
                'vectors_sha256': report['vectors_sha256'], 'cases': len(report['cases'])})
            save()
            print(json.dumps(state['completed_tasks'][-1]), flush=True)
        state.update(status='complete', current_task=None)
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
