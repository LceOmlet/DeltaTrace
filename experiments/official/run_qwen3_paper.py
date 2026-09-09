"""Run every released Table 1 task through the frozen Qwen3 DT entry.

Only orchestration is implemented here. Inputs, attribution and metrics remain
in evaluate.py and the pinned upstream functions. Completed tasks are verified
before reuse; failed attempts are retained in separate directories.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sha = lambda data: hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((HERE / 'protocol.json').read_bytes())
    sources = {name: sha((HERE / name).read_bytes()) for name in
               ('evaluate.py', 'score_views.py', 'protocol.json', 'summarize.py', 'run_qwen3_paper.py')}
    sources['clean_sources.json'] = sha((ROOT / 'deltatrace/clean/sources.json').read_bytes())
    state_path = args.output / 'run.json'
    if state_path.exists():
        state = json.loads(state_path.read_bytes())
        assert state['sources'] == sources, 'Resume requires identical experiment sources.'
        assert state['environment_sha256'] == sha(args.environment.read_bytes())
    else:
        state = {'status': 'prepared', 'family': 'qwen3', 'selection': 'paper',
                 'sources': sources, 'environment_sha256': sha(args.environment.read_bytes()),
                 'tasks': list(protocol['tasks']),
                 'expected_cases': sum(row['count'] for row in protocol['tasks'].values()),
                 'attempts': [], 'created': time.time(), 'ft': 'published',
                 'dt_backend': 'clean', 'sample_batch': 1}
    state.update(status='running', pid=os.getpid(), updated=time.time())
    write_json(state_path, state)
    child_env = os.environ.copy()
    child_env.update(MACA_PATH='/opt/maca', TRITON_ENABLE_PERSISTENT_AUTOTUNE_CONFIGS='0',
                     HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    for task in state['tasks']:
        attempts = [row for row in state['attempts'] if row['dataset'] == task]
        completed = [row for row in attempts if row['status'] == 'complete']
        if completed:
            assert len(completed) == 1
            old = completed[0]
            assert sha((args.output / old['result']).read_bytes()) == old['results_sha256']
            check = subprocess.run([sys.executable, '-B', str(HERE / 'summarize.py'),
                                    str(args.output / old['result'])], capture_output=True, text=True)
            check.check_returncode()
            continue
        output = args.output / task / ('attempt-%03d' % (len(attempts) + 1))
        output.parent.mkdir(exist_ok=True)
        argv = [sys.executable, '-B', str(HERE / 'evaluate.py'), '--family', 'qwen3',
                '--environment', str(args.environment), '--selection', 'paper',
                '--datasets', task, '--ft', 'published', '--dt-backend', 'clean',
                '--sample-batch', '1', '--output', str(output)]
        log_path = output.with_suffix('.log')
        with log_path.open('w', encoding='utf-8') as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=child_env)
            row = {'dataset': task, 'attempt': len(attempts) + 1, 'status': 'running',
                   'pid': child.pid, 'started': time.time(),
                   'result': str((output / 'results.json').relative_to(args.output)),
                   'log': str(log_path.relative_to(args.output))}
            state['attempts'].append(row)
            state['current_task'] = task
            write_json(state_path, state)
            code = child.wait()
        row.update(exit_code=code, ended=time.time(), status='failed' if code else 'verifying')
        if code == 0:
            check = subprocess.run([sys.executable, '-B', str(HERE / 'summarize.py'),
                                    str(output / 'results.json')], capture_output=True, text=True)
            if check.returncode:
                row.update(status='failed', verification_error=check.stderr)
            else:
                summary = json.loads(check.stdout)
                assert summary['published_number_comparison']
                assert len(summary['tasks']) == 1 and summary['tasks'][0]['dataset'] == task
                assert summary['tasks'][0]['count'] == protocol['tasks'][task]['count']
                write_json(output / 'summary.json', summary)
                row.update(status='complete', results_sha256=sha((output / 'results.json').read_bytes()),
                           vectors_sha256=sha((output / 'vectors.npz').read_bytes()),
                           summary_sha256=sha((output / 'summary.json').read_bytes()))
        state['updated'] = time.time()
        if row['status'] != 'complete':
            state['status'] = 'failed'
            write_json(state_path, state)
            raise SystemExit(1)
        write_json(state_path, state)
        print(json.dumps({'completed_task': task, 'count': protocol['tasks'][task]['count']}), flush=True)
    state.update(status='complete', ended=time.time(), current_task=None)
    write_json(state_path, state)


if __name__ == '__main__':
    main()
