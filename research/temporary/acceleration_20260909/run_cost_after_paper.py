"""Run one prepared cost experiment after the paper job succeeds and GPU is free.

This is a single dependency between existing experiments, not a metric/model
implementation. Keep a receipt and stop on failure; do not retry failed timings.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    scripts = ['benchmark_formal_task_groups.py', 'verify_formal_task_group_costs.py', 'run_cost_after_paper.py']
    sources = {name: sha(here / name) for name in scripts}
    plan = json.loads(args.plan.read_bytes())
    assert sha(args.quality) == plan['quality_report_sha256']
    assert sha(args.quality.parent / 'vectors.npz') == plan['quality_vectors_sha256']
    assert sha(args.release / 'experiments/official/batching.py') == plan['batching_sha256']
    assert sha(args.release / 'deltatrace/clean/sources.json') == plan['clean_sources_sha256']
    for model in json.loads((args.release / 'deltatrace/clean/sources.json').read_bytes())['models'].values():
        for name, row in model['files'].items():
            assert sha(args.release / name) == row['sha256'], name
    assert sha(args.release / 'deltatrace/accelerated/sources.json') == plan['acceleration_sources']['manifest_sha256']
    for name, digest in plan['acceleration_sources']['files'].items():
        assert sha(args.release / name) == digest, name
    state = {'status': 'waiting_for_paper', 'pid': os.getpid(), 'created': time.time(),
             'paper': str(args.paper), 'sources': sources, 'plan_sha256': sha(args.plan),
             'environment_sha256': sha(args.environment), 'benchmark_launched': False}

    def save():
        temp = args.output / 'controller.partial'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(args.output / 'controller.json')

    save()
    try:
        while True:
            paper = json.loads((args.paper / 'run/run.json').read_bytes())
            state.update(checked=time.time(), paper_status=paper['status'],
                         paper_current_task=paper.get('current_task'))
            if paper['status'] == 'failed':
                raise RuntimeError('Paper experiment failed; cost job was not launched.')
            if paper['status'] != 'complete':
                pid = paper['pid']
                try:
                    command = Path(f'/proc/{pid}/cmdline').read_bytes()
                except FileNotFoundError:
                    command = b''
                if not (b'run_qwen3_paper.py' in command and str(args.paper).encode() in command):
                    if json.loads((args.paper / 'run/run.json').read_bytes())['status'] == 'complete':
                        continue
                    raise RuntimeError('Paper controller is no longer live and is not complete; do not launch cost job.')
                state.update(status='waiting_for_paper', verified_live_paper_pid=pid)
                save()
                time.sleep(30)
                continue
            completed = [row for row in paper['attempts'] if row['status'] == 'complete']
            assert len(completed) == len(paper['tasks']) == 13 and paper['expected_cases'] == 1243
            assert {row['dataset'] for row in completed} == set(paper['tasks'])
            summary = json.loads((args.paper / 'publication/summary.json').read_bytes())
            if summary['status'] != 'complete':
                state['status'] = 'waiting_for_final_export'
                save()
                time.sleep(30)
                continue
            assert summary['completed_tasks'] == 13 and summary['completed_cases'] == 1243
            smi = subprocess.run(['/usr/bin/mx-smi'], capture_output=True, text=True, check=True).stdout
            assert '| Process:' in smi and 'End of Log' in smi
            section = smi.split('| Process:', 1)[1]
            pids = [int(match[1]) for match in re.findall(r'^\|\s+(\d+)\s+(\d+)\s+.*\|\s*$', section, re.M)]
            state.update(gpu_process_pids=pids, gpu_observation=smi)
            if pids:
                state['status'] = 'waiting_for_free_gpu'
                save()
                time.sleep(30)
                continue
            state.update(paper_run_sha256=sha(args.paper / 'run/run.json'),
                         paper_publication_sha256=sha(args.paper / 'publication/summary.json'))
            break
        for name, digest in sources.items():
            assert sha(here / name) == digest, name
        assert sha(args.plan) == state['plan_sha256'] and sha(args.environment) == state['environment_sha256']
        argv = [sys.executable, '-B', str(here / 'benchmark_formal_task_groups.py'),
                '--release', str(args.release), '--environment', str(args.environment),
                '--quality', str(args.quality), '--plan', str(args.plan),
                '--plan-sha256', state['plan_sha256'], '--output', str(args.output / 'run')]
        with (args.output / 'benchmark.log').open('w') as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            state.update(status='running_cost', benchmark_launched=True, child_pid=child.pid,
                         command=argv, started=time.time())
            save()
            code = child.wait()
        state.update(exit_code=code, ended=time.time())
        if code:
            raise RuntimeError(f'Cost experiment failed with exit code {code}; attempt preserved.')
        verify = [sys.executable, '-B', str(here / 'verify_formal_task_group_costs.py'),
                  '--results', str(args.output / 'run/results.json'), '--plan', str(args.plan),
                  '--quality', str(args.quality), '--output', str(args.output / 'summary.json')]
        with (args.output / 'verification.log').open('w') as log:
            subprocess.run(verify, stdout=log, stderr=subprocess.STDOUT, check=True)
        state.update(status='complete', results_sha256=sha(args.output / 'run/results.json'),
                     summary_sha256=sha(args.output / 'summary.json'))
    except Exception:
        state.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
