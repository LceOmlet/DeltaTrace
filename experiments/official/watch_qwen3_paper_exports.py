"""Export newly completed paper tasks without occupying the attribution GPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    previous = None
    exporter = Path(__file__).resolve().with_name('export_qwen3_paper.py')
    while True:
        state = json.loads((args.run / 'run.json').read_bytes())
        completed = [(a['dataset'], a['results_sha256']) for a in state['attempts'] if a['status'] == 'complete']
        identity = (state['status'], completed)
        if completed and identity != previous:
            argv = [sys.executable, '-B', str(exporter), '--run', str(args.run), '--output', str(args.output)]
            if state['status'] != 'complete':
                argv.append('--allow-partial')
            subprocess.run(argv, check=True)
            previous = identity
        if state['status'] == 'complete':
            return
        if state['status'] == 'failed':
            raise SystemExit('The experiment failed; completed tasks have been exported, but the full table is incomplete.')
        time.sleep(30)


if __name__ == '__main__':
    main()
