"""Transport the frozen CPU query preparation, without recollecting the corpus."""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from stage_environment_entry import SSH, SCP, ROOT, ENTRY

if __name__ == '__main__':
    raise SystemExit('Withdrawn candidate: retain prior CPU artifacts; do not prepare or launch new queries.')
    remote = ROOT+'/receipts/credit-research-20261008-v1'
    for name in ('prepare_queries.py', 'manifest.json'):
        subprocess.run(SCP+[str(HERE/name), SSH[-1]+':'+remote+'/'+name], check=True, timeout=45)
    script = (f'source {ENTRY}/metax-entry.env.sh\n'
              f'CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" {remote}/prepare_queries.py '
              f'--corpus {remote}/corpus.json --manifest {remote}/manifest.json '
              f'--output {remote}/development-queries.json\n')
    (HERE/'queries-command.sh').write_text(script)
    run = subprocess.run(SSH+['bash', '-s'], input=script.encode(), capture_output=True, timeout=90)
    (HERE/'queries.stdout.txt').write_bytes(run.stdout)
    (HERE/'queries.stderr.txt').write_bytes(run.stderr)
    run.check_returncode()
    subprocess.run(SCP+[SSH[-1]+':'+remote+'/development-queries.json', str(HERE/'development-queries.json')],
                   check=True, timeout=45)
    print(run.stdout.decode())
