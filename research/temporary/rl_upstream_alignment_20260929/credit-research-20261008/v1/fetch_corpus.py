"""Run the CPU-only inventory with the existing provisioned interpreter."""
from pathlib import Path
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from stage_environment_entry import SSH, SCP, ROOT, ENTRY

if __name__ == '__main__':
    remote = ROOT+'/receipts/credit-research-20261008-v1'
    run = subprocess.run(SSH+['mkdir', '-p', remote], capture_output=True, timeout=30)
    run.check_returncode()
    subprocess.run(SCP+[str(HERE/'collect_corpus.py'), SSH[-1]+':'+remote+'/collect_corpus.py'],
                   check=True, timeout=45)
    script = (f'source {ENTRY}/metax-entry.env.sh\n'
              f'CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" {remote}/collect_corpus.py '
              f'--output {remote}/corpus.json\n')
    (HERE/'collection-command.sh').write_text(script)
    run = subprocess.run(SSH+['bash', '-s'], input=script.encode(), capture_output=True, timeout=150)
    (HERE/'collection.stdout.txt').write_bytes(run.stdout)
    (HERE/'collection.stderr.txt').write_bytes(run.stderr)
    run.check_returncode()
    subprocess.run(SCP+[SSH[-1]+':'+remote+'/corpus.json', str(HERE/'corpus.json')],
                   check=True, timeout=45)
    print(run.stdout.decode())
