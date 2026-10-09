"""Prepare the fixed probability sample on existing remote CPU resources."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ENTRY,ROOT,SSH,SCP

REMOTE = ROOT+'/receipts/credit-tail-probability-sample-20261009-v1'
LOCAL = HERE/'tail-probability-sample-v1'


def main():
    LOCAL.mkdir(exist_ok=True)
    assert not (LOCAL/'sample.json').exists(), 'Use the existing sample; never redraw'
    names = ['prepare_tail_probability_sample.py','tail_probability_statistics.py',
        'test_tail_probability_statistics.py','corpus.json','manifest.json','credit-strata.json']
    source = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        files={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names})
    source_path = LOCAL/'source.json'
    source_path.write_text(json.dumps(source,indent=2)+'\n')
    subprocess.run(SSH+['bash','-s'],input=('set -eu\nmkdir -p '+REMOTE+'\n').encode(),check=True,timeout=30)
    for path in [HERE/n for n in names]+[source_path]:
        subprocess.run(SCP+[str(path),SSH[-1]+':'+REMOTE+'/'+path.name],check=True,timeout=45)
    command = ('set -eu\nsource '+ENTRY+'/metax-entry.env.sh\ncd '+REMOTE+'\n'
        'CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" test_tail_probability_statistics.py\n'
        'CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" prepare_tail_probability_sample.py'
        ' --directory '+REMOTE+' --output '+REMOTE+'/sample.json\n')
    (LOCAL/'prepare-command.sh').write_text(command,encoding='utf-8',newline='\n')
    run = subprocess.run(SSH+['bash','-s'],input=command.encode(),capture_output=True,timeout=90)
    (LOCAL/'prepare.stdout').write_bytes(run.stdout)
    (LOCAL/'prepare.stderr').write_bytes(run.stderr)
    print(run.stdout.decode(errors='replace'))
    if run.returncode:
        print(run.stderr.decode(errors='replace'))
    run.check_returncode()
    subprocess.run(SCP+[SSH[-1]+':'+REMOTE+'/sample.json',str(LOCAL/'sample.json')],check=True,timeout=45)
    expected = json.loads(run.stdout.splitlines()[-1])['output']
    assert hashlib.sha256((LOCAL/'sample.json').read_bytes()).hexdigest() == expected['sha256']


if __name__ == '__main__':
    main()
