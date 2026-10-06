"""Compose the existing CPU artifact-analysis launcher for this diagnostic."""
import ast
import hashlib
import json
import subprocess

import run_textcraft_native_adam_analysis as prior
from stage_environment_entry import AUDIT, SSH, SCP
from stage_textcraft_label_gradients import OUT, LOCAL

NAME = 'analyze_textcraft_label_gradients.py'


if __name__ == '__main__':
    sha = hashlib.sha256((AUDIT / NAME).read_bytes()).hexdigest()
    script = prior.SCRIPT.replace('__OUT__', OUT).replace('__BASE__', prior.BASE)
    script = script.replace('__NAME__', NAME).replace('__SHA__', sha)
    original_argv = ("argv=[env['VENV_PYTHON'],'-u',str(source),'--input-dir',str(out),\n"
                     " '--minibatch-path',str(base/'native-optimizer-minibatch.pkl'),\n"
                     " '--output',str(out/'native-adam-analysis.json')]")
    assert original_argv in script
    script = script.replace(original_argv,
                            "argv=[env['VENV_PYTHON'],'-u',str(source),'--root',str(out)]")
    script = script.replace('native-adam-analysis.json', 'label-gradient-analysis.json')
    script = script.replace("env['CUDA_VISIBLE_DEVICES']=''",
                            "env['CUDA_VISIBLE_DEVICES']=''\nenv['OMP_NUM_THREADS']='1'")
    ast.parse(script.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0])
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / 'analysis-run.sh').write_text(script, encoding='utf-8')
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
    (LOCAL / 'analysis-cpu-run.stdout.txt').write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    for name in ('analysis-cpu-run.json', 'analysis.stdout.txt'):
        subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/{name}', str(LOCAL / name)], check=True)
    result.check_returncode()
    subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/label-gradient-analysis.json', str(LOCAL)], check=True)
    receipt = json.loads((LOCAL / 'analysis-cpu-run.json').read_bytes())
    assert hashlib.sha256((LOCAL / 'label-gradient-analysis.json').read_bytes()).hexdigest() == receipt['output']['sha256']
