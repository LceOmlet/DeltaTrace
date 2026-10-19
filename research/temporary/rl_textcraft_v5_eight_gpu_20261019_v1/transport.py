"""Strict existing MetaX transport; use the provisioned venv entry unchanged."""
import importlib.util
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('v5_eight_gpu_strict_entry',
    HERE.parent / 'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)

def run_python(code, *, timeout=55):
    script = ('source ' + shlex.quote(entry.ENTRY + '/metax-entry.env.sh') +
        '\n"$VENV_PYTHON" - <<\'PY\'\n' + code + '\nPY\n')
    result = subprocess.run(entry.SSH + ['bash', '-s'], input=script.encode('utf-8'),
        capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace')[-5500:])
    return result.stdout.decode('utf-8')

def upload(path, destination):
    subprocess.run(entry.SCP + [str(path), entry.SSH[-1] + ':' + destination], check=True)
