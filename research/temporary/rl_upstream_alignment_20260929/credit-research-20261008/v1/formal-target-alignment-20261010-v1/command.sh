source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
from pathlib import Path
import json,os,subprocess,hashlib
r=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');m=json.loads((r/'runs/textcraft-formal-stable-20261009-v1/source.json').read_bytes())
out=r/'receipts/formal-target-alignment-20261010-v1'
script=out/'audit_formal_target_alignment_20261010.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()=='37bd8363d9b0c9872bf804a4ce94952b5cda279dd9c6a35a40e5a26a8a2dc270'
assert not (out/'result.json').exists()
env=dict(os.environ);env.update(m['environment'])
env.update(CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
a=subprocess.run([env['VENV_PYTHON'],str(script),str(out/'result.json')],env=env,capture_output=True)
(out/'stdout.txt').write_bytes(a.stdout);(out/'stderr.txt').write_bytes(a.stderr)
print(a.stdout.decode(errors='replace'));print(a.stderr.decode(errors='replace'))
a.check_returncode()
PY
