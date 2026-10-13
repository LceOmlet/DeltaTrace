"""Use the actual new candidate launch environment for its CPU-only RPC client."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('strict_entry',HERE.parent/'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
name=sys.argv[1];local=HERE/name;assert local.parent==HERE and local.is_file()
out=entry.ROOT+'/receipts/textcraft-gradient-rootcause-20261013-v1'
subprocess.run(entry.SCP+[str(local),entry.SSH[-1]+':'+out+'/'+name],check=True)
script=rf'''source {shlex.quote(entry.ENTRY+'/metax-entry.env.sh')}
"$VENV_PYTHON" - {shlex.quote(out+'/'+name)} <<'PY'
import json,os,subprocess,sys
from pathlib import Path
run=Path({entry.ROOT!r})/'runs/textcraft-dt-global-token-mean-temporal-20261014-v1'
env=dict(os.environ);env.update(json.loads((run/'launch.json').read_bytes())['effective_environment']);env['CUDA_VISIBLE_DEVICES']=''
r=subprocess.run([env['VENV_PYTHON'],'-u',sys.argv[1]],env=env)
raise SystemExit(r.returncode)
PY
'''
r=subprocess.run(entry.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=650)
(HERE/(local.stem+'.stdout.txt')).write_bytes(r.stdout);(HERE/(local.stem+'.stderr.txt')).write_bytes(r.stderr)
if r.returncode:print(r.stderr.decode(errors='replace'));raise SystemExit(r.returncode)
d=json.loads(r.stdout);(HERE/(local.stem+'.json')).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
print('receipt',d.get('receipt'))
