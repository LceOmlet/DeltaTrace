"""Stage one named CPU-only diagnostic using the strict recorded SSH entry."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('strict_entry',HERE.parent/'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
name=sys.argv[1]
local=HERE/name
assert local.parent==HERE and local.is_file()
remote=entry.ROOT+'/receipts/textcraft-gradient-rootcause-20261013-v1'
subprocess.run(entry.SSH+['mkdir','-p',remote],check=True)
subprocess.run(entry.SCP+[str(local),entry.SSH[-1]+':'+remote+'/'+name],check=True)
script=f'''source {shlex.quote(entry.ENTRY+'/metax-entry.env.sh')}
"$VENV_PYTHON" - {shlex.quote(remote+'/'+name)} <<'PY'
import json,os,subprocess,sys
from pathlib import Path
root=Path({entry.ROOT!r})
env=dict(os.environ);env.update(json.loads((root/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1/launch.json').read_bytes())['effective_environment']);env['CUDA_VISIBLE_DEVICES']=''
env['ROOTCAUSE_METHOD']={sys.argv[2] if len(sys.argv)>2 else 'dt'!r}
r=subprocess.run([env['VENV_PYTHON'],'-u',sys.argv[1],*{sys.argv[3:]!r}],env=env)
raise SystemExit(r.returncode)
PY
'''
r=subprocess.run(entry.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=650)
suffix='-'+sys.argv[2] if len(sys.argv)>2 else ''
(HERE/(local.stem+suffix+'.stdout.txt')).write_bytes(r.stdout)
(HERE/(local.stem+suffix+'.stderr.txt')).write_bytes(r.stderr)
if r.returncode:
 print(r.stderr.decode(errors='replace'));raise SystemExit(r.returncode)
data=json.loads(r.stdout)
(HERE/(local.stem+suffix+'.json')).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
print('receipt',data.get('receipt'))
windows=data.get('report',{}).get('windows',{})
if isinstance(windows,dict):
 for n,w in windows.items():
  if all(k in w for k in ('unique_trajectories','policy_tokens','target_tokens','groups')):
   print(n,w['unique_trajectories'],w['policy_tokens'],w['target_tokens'],w['groups'])
