"""Read the specific finite native PG job; never restart it on observation timeouts."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('strict', HERE.parent / 'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
code = """import hashlib,json,pathlib,psutil,time
d=pathlib.Path(%r)/'receipts/textcraft-gradient-rootcause-20261013-v1/native-saved-update-natural-PG-v99'
j=json.loads((d/'job.json').read_bytes())
try:
 p=psutil.Process(j['pid']);alive=p.create_time()==j['birth'] and p.status()!=psutil.STATUS_ZOMBIE
except psutil.NoSuchProcess:alive=False
def read_json(path):
 try:return json.loads(path.read_bytes())
 except (FileNotFoundError,json.JSONDecodeError):return None
state=read_json(d/'result.json')
forward_bindings=[]
if (d/'stdout.log').exists():
 for line in (d/'stdout.log').read_text(errors='replace').splitlines():
  if not line.startswith('{'):continue
  try:value=json.loads(line)
  except json.JSONDecodeError:continue
  if value.get('phase')=='native_old_policy_point':forward_bindings.append(value)
chains=[]
for path in sorted(d.glob('iteration*-rank*-chain.json')):
 value=read_json(path)
 if value is not None:chains.append(dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),value=value))
native_PG_reports=[]
for path in sorted(d.glob('iteration*-rank*-PG.json')):
 value=read_json(path)
 if value is not None:
  keep={key:value[key] for key in ('rank','sources','effective_config','passes','optimizer_step_executed','native_gradient_clipping_executed')}
  native_PG_reports.append(dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),value=keep))
workers=[] if not alive else [dict(pid=x.pid,birth=x.create_time(),phase=x.name()) for x in p.children(recursive=True) if x.name().startswith('ray::ObservedWorker')]
tails={}
for name in ('stdout.log','stderr.log'):
 path=d/name
 if path.exists():
  with path.open('rb') as f:f.seek(max(0,path.stat().st_size-16384));tails[name]=f.read().decode(errors='replace').splitlines()[-10:]
print(json.dumps(dict(unix=time.time(),job=j,alive=alive,workers=workers,state=state,chain_reports=chains,native_PG_reports=native_PG_reports,forward_bindings=forward_bindings,tails=tails)))
""" % entry.ROOT
r = subprocess.run(entry.SSH + ['/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python', '-'], input=code.encode(), capture_output=True, check=True)
report = json.loads(r.stdout)
path = HERE / 'native-saved-update-natural-PG-v99-status-latest.json'
path.write_text(json.dumps(dict(receipt=entry.ROOT + '/receipts/textcraft-gradient-rootcause-20261013-v1/native-saved-update-natural-PG-v99', report=report), indent=2) + '\n', encoding='utf-8')
state = report['state'] or {}
if state.get('phase') == 'failed' or state.get('phase', '').startswith('complete_'):
    (HERE / 'native-saved-update-natural-PG-v99-status-final.json').write_bytes(path.read_bytes())
display_tails = {key: [line if len(line) < 500 else line[:300] + '…' for line in value[-4:]] for key, value in report['tails'].items()}
print(json.dumps(dict(unix=report['unix'], pid=report['job']['pid'], birth=report['job']['birth'], alive=report['alive'], phase=state.get('phase'), case=state.get('iteration'), workers=report['workers'], completed_cases=[dict(case=x['iteration'], norms=x['gradient_statistics']['norms']) for x in state.get('observations', [])], chains=len(report['chain_reports']), tails=display_tails), ensure_ascii=False))
