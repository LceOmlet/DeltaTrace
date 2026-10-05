"""Read stopped TextCraft owner records without running or modifying training."""
import hashlib
import json
from pathlib import Path
import subprocess

from stage_environment_entry import ROOT, SSH, SCP, AUDIT

REMOTE = ROOT + '/receipts/textcraft-degradation-20261005'
LOCAL = AUDIT / 'textcraft-degradation-20261005'

SCRIPT = r'''/opt/conda/bin/python - <<'PY'
import hashlib,json,pathlib,re,shutil,time
root=pathlib.Path('__ROOT__')
run=root/'runs/textcraft-rollout-scope-20261002/textcraft-dt'
out=root/'receipts/textcraft-degradation-20261005';out.mkdir(exist_ok=True)
log=run/'train.log'; raw=log.read_bytes();lines=raw.decode(errors='replace').splitlines()
wanted={32,35,36,39,40,43,55,83,85,93,109}
metric_rows=[];selected=[];pending=[];warnings=[]
for number,line in enumerate(lines,1):
 if 'grad_norm is not finite' in line:
  warnings.append(dict(line=number,text=line))
 marker='[DT EOS minimum] '
 if marker in line:
  try: value=json.JSONDecoder().raw_decode(line.split(marker,1)[1])[0]
  except (ValueError,TypeError): continue
  pending.append(dict(line=number,data=value))
 match=re.search(r'\bstep:(\d+)\s+-\s+global_seqlen/',line)
 if match:
  step=int(match.group(1))
  metric_rows.append(dict(step=step,line=number,text=line))
  if step in wanted: selected.append(dict(step=step,minimum_records=pending))
  pending=[]
for step in [35,39,40,43,55,93]:
 source=run/f'rollouts/{step}.jsonl'
 if source.exists():shutil.copyfile(source,out/f'rollout-{step}.jsonl')
source_manifest={}
for name in ['source.json','launch.json','job.json']:
 source=run/name;shutil.copyfile(source,out/name)
 source_manifest[name]=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())
receipt=dict(unix=time.time(),source_log=dict(path=str(log),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)),
 source_manifest=source_manifest,metrics=metric_rows,nonfinite_gradient_warnings=warnings,
 dt_minimum_records_by_completed_iteration=selected,
 scope='Original stopped records only; no reconstructed token IDs or new model computation')
(out/'records.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(out=str(out),metrics=len(metric_rows),nonfinite_warnings=len(warnings),
 selected=[dict(step=x['step'],records=len(x['minimum_records']),
  sample_type=type(x['minimum_records'][0]['data']).__name__ if x['minimum_records'] else None,
  sample_keys=list(x['minimum_records'][0]['data'])[:30] if x['minimum_records'] and isinstance(x['minimum_records'][0]['data'],dict) else None) for x in selected],
 files=[dict(name=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.iterdir() if p.is_file()])) )
PY
'''

if __name__ == '__main__':
    LOCAL.mkdir(exist_ok=True)
    result=subprocess.run(SSH+['bash','-s'],input=SCRIPT.replace('__ROOT__',ROOT).encode(),capture_output=True,check=True)
    (LOCAL/'collection.stdout.json').write_bytes(result.stdout)
    print(result.stdout.decode(errors='replace'))
    subprocess.run(SCP+['-r',f'{SSH[-1]}:{REMOTE}/.',str(LOCAL)],check=True)
