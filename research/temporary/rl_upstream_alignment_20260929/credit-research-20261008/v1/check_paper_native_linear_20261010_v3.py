"""Call the existing captured-Linear checker unchanged, with its owner defaults."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ROOT,ENTRY,SSH,SCP

OUT = HERE/'paper-native-rows-20261010-v3'
REMOTE = ROOT+'/receipts/paper-native-rows-20261010-v3'
script = OUT/'check_native_linear_roundoff.py'
assert hashlib.sha256(script.read_bytes()).hexdigest() == '21bdd3b3205b6b7a6d789d80edffcea00d614cd5f00ba1ae9f2d2f8276f89ce5'
result = json.loads((OUT/'result.json').read_bytes())
assert result['phase']=='complete' and result['native_forward_calls']==1
for row,first in result['operator_observation']['first_unequal_by_row'].items():
    event=result['operator_observation']['events'][first['event']]
    assert event['operation'].endswith('.base')
    assert event['inputs']['input']['rows'][int(row)]['equal']
subprocess.run(SCP+[str(script),SSH[-1]+':'+REMOTE+'/'],check=True,timeout=30)
body = r'''
import hashlib,json,subprocess,time
from pathlib import Path
out=Path(REMOTE)
assert (out/'completed.json').exists()
script=out/'check_native_linear_roundoff.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()=='21bdd3b3205b6b7a6d789d80edffcea00d614cd5f00ba1ae9f2d2f8276f89ce5'
argv=[PYTHON,str(script),'--folder',str(out),'--output',str(out/'linear-roundoff.json')]
start=time.time()
run=subprocess.run(argv,capture_output=True,text=True,check=True,timeout=55)
receipt=dict(unix=start,completed_unix=time.time(),argv=argv,stdout=run.stdout,stderr=run.stderr,
 script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),GPU_calls=0,thresholds_changed=False)
(out/'linear-check-invocation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''
body = 'REMOTE='+repr(REMOTE)+'\nPYTHON='+repr('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python')+'\n'+body
shell = 'source '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+body+'\nPY\n'
(OUT/'linear-check-command.sh').write_text(shell,encoding='utf-8',newline='\n')
run = subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,timeout=65)
(OUT/'linear-check.stderr').write_bytes(run.stderr)
run.check_returncode()
d = json.loads(run.stdout)
(OUT/'linear-check-invocation.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
print(json.dumps(d))
