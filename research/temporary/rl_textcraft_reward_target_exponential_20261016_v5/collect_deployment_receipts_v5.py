"""Copy exact completed deployment receipts through the recorded SSH/SCP owner."""
import json
from pathlib import Path
import subprocess

from transport import AUDIT, HERE, entry

script = r'''
import json
from pathlib import Path
audit=Path(AUDIT)
job=json.loads((audit/'DTPO-v5-job.json').read_bytes())
run=Path(job['run'])
copies={
    'initial-state-and-existing-passive-records-v5.json':run/'runtime-overrides/initial-state-and-existing-passive-records-v5.json',
    'deployed-launch.json':run/'launch.json',
    'deployed-source.json':run/'source.json',
}
for name,path in copies.items():
    (audit/name).write_bytes(path.read_bytes())
observation=json.loads((audit/'deployment-actual-observation.json').read_bytes())
context=[]
for error in observation.get('original_log_errors') or []:
    path=Path(error['path']);lines=path.read_text(errors='replace').splitlines()
    index=error['line']-1
    context.append(dict(path=str(path),first_line=max(1,index-3),
        lines=lines[max(0,index-4):index+32]))
(audit/'deployment-existing-log-context.json').write_text(json.dumps(context,indent=2)+'\n')
print(json.dumps(context,indent=2))
'''.replace('Path(AUDIT)', 'Path('+repr(AUDIT)+')')
entry.remote('source '+entry.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES="" "$VENV_PYTHON" - <<\'PY\'\n'+script+'\nPY\n')
names=['DTPO-v5-job.json','deployment-actual-observation.json',
       'initial-state-and-existing-passive-records-v5.json',
       'deployed-launch.json','deployed-source.json','deployment-existing-log-context.json']
for name in names:
    subprocess.run(entry.SCP+[entry.SSH[-1]+':'+AUDIT+'/'+name,
        str(HERE/'receipts'/name)],check=True)
print(json.dumps(dict(downloaded=names),indent=2))
