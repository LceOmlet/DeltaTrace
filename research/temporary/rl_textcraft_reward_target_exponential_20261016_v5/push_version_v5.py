"""Push the authorized current branch and verify its exact remote commit."""
import json
from pathlib import Path
import re
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
label=sys.argv[1];assert label in ('source','deployment')
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
result=subprocess.run(['git','push','origin','HEAD'],cwd=REPO,capture_output=True,text=True)
record=dict(unix=time.time(),branch=branch,commit=head,push_returncode=result.returncode)
if result.returncode:
    record['error']=re.sub(r'(https?://)[^/@\s]+@',r'\1<redacted>@',result.stderr+result.stdout)
else:
    remote=subprocess.run(['git','ls-remote','origin','refs/heads/'+branch],cwd=REPO,capture_output=True,text=True)
    record['remote_commit']=remote.stdout.split()[0] if remote.returncode==0 and remote.stdout.split() else None
    record['remote_commit_verified']=record['remote_commit']==head
path=HERE/'receipts'/('git-push-'+label+'.json')
path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
raise SystemExit(result.returncode or (0 if record.get('remote_commit_verified') else 1))
