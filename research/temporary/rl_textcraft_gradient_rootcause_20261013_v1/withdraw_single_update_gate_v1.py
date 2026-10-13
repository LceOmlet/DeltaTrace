"""Withdraw the rejected gate in prepared-only metadata; never launch training."""
import hashlib
import json
from pathlib import Path
import time
OUT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-gradient-rootcause-20261013-v1')
prepared=OUT/'formal-DT-global-token-mean-fresh-v1-prepared'
path=prepared/'launch.json'
before=path.read_bytes();data=json.loads(before)
assert data['status']=='prepared_only_not_submitted_or_accepted'
saved=prepared/'launch.before-user-scope-correction-v1.json'
if not saved.exists():saved.write_bytes(before)
old=data.pop('behavior_verification_pending',None)
correction={'unix':time.time(),'single_update_validation_gate':'withdrawn_by_user_correction',
 'raw_measurement':old or str(OUT/'global-token-mean-v3-behavior-retry-v4/result.json'),
 'scope':'A single PPO update need not improve validation and cannot diagnose accelerating gradient norm.',
 'rootcause_resolved':False,'formal_submission':False}
data['user_scope_correction']=correction
path.write_text(json.dumps(data,indent=2)+'\n')
report={'scope':__doc__,'before_SHA256':hashlib.sha256(before).hexdigest(),
 'after_SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),'prepared_launch':str(path),**correction}
dest=OUT/'withdrawn-single-update-gate-v1.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(dest),'report':report}))
