"""CPU regression of the queued-request seam; leaves live training untouched."""
from stage_environment_entry import remote, ROOT, ENTRY, AUDIT, SCP, SSH
import subprocess

remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
CUDA_VISIBLE_DEVICES='' MACA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$VENV_PYTHON" - <<'PY'
from pathlib import Path
import os,json,subprocess,hashlib,time
r=Path('@ROOT@')
j=next(j for j in json.loads((r/'active-training.json').read_text())['jobs'] if j['task']=='AppWorld')
out=r/'receipts/owner-b8-dispatch-20260930/appworld-request-dispatch'
candidate=out/'candidate'
source=json.loads((Path(j['output'])/'source.json').read_text())
prepared=json.loads(Path(source['prepared_receipt']).read_text())
env=os.environ.copy()
env['PYTHONPATH']=':'.join([str(candidate),j['entry'],j['verl_root'],prepared['loop_root'],env['PYTHONPATH']])
env['LOOP_ROLLOUT_BASELINE']=str(Path(j['entry'])/'loop-original-rollout-worker.py')
bootstrap="import os,site,sys;site.addsitedir(os.environ['LOOP_EXTRAS']);import loop_owner_rollout;print('actual_transport='+loop_owner_rollout.__file__,flush=True);assert loop_owner_rollout.__file__==sys.argv[1];import pytest;raise SystemExit(pytest.main(sys.argv[2:]))"
args=[env['VENV_PYTHON'],'-c',bootstrap,str(candidate/'loop_owner_rollout.py'),
      '-q','--import-mode=importlib',str(out/'test_loop_transport_dispatch.py'),
      str(Path(j['entry'])/'test_loop_owner_worker.py')+'::test_transport_receives_already_queued_item_before_feeder_finishes',
      str(Path(j['entry'])/'test_loop_owner_worker.py')+'::test_original_scheduler_collection_and_cleanup_are_unchanged',
      '--junitxml='+str(out/'after-import-bound.xml')]
result=subprocess.run(args,env=env,cwd=out,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(out/'after-import-bound.log').write_text(result.stdout)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record=json.loads((out/'candidate.json').read_text())
record.update(verified_unix=time.time(),test_returncode=result.returncode,
              tested_import=str(candidate/'loop_owner_rollout.py'),
              test_command=args,
              prior_harness_failure='Default pytest path insertion loaded the live old module; LOOP extras were not loaded. Kept after.xml/log; corrected only harness import paths and reused installed extras.')
record['tests']['after_import_bound']=sha(out/'after-import-bound.xml')
(out/'verified-candidate.json').write_text(json.dumps(record,indent=2)+'\n')
print(result.stdout[-2500:]);print('returncode',result.returncode)
if result.returncode:raise SystemExit(result.returncode)
PY
'''.replace('@ENTRY@', ENTRY).replace('@ROOT@', ROOT))

local = AUDIT/'appworld-request-dispatch-20261001'
local.mkdir(exist_ok=True)
for name in ['before.xml', 'after.xml', 'after-import-bound.xml',
             'after-import-bound.log', 'verified-candidate.json']:
    subprocess.run(SCP+[f'{SSH[-1]}:{ROOT}/receipts/owner-b8-dispatch-20260930/appworld-request-dispatch/{name}',
                        str(local/name)],check=True)
