set -e
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 MACA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PYR'
from pathlib import Path
import json,hashlib,os,sys,subprocess,time,psutil
r=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=json.loads((r/'active-training.json').read_bytes());j=next(j for j in a['jobs'] if j['task']=='AppWorld')
assert j['pid']==1119928 and not psutil.pid_exists(j['pid'])
old=r/'receipts/appworld-efficiency-20261007/native-conv-production-prepared-v1/prepared.json'
assert sha(old)=='e9dacc24168d1849ec9b8da9b1024bed7e0b349ce308b70aba66ab02c52b1644'
p=json.loads(old.read_bytes());d=r/'receipts/appworld-efficiency-20261007/native-conv-canonical-owner-v2';assert not d.exists();d.mkdir()
iso=p['isolated_owners']['details'];target=Path(next(x for x in iso['originals'] if x.endswith('modeling_qwen3_5.py')))
assert sha(target)=='f7e1a804fa12684bd1cc225c85cdf5f0b5996f30d66263f11de13449f53272be'
candidate=Path(iso['hf_candidate']['path']);assert sha(candidate)=='59f9c339e3c01672b67ba09b0e56e2e960ba28c6c3d9975faadf00d15ee0b4c8'
backup=d/'modeling_qwen3_5.before.py';backup.write_bytes(target.read_bytes())
# This is the already operator-tested minimal owner patch, not a replacement module/loader.
# Atomic change inside the recorded task VENV; no package installation or task owner edit.
tmp=target.with_name('modeling_qwen3_5.dt-staged.py');assert not tmp.exists();tmp.write_bytes(candidate.read_bytes());os.replace(tmp,target)
assert sha(target)==sha(candidate)
p['previous_preparation']={'path':str(old),'sha256':sha(old)}
p['base_source']={'path':j['source_receipt'],'sha256':sha(j['source_receipt'])}
p['pythonpath']=':'.join(q for q in p['pythonpath'].split(':') if q!=str(Path(iso['sitecustomize']['path']).parent))
for key in ['DT_PREFIX_NATIVE_CONV_INITIAL_STATES','DT_CONV_ISOLATED_IMPORT_ROOT']:p['resource_environment'].pop(key,None)
p['canonical_HF_owner']={'path':str(target),'sha256':sha(target),'before_backup':str(backup),'before_sha256':sha(backup),'patch_scope':'Exact previously tested minimal HF cached-convolution API patch; installed canonical module, no eager sitecustomize import in environment children','installed_unix':time.time()}
p['isolated_owners']['current_use']='DT linked owner tree only; old HF namespace and sitecustomize are diagnostic history and are no longer on PYTHONPATH'
p['preparation_script']={'path':str(r/'receipts/appworld-efficiency-20261007/prepare-native-conv-canonical-owner-v2.sh'),'sha256':sha(r/'receipts/appworld-efficiency-20261007/prepare-native-conv-canonical-owner-v2.sh')}
cpu=os.environ.copy();cpu.update(p['resource_environment'],PYTHONPATH=p['pythonpath'],CUDA_VISIBLE_DEVICES='-1',MACA_VISIBLE_DEVICES='-1')
for key in ['DT_PREFIX_NATIVE_CONV_INITIAL_STATES','DT_CONV_ISOLATED_IMPORT_ROOT']:cpu.pop(key,None)
code="import inspect,json,hashlib,torch;import deltatrace_rollout;import qwen35_dense_finite_runner as runner;import qwen35_gdn_finite as finite;from transformers.models.qwen3_5 import modeling_qwen3_5 as model;from pathlib import Path;print(json.dumps({'sources':{k:{'path':str(Path(inspect.getsourcefile(m))),'sha256':hashlib.sha256(Path(inspect.getsourcefile(m)).read_bytes()).hexdigest()} for k,m in [('producer',deltatrace_rollout),('runner',runner),('finite',finite),('model',model)]},'cuda_initialized':torch.cuda.is_initialized()}))"
c=subprocess.run([sys.executable,'-c',code],env=cpu,capture_output=True,text=True);(d/'cpu-import.stdout.txt').write_text(c.stdout);(d/'cpu-import.stderr.txt').write_text(c.stderr);c.check_returncode()
v=json.loads(c.stdout.splitlines()[-1]);assert v['sources']['model']=={'path':str(target),'sha256':sha(target)} and not v['cuda_initialized']
for name in ['producer','runner','finite']:assert v['sources'][name]==p['cpu_actual_imports']['details']['sources'][name]
(d/'cpu-import-identity.json').write_text(json.dumps(v,indent=2)+'\n');p['cpu_actual_imports']={'path':str(d/'cpu-import-identity.json'),'sha256':sha(d/'cpu-import-identity.json'),'details':v}
p['status']='prepared_correct_canonical_owner_no_eager_service_import_not_submitted';p['prepared_unix']=time.time()
p['production_import_scope']='Original canonical installed HF owner with minimal tested patch; no eager namespace loader in any task service; original LOOP service parameters unchanged'
(d/'prepared.json').write_text(json.dumps(p,indent=2)+'\n');print(json.dumps({'prepared':str(d/'prepared.json'),'sha256':sha(d/'prepared.json'),'HF_owner':p['canonical_HF_owner'],'pythonpath':p['pythonpath']}))
PYR
