set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'

import os,json,hashlib,subprocess,time
from pathlib import Path
root=Path(ROOT);out=root/'receipts/credit-collection-error-gradients-20261008-v1';check=out/'default-path-check';check.mkdir(exist_ok=True)
source=json.loads((root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json').read_bytes())
base=root/'receipts/credit-collection-gradients-20261008-v1'
(check/'gradient-inputs.json').write_bytes((base/'gradient-inputs.json').read_bytes())
env=dict(os.environ,**source['environment']);env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None);env['CUDA_VISIBLE_DEVICES']='-1';env['DT_COLLECTION_DIAGNOSTIC_OUT']=str(check)
dt=Path(env['DT_ROOT']);q=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
env['PYTHONPATH']=':'.join([str(out),str(dt),env.get('DT_OFFICIAL_ROOT') or q['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],q['ft_extension_root']])
run=subprocess.run([env['VENV_PYTHON'],str(out/'inspect_collection_gradients.py'),'--inspect-only'],cwd=check,env=env,capture_output=True,timeout=90)
(check/'stdout.txt').write_bytes(run.stdout);(check/'stderr.txt').write_bytes(run.stderr)
if run.returncode:print(run.stderr.decode(errors='replace'));run.check_returncode()
old=json.loads((base/'input-inspection.json').read_bytes());default=json.loads((check/'input-inspection.json').read_bytes());errors=json.loads((out/'input-inspection.json').read_bytes())
assert old['mapping']==default['mapping']==errors['mapping']
assert default['labels']==old['labels'] and default['rows']==errors['rows']==256
assert not default['cuda_initialized'] and not errors['cuda_initialized']
assert errors['coefficient_errors']['counts']=={'predicted_tail_error':37,'uniform_bounded_error':126,'uniform_missed_native_tail_error':2}
default_prepared=json.loads(run.stdout);error_prepared=json.loads((out/'prepare.stdout.txt').read_bytes())
assert default_prepared['base_metadata_preserved'] and error_prepared['base_metadata_preserved']
assert default_prepared['labels_by_minibatch']==[default['labels']]*4
assert error_prepared['labels_by_minibatch']==[errors['labels'][:3]]*3+[errors['labels']]
assert sum(map(len,error_prepared['labels_by_minibatch']))==13
print(json.dumps({'status':'Both real-artifact DataProto preparation paths passed; no model/backward','unix':time.time(),'driver_sha256':hashlib.sha256((out/'inspect_collection_gradients.py').read_bytes()).hexdigest(),'default_original_mapping_exact':True,'default_labels':default['labels'],'error_labels':errors['labels'],'error_counts':errors['coefficient_errors']['counts'],'cuda_initialized':False,'default_labels_by_minibatch':default_prepared['labels_by_minibatch'],'error_labels_by_minibatch':error_prepared['labels_by_minibatch'],'base_metadata_preserved':True,'default_check_directory':str(check)}))

PY
