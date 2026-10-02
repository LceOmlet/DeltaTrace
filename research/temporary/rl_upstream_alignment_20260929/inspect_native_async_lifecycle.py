"""Read installed owners and current ranks; do not attach or create engines."""
from stage_environment_entry import remote,ROOT,ENTRY

remote(fr'''source {ENTRY}/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import ast,hashlib,json,pathlib,psutil,time
site=pathlib.Path('/opt/conda/lib/python3.12/site-packages')
root=pathlib.Path('{ROOT}');base=root/'candidates/appworld-native-async-015-transport-20261002'
output=base/'probe-lifecycle';output.mkdir(exist_ok=True)
active=json.loads((root/'active-training.json').read_text())
job=next(j for j in active['jobs'] if j['task']=='AppWorld')
driver=psutil.Process(job['pid'])
assert abs(driver.create_time()-job['observed_process_created_unix'])<.02
report=dict(observed_unix=time.time(),formal_pid=driver.pid,sources=[],workers=[])
names=['vllm/v1/worker/gpu_worker.py','vllm/v1/worker/worker_base.py',
       'vllm/distributed/parallel_state.py','vllm/v1/executor/abstract.py',
       'vllm/v1/executor/external_launcher.py','vllm/config/parallel.py']
for p in (site/'vllm_metax').rglob('*.py'):
 if 'worker' in str(p) or p.name=='platform.py':names.append(str(p.relative_to(site)))
for name in names:
 p=site/name
 if not p.exists():continue
 s=p.read_text();text=[]
 try:
  for node in ast.walk(ast.parse(s)):
   if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in [
     'init_device','init_worker','init_distributed_environment','init_worker_distributed_environment',
     '_init_executor','sleep','wake_up','add_lora','update_params','_get_worker_wrapper']:
    text.append(dict(name=node.name,source=ast.get_source_segment(s,node)))
 except SyntaxError:pass
 if text:
  report['sources'].append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),methods=text))
  print(name,[(t['name'],len(t['source'])) for t in text])
for child in driver.children(recursive=True):
 try:
  if child.pid not in [1213516,1217549]:continue
  env=child.environ();r=dict(pid=child.pid,name=child.name(),environment={{k:env.get(k) for k in [
    'RANK','WORLD_SIZE','LOCAL_RANK','CUDA_VISIBLE_DEVICES','MACA_VISIBLE_DEVICES','MASTER_ADDR','MASTER_PORT']}})
  report['workers'].append(r);print(json.dumps(r))
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
(output/'installed-lifecycle-sources.json').write_text(json.dumps(report,indent=2)+'\n')
PY
''')
