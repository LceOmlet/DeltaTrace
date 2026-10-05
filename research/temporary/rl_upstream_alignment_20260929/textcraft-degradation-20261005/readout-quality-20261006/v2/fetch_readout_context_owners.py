"""Read-only fetch of exact native owner bytes and an already saved request map."""
import hashlib
import importlib.util
import json
import pathlib
import shlex
import subprocess
import time

HERE = pathlib.Path(__file__).resolve().parent
AUDIT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location('existing_stage_environment_entry', AUDIT / 'stage_environment_entry.py')
STAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STAGE)
DEG = HERE.parents[1]
CONFIG = DEG / 'native-minibatch-v4' / 'native-minibatch-effective-config.json'
MAPPING = DEG / 'native-minibatch-v4' / 'native-minibatch-readout-mapping.json'
cfg = json.loads(CONFIG.read_text())
mapping = json.loads(MAPPING.read_text())
owner = cfg['env']['textcraft']['owner_root'] + '/AgentGym-RL'
files = {
    'actual-agentgym-schemas.py': owner + '/verl/workers/rollout/schemas.py',
    'actual-agentgym-vllm-rollout.py': owner + '/verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py',
    'actual-agentgym-rl-dataset.py': owner + '/verl/utils/agent_dataset/rl_dataset.py',
    'old-stopped-agentgym-schema-before-artifacts.py': STAGE.ENTRY + '/textcraft-schema-before-artifacts.py',
    'original-readout-mapped-requests.json': mapping['mapped_requests_artifact']['path'],
}
program = 'import pathlib,hashlib,json,os,time\nfiles=' + repr(files) + '''
out = {'scope':'exact named source / already saved artifact reads only', 'pid':os.getpid(), 'created_unix':time.time(), 'files':{}}
for name,path in files.items():
 p=pathlib.Path(path)
 if not p.is_file():
  out['files'][name]={'path':path,'exists':False}
  continue
 b=p.read_bytes()
 out['files'][name]={'path':path,'exists':True,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'mtime_ns':p.stat().st_mtime_ns}
print(json.dumps(out,sort_keys=True))
'''
started = time.time()
run = subprocess.run(STAGE.SSH + ['/opt/conda/bin/python -c ' + shlex.quote(program)], capture_output=True, text=True, check=True, timeout=60)
(HERE / 'readout-context-owner-fetch.stdout.txt').write_text(run.stdout + run.stderr, encoding='utf-8')
receipt = json.loads(run.stdout)
for name, record in receipt['files'].items():
    if not record['exists']:
        continue
    local = HERE / name
    subprocess.run(STAGE.SCP + [STAGE.SSH[-1] + ':' + record['path'], str(local)], check=True, capture_output=True, timeout=60)
    data = local.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    assert actual == record['sha256']
    record['local_path'] = str(local)
    record['local_sha256'] = actual
    if name == 'original-readout-mapped-requests.json':
        assert actual == mapping['mapped_requests_artifact']['sha256']
receipt['input_sources'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (CONFIG, MAPPING, AUDIT / 'stage_environment_entry.py')}
receipt['operations'] = {'new_model':0,'model_forward':0,'backward':0,'optimizer_update':0,'sampling':0,'source_edits':0}
receipt['wall_seconds'] = time.time() - started
(HERE / 'readout-context-owner-fetch.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:{'exists':v['exists'],'sha256':v.get('sha256'),'bytes':v.get('bytes')} for k,v in receipt['files'].items()},sort_keys=True))
