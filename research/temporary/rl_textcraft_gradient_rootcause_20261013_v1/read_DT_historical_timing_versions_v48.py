"""Read old/new DT timing evidence and recorded source identity; no model calls."""
import hashlib
import json
from pathlib import Path
import re
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
RUNS = {
    'old': ROOT/'runs/textcraft-dt-formal-native-graph-fresh-20261012-v1',
    'new': ROOT/'runs/textcraft-dt-grpo-multiplier-no-whiten-20261014-v1',
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

report = dict(unix=time.time(),runs={},model_calls=0,DT_calls=0,training_mutations=0)
for label, run in RUNS.items():
    launch = json.loads((run/'launch.json').read_bytes())
    env = launch['effective_environment']
    wanted = [k for k in env if k.startswith('DT_') or k in
              ('VERL_ROOT','VENV_PYTHON','PYTHONPATH','MACA_PATH','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES')]
    item = dict(run=str(run),launch_sha256=sha(run/'launch.json'),
                pid=launch['pid'],birth=launch['birth'],
                environment={k:env[k] for k in wanted},files={},evidence=[],worker_logs={})
    dtroot = Path(env['DT_ROOT'])
    paths = [Path(env['DT_ENVIRONMENT_JSON'])]
    paths += [dtroot/p for p in (
        'clean/qwen35/qwen35_gdn_finite.py',
        'clean/qwen35/qwen35_dense_finite_runner.py',
        'clean/qwen35/qwen35_decoder_finite.py',
        'accelerated/qwen35/qwen35_retained_capture.py',
        'accelerated/qwen35/qwen35_code_local_capture.py')]
    for directory in env['PYTHONPATH'].split(':'):
        paths += [Path(directory)/n for n in ('reward_readout.py','deltatrace_rollout.py')]
    for p in dict.fromkeys(paths):
        if p.is_file():
            item['files'][str(p)] = dict(SHA256=sha(p))
            if p.suffix=='.json': item['files'][str(p)]['value']=json.loads(p.read_bytes())
    for session in Path('/tmp/ray').glob('session_*_'+str(launch['pid'])):
        item['ray_session']=str(session)
        for p in sorted((session/'logs').glob('worker-*')):
            if p.suffix not in ('.out','.err'): continue
            evidence=[]
            first_batches=[]; round_index=0
            for lineno,line in enumerate(p.read_text(errors='replace').splitlines(),1):
                if '[DT direct joint minibatch]' in line:
                    m=re.search(r'trajectories=(\d+) length=(\d+) batch=(\d+)/(\d+) seconds=([\d.eE+-]+)',line)
                    if m:
                        b,n=int(m[3]),int(m[4])
                        if b==1: round_index+=1
                        if round_index<=2:
                            first_batches.append(dict(line=lineno,round=round_index,trajectories=int(m[1]),
                                length=int(m[2]),batch=b,planned=n,seconds=float(m[5])))
                if any(marker in line for marker in (
                    '[DT direct target workload]','step:','DT numerical','DT runtime','retained capture',
                    'capture_host','host_gate','Nonfinite DT','OutOfMemoryError')):
                    if len(evidence)<18 or 'step:1 ' in line or 'step:2 ' in line:
                        evidence.append(dict(line=lineno,text=line[-4500:]))
            if first_batches or evidence:
                item['worker_logs'][str(p)]=dict(first_two_round_batches=first_batches,evidence=evidence[:22])
    for directory in ('runtime-overrides','worker-runtime-overrides'):
        for p in sorted((run/directory).glob('*.json')):
            item['evidence'].append(dict(path=str(p),SHA256=sha(p),value=json.loads(p.read_bytes())))
    report['runs'][label]=item

dest=OUT/'DT-historical-timing-versions-v48.json'
dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
