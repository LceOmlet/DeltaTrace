"""Bounded CPU reproduction using the actual imported VERL loss functions.

Synthetic scalar log probabilities test a suspected arithmetic mechanism only.
They cannot identify the cause of the unsaved iteration14 incident. No model,
DT, optimizer, worker RPC, production patch or replacement loss is used.
"""
import importlib.util
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
RAW = HERE / 'direct-credit-records-20261009-v1'
spec = importlib.util.spec_from_file_location('transport', HERE.parents[1] / 'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)

probe = r'''
import hashlib,inspect,json,os,resource,time
from pathlib import Path
import psutil
import torch
torch.set_num_threads(1)
from verl.trainer.ppo import core_algos as owner
source=Path(inspect.getsourcefile(owner))
started=time.perf_counter()
records=[]
for gap in [20.0,80.0,88.0,89.0,100.0]:
    for dtype in [torch.float32,torch.float64]:
        for branch in ['low_var_kl','dual_clip_negative','clip_positive']:
            if branch=='low_var_kl':
                logp=torch.tensor([[-1.0-gap]],dtype=dtype,requires_grad=True)
                ref=torch.tensor([[-1.0]],dtype=dtype)
                loss=owner.kl_penalty(logp,ref,'low_var_kl').sum()
            else:
                logp=torch.tensor([[-1.0]],dtype=dtype,requires_grad=True)
                old=torch.tensor([[-1.0-gap]],dtype=dtype)
                advantage=torch.full_like(old,-1.0 if branch=='dual_clip_negative' else 1.0)
                loss,*_=owner.compute_policy_loss(old,logp,advantage,torch.ones_like(old),cliprange=.2,clip_ratio_c=3.0)
            gradient,=torch.autograd.grad(loss,logp)
            records.append(dict(branch=branch,dtype=str(dtype),log_probability_gap=gap,
                loss=float(loss),gradient=float(gradient),
                loss_finite=bool(torch.isfinite(loss)),gradient_finite=bool(torch.isfinite(gradient))))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(unix=time.time(),seconds=time.perf_counter()-started,
    owner_path=str(source),owner_resolved=str(source.resolve()),
    owner_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    functions={n:dict(module=getattr(owner,n).__module__,qualname=getattr(owner,n).__qualname__,
        source=inspect.getsource(getattr(owner,n))) for n in ['kl_penalty','compute_policy_loss']},
    torch_version=torch.__version__,records=records,
    peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    PSS_bytes=psutil.Process().memory_full_info().pss,CUDA_initialized=False,
    CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
    model_calls=0,DT_calls=0,optimizer_calls=0,worker_RPC=0,production_changes=0,
    scope='Synthetic arithmetic mechanism reproduction, not an iteration14 replay or numerical tolerance test.'),allow_nan=True))
'''

outer = r'''
import hashlib,json,os,subprocess
from pathlib import Path
p=Path(SOURCE)
assert hashlib.sha256(p.read_bytes()).hexdigest()=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
source=json.loads(p.read_bytes())
env=dict(os.environ,**source['environment']);env.pop('RAY_ADDRESS',None)
env['CUDA_VISIBLE_DEVICES']='-1';env['PYTHONPATH']=source['pythonpath']
r=subprocess.run([env['VENV_PYTHON'],'-c',PROBE],env=env,capture_output=True,text=True,timeout=40)
print(r.stdout,end='')
if r.returncode:raise RuntimeError(r.stderr)
'''.replace('SOURCE', repr(transport.ROOT+'/runs/textcraft-formal-stable-20261009-v1/source.json'), 1).replace('PROBE', repr(probe), 1)

if __name__ == '__main__':
    stem = RAW / 'native-loss-exp-backward-20261010-v1'
    assert not stem.with_suffix('.json').exists(), 'Inspect the completed result instead of rerunning.'
    command = 'source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+outer+'\nPY\n'
    stem.with_suffix('.command.txt').write_text(command,encoding='utf-8')
    result = subprocess.run(transport.SSH+['bash','-s'],input=command.encode(),capture_output=True,timeout=50)
    stem.with_suffix('.stdout.txt').write_bytes(result.stdout)
    stem.with_suffix('.stderr.txt').write_bytes(result.stderr)
    result.check_returncode()
    data = json.loads(result.stdout)
    stem.with_suffix('.json').write_bytes(result.stdout)
    print(json.dumps(dict(saved=str(stem.with_suffix('.json')),
        owner_path=data['owner_path'],owner_sha256=data['owner_sha256'],seconds=data['seconds'],
        finite_loss_nonfinite_gradient_cases=[x for x in data['records'] if x['loss_finite'] and not x['gradient_finite']],
        peak_RSS_bytes=data['peak_RSS_bytes'],CUDA_initialized=data['CUDA_initialized']),ensure_ascii=False))
