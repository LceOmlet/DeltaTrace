"""Replay the saved iteration37 loss inputs through their actual VERL owner.

CPU only; no model, DT, optimizer, worker RPC, training patch or synthetic input.
Separates the native PPO and KL gradient paths at the captured log-prob tensor.
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
import hashlib,inspect,json,math,os,resource,time
from pathlib import Path
import psutil,torch
torch.set_num_threads(1)
from verl.trainer.ppo import core_algos as owner
source=Path(inspect.getsourcefile(owner))
assert hashlib.sha256(source.read_bytes()).hexdigest()=='fc2f992b16fb7fb23aebc683ad5f00136cf61fc9013cd426f5046983babe7299'
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-actor-incidents-20261010-v1')
paths=[root/'rank0-pid987808/loss-backward-1791646910870699040.pt',root/'rank1-pid989860/loss-backward-1791646910895854293.pt']
records=[];started=time.perf_counter()
for path in paths:
 data=torch.load(path,map_location='cpu',weights_only=False);t=data['tensors'];record=data['record']
 assert record['update_index']==20 and record['optimizer_index']==0 and record['microbatch_index']==1
 assert not record['missing_tensors'] and record['kl_penalty']=='low_var_kl'
 x=t['log_prob'].detach().requires_grad_(True);mask=t['response_mask']
 pg,*_=owner.compute_policy_loss(t['old_log_prob'],x,t['advantages'],mask,cliprange=.2,clip_ratio_c=3.0,loss_agg_mode='token-mean')
 pggrad,=torch.autograd.grad(pg,x)
 x=t['log_prob'].detach().requires_grad_(True)
 km=owner.kl_penalty(x,t['ref_logprob'],'low_var_kl')
 kl=owner.agg_loss(loss_mat=km,loss_mask=mask,loss_agg_mode='token-mean')
 kg,=torch.autograd.grad(kl,x)
 actual=t['log_prob_gradient'];bad=~torch.isfinite(actual);pgbad=~torch.isfinite(pggrad);kbad=~torch.isfinite(kg)
 delta=t['ref_logprob']-t['log_prob'];ratio_delta=t['log_prob']-t['old_log_prob']
 positions=bad.nonzero().tolist();points=[]
 for row,col in positions[:32]:
  points.append(dict(row=row,column=col,mask=float(mask[row,col]),log_prob=float(t['log_prob'][row,col]),old_log_prob=float(t['old_log_prob'][row,col]),ref_logprob=float(t['ref_logprob'][row,col]),advantage=float(t['advantages'][row,col]),ref_minus_actor=float(delta[row,col]),actor_minus_old=float(ratio_delta[row,col]),actual_grad=float(actual[row,col]),PPO_grad=float(pggrad[row,col]),KL_grad=float(kg[row,col])))
 summaries={}
 for name,value in t.items():
  finite=torch.isfinite(value);v=value[finite]
  summaries[name]=dict(shape=list(value.shape),dtype=str(value.dtype),nonfinite=int((~finite).sum()),finite_min=float(v.min()) if v.numel() else None,finite_max=float(v.max()) if v.numel() else None)
 records.append(dict(pid=record['pid'],input_path=str(path),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),input_bytes=path.stat().st_size,record=record,tensors=summaries,PPO_loss=float(pg),KL_loss=float(kl),actual_bad_count=int(bad.sum()),PPO_bad_count=int(pgbad.sum()),KL_bad_count=int(kbad.sum()),KL_bad_mask_exactly_matches_captured=bool(torch.equal(bad,kbad)),actual_bad_mask0_count=int((bad & (mask==0)).sum()),actual_bad_active_count=int((bad & (mask!=0)).sum()),ref_minus_actor_max=float(delta.max()),actor_minus_old_max=float(ratio_delta.max()),FP32_exp_overflow_log_threshold=math.log(torch.finfo(torch.float32).max),bad_points=points,all_bad_positions_recorded=len(positions)<=32))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(unix=time.time(),seconds=time.perf_counter()-started,owner_path=str(source),owner_resolved=str(source.resolve()),owner_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),functions={n:inspect.getsource(getattr(owner,n)) for n in ['compute_policy_loss','kl_penalty','agg_loss']},torch_version=torch.__version__,records=records,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,PSS_bytes=psutil.Process().memory_full_info().pss,CUDA_initialized=False,model_calls=0,DT_calls=0,optimizer_calls=0,worker_RPC=0,production_changes=0,loss_autograd_calls=4,scope='Actual iteration37 saved loss inputs; native CPU PPO/KL branch gradients, not a model replay or numerical-tolerance test.'),allow_nan=True))
'''

outer = r'''
import hashlib,json,os,subprocess
from pathlib import Path
p=Path(SOURCE)
assert hashlib.sha256(p.read_bytes()).hexdigest()=='1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658'
source=json.loads(p.read_bytes());env=dict(os.environ,**source['environment']);env.pop('RAY_ADDRESS',None)
env['CUDA_VISIBLE_DEVICES']='-1';env['PYTHONPATH']=source['pythonpath']
r=subprocess.run([env['VENV_PYTHON'],'-c',PROBE],env=env,capture_output=True,text=True,timeout=50)
print(r.stdout,end='')
if r.returncode:raise RuntimeError(r.stderr)
'''.replace('SOURCE',repr(transport.ROOT+'/runs/textcraft-formal-stable-20261009-v1/source.json'),1).replace('PROBE',repr(probe),1)

if __name__ == '__main__':
 stem=RAW/'native-incident37-loss-20261010-v1'
 assert not stem.with_suffix('.json').exists(),'Read the completed replay instead of repeating it'
 command='source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+outer+'\nPY\n'
 stem.with_suffix('.command.txt').write_bytes(command.encode())
 result=subprocess.run(transport.SSH+['bash','-s'],input=command.encode(),capture_output=True,timeout=60)
 stem.with_suffix('.stdout.txt').write_bytes(result.stdout);stem.with_suffix('.stderr.txt').write_bytes(result.stderr)
 result.check_returncode();data=json.loads(result.stdout);stem.with_suffix('.json').write_bytes(result.stdout)
 print(json.dumps(dict(saved=str(stem.with_suffix('.json')),seconds=data['seconds'],owner_sha256=data['owner_sha256'],records=[{k:v for k,v in d.items() if k not in ['tensors','record']} for d in data['records']],peak_RSS_bytes=data['peak_RSS_bytes'],CUDA_initialized=data['CUDA_initialized']),ensure_ascii=False))
