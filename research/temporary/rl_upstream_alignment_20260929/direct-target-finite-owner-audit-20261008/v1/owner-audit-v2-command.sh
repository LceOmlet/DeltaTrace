set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import json,hashlib,torch,time,subprocess,psutil,sys,os
from pathlib import Path
root=Path("/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922")
p=root/"receipts/direct-target-extreme-operator-20261007-v1/results/rank0-decoder27-fa.pt"
source=json.loads((root/"runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json").read_bytes());dt=Path(source["environment"]["DT_ROOT"]);q=json.loads(Path(source["environment"]["DT_ENVIRONMENT_JSON"]).read_bytes())["qwen35"];sys.path[:0]=[str(dt),str(dt/"clean/qwen35"),str(dt/"clean/qwen3"),q["official_root"],q["ft_extension_root"],*source["pythonpath"].split(":")]
os.environ.update(source["environment"])
os.environ["CUDA_VISIBLE_DEVICES"]="4"
d=torch.load(p,map_location="cpu",weights_only=False,mmap=True)
r={"unix":time.time(),"path":str(p),"keys":list(d),"ops":{k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in d['ops'].items()},"coefficients":{k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in d['coefficients'].items()},"owner_path":d['owner_path'],"owner_sha256":d['owner_sha256']}
owner=Path(d['owner_path']);r['owner_text']=owner.read_text();r['actual_owner_sha256']=hashlib.sha256(owner.read_bytes()).hexdigest()
source=json.loads((root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json').read_bytes());r['source_sha256']=hashlib.sha256((root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json').read_bytes()).hexdigest()
q=json.loads(Path(source['environment']['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
search=[str(Path(source['environment']['DT_ROOT'])/'clean/qwen35'),q['ft_extension_root']]
cmd=['rg','--files',*search,'-g','*finite*','-g','*test*','-g','*reference*']
v=subprocess.run(cmd,capture_output=True,text=True);r['reference_candidates']={'command':cmd,'returncode':v.returncode,'stdout':v.stdout,'stderr':v.stderr}
r['text_driver_same_birth']=psutil.pid_exists(2833207) and psutil.Process(2833207).create_time()==1791370325.16
r['text_releases']=[(root/f'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt/rank{i}-release-update').exists() for i in (0,1)]
r['physical']=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
r['cuda_initialized']=torch.cuda.is_initialized();assert not r['cuda_initialized']
print(json.dumps(r))

PY
