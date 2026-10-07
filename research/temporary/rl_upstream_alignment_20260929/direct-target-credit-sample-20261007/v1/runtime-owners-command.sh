set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'

from pathlib import Path
import json,os,subprocess,hashlib
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');out=root/'receipts/direct-target-credit-sample-20261007-v1';pop=json.loads((out/'population.json').read_bytes());results={}
child=r"""
import hashlib,inspect,json,os
from pathlib import Path
import torch,counterfactual
from transformers import AutoTokenizer
torch.set_num_threads(1)
f=counterfactual.reward_event_token_credit;owner=Path(counterfactual.__file__);unwrapped=Path(inspect.getsourcefile(inspect.unwrap(f)));wrapped=Path(inspect.getsourcefile(f));data=json.loads(os.environ['DIAG_LABEL_IDS'])
tok=AutoTokenizer.from_pretrained('/mnt/si0021787ci2/default/models/Qwen3.5-9B',local_files_only=True)
r=dict(owner_path=str(owner),owner_sha256=hashlib.sha256(owner.read_bytes()).hexdigest(),unwrapped_path=str(unwrapped),wrapped_path=str(wrapped),labels={str(i):tok.decode([i],skip_special_tokens=False,clean_up_tokenization_spaces=False) for i in data},cuda_initialized=torch.cuda.is_initialized())
assert not r['cuda_initialized'];print(json.dumps(r))
"""
for task in ('appworld','textcraft'):
 source=json.loads(Path(pop['tasks'][task]['source']['path']).read_bytes());data=json.loads((out/('results-'+task)/(task+'-rank0.json')).read_bytes());ids=sorted({v for mode in data['modes'].values() for row in mode['rows'] for v in row.get('dominant_future_labels',[])})
 env=dict(os.environ,**source['environment']);env['CUDA_VISIBLE_DEVICES']='-1';env.pop('MACA_VISIBLE_DEVICES',None);env['HF_HUB_OFFLINE']='1';env['TOKENIZERS_PARALLELISM']='false';env['DIAG_LABEL_IDS']=json.dumps(ids)
 dt=Path(env['DT_ROOT']);qwen=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35'];env['PYTHONPATH']=':'.join([str(out),str(dt),env.get('DT_OFFICIAL_ROOT') or qwen['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],qwen['ft_extension_root']])
 p=subprocess.run([env['VENV_PYTHON'],'-c',child],capture_output=True,text=True,env=env,check=True);(out/(task+'-cpu-owner.stderr.txt')).write_text(p.stderr);r=json.loads(p.stdout);assert r['owner_sha256']==pop['tasks'][task]['credit_owner']['sha256'];results[task]=r
(out/'runtime-owners-and-labels.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results))

PY
