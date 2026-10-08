source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-single-background-nonfinite-appworld-20261009-v1'
SOURCE='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/direct-target-mlp-token-chunk-20261007-v1/deltatrace/clean/qwen35/qwen35_gdn_finite.py'
import torch,json,ast,hashlib,time
from pathlib import Path
import torch.nn.functional as F
torch.set_num_threads(8)
out=Path(OUT);p=out/'results/rank0-first-nonfinite.pt'
x=torch.load(p,map_location='cpu',weights_only=False)
a=x['args'];result={'unix':time.time(),'artifact_sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),'callback':x['callback'],'layer':x['layer'],'args':[]}
for i,v in enumerate(a):
 valid=torch.isfinite(v)
 result['args'].append(dict(index=i,shape=list(v.shape),dtype=str(v.dtype),nonfinite=int((~valid).sum()),finite_maxabs=float(v[valid].abs().max()),first_nonfinite=(~valid).nonzero()[:12].tolist()))
bad=(~torch.isfinite(x['output'])).nonzero();result['output_nonfinite']=int(len(bad));result['bad_time_positions']=bad[:10000,2].unique().tolist();result['samples']=[]
source=Path(SOURCE).read_text();tree=ast.parse(source);names=('_scalar_secant','_silu_derivative','_conv_silu_finite_rule')
selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert len(selected)==3
ns={'torch':torch,'F':F};exec(compile(ast.Module(body=selected,type_ignores=[]),SOURCE,'exec'),ns)
for b,c,t in bad[:16].tolist():
 inputs=[a[0][2*b:2*b+2,c:c+1,t:t+1],a[1][2*b:2*b+2,c:c+1,t:t+1],a[2][b:b+1,c:c+1,t:t+1]]
 value=ns['_conv_silu_finite_rule'](*inputs)
 result['samples'].append(dict(position=[b,c,t],pre=inputs[0].flatten().tolist(),native_silu=inputs[1].flatten().tolist(),upstream=inputs[2].flatten().tolist(),eager_owner=value.flatten().tolist(),compiled=x['output'][b,c,t].item()))
path=out/'results/rank0-first-nonfinite-cpu.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

PY
