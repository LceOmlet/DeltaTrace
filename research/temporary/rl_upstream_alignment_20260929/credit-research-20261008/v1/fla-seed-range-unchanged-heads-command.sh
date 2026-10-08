source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
ROOT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
import torch,hashlib,json,base64
from pathlib import Path
torch.set_num_threads(8)
root=Path(ROOT);old_path=root/'receipts/credit-single-background-nonfinite-appworld-20261009-v2/results/rank0-first-nonfinite.pt';new_path=root/'receipts/fla-seed-range-20261009-v2/result.pt'
old=torch.load(old_path,map_location='cpu',weights_only=False)['output'];new=torch.load(new_path,map_location='cpu',weights_only=False);exponent=new['exponent'];result={'old_sha256':hashlib.file_digest(old_path.open('rb'),'sha256').hexdigest(),'new_sha256':hashlib.file_digest(new_path.open('rb'),'sha256').hexdigest(),'overflow_heads':int((exponent>0).sum()),'unchanged_heads':int((exponent==0).sum()),'fields':{}}
for name,value in old.items():
 mask=(exponent==0) if value.ndim==4 else (exponent[...,0]==0)
 a=value[mask.expand_as(value)];z=new['outputs'][0][name][mask.expand_as(value)]
 result['fields'][name]={'old_nonfinite':int((~torch.isfinite(a)).sum()),'exactly_equal':torch.equal(a,z),'maxabs_difference':float((a-z).abs().max())}
out=root/'receipts/fla-seed-range-20261009-v2';(out/'unmodified-heads.json').write_text(json.dumps(result,indent=2)+'\n')
files=[]
for revision in ('v1','v2'):
 folder=root/('receipts/fla-seed-range-20261009-'+revision)
 for name in ('launch.json','result.json','driver.log','check_fla_seed_range.py','unmodified-heads.json'):
  p=folder/name
  if p.exists():
   raw=p.read_bytes();files.append({'name':revision+'-'+name,'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'base64':base64.b64encode(raw).decode()})
p=root/'candidates/direct-target-mlp-token-chunk-20261007-v1/deltatrace/clean/qwen35/qwen35_gdn_finite.py';raw=p.read_bytes();files.append({'name':'original-AppWorld-GDN.py','path':str(p),'resolved':str(p.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'base64':base64.b64encode(raw).decode()})
print(json.dumps({'comparison':result,'files':files}))

PY
