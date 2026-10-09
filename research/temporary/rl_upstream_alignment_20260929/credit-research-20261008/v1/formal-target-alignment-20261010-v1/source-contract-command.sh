source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
from pathlib import Path
import json,hashlib,psutil
r=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');f=r/'runs/textcraft-formal-stable-20261009-v1'
m=json.loads((f/'source.json').read_bytes());clean=Path(m['dt_root'])/'clean/qwen35'
wrap=Path(m['dt_root'])/'experiments/rl/deltatrace_credit.py'
spec=[
 ('configured_trace_wrapper',wrap,'8046762ae2fd149b299e29f9a331d8ae1aed665f2de895573e78e61ebc2d8497',[(41,63),(74,133)]),
 ('target_packing_and_logprob_owner',clean/'qwen35_answer_finite.py','d47333ea68fb7a332e7d1dce7913c989d875ea262dfe49cfa4f20f7c35ebe03e',[(15,57),(186,195)]),
 ('active_finite_runner',clean/'qwen35_dense_finite_runner.py','5f14bb3cdb491e4d5b3b531607e00936bae76e055e2286ed60e096c6c5d2e555',[(366,380),(522,529)]),
 ('original_direct_readout',f/'entry/reward_readout.py','814cfe929b4afcc5ce3570450ea8aca269b1097db7abed83dc917df3f1d1cc9b',[(500,533)]),
]
records=[]
for name,p,expected,ranges in spec:
 raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest();assert sha==expected,(name,sha)
 lines=raw.decode().splitlines()
 records.append(dict(name=name,path=str(p),resolved=str(p.resolve()),sha256=sha,
  excerpts=[dict(start=start,end=end,lines=lines[start-1:end]) for start,end in ranges]))
over=f/'runtime-overrides/credit-records-20261009-v2.json'
override=json.loads(over.read_bytes());assert override['complete']
records.append(dict(name='completed_passive_record_override',path=str(over),sha256=hashlib.sha256(over.read_bytes()).hexdigest(),
 active_method_sources=[dict(path=v['active_method_source'],sha256=v['active_method_source_sha256'],method_math_AST_unchanged=v['method_math_AST_unchanged']) for v in override['results']]))
print(json.dumps(dict(sources=records,scope='Source excerpts and previously completed passive-record AST check only; no model or runtime mutation. Actual per-target GPU scores and pre-scatter vectors are not reconstructed.')))
PY
