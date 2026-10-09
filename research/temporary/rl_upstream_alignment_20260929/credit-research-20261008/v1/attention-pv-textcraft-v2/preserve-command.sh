source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/credit-attention-pv-textcraft-20261009-v2'

import hashlib,json,psutil,subprocess,tarfile,time
from pathlib import Path
out=Path(OUT);launch=json.loads((out/'launch.json').read_bytes())
try:alive=psutil.Process(launch['pid']).create_time()==launch['birth']
except psutil.Error:alive=False
assert not alive
records=[json.loads((out/'results'/f'rank{rank}.json').read_bytes()) for rank in (0,1)]
assert all(v['phase']=='complete' for v in records)
artifacts=[a for v in records for b in v['batches'] for a in b['attention_PV_readout']['original_operand_artifacts']]
assert len({a['path'] for a in artifacts})==len(artifacts)
for item in artifacts:assert Path(item['path']).stat().st_size==item['bytes']
manifest=dict(files=artifacts,bytes=sum(a['bytes'] for a in artifacts),scope='All original operands retained remotely; SHA256 computed during saving, sizes rechecked here. Large tensors are not copied to local.',copied_local=False)
(out/'raw-operator-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
names=['raw-operator-manifest.json','launch.json','preparation.json','driver.log','observer-source.tar','suboperation-protocol.json',*launch['scripts'],'results/rank0.json','results/rank1.json','results/rank0-phases.jsonl','results/rank1-phases.jsonl']
names=list(dict.fromkeys(names));names=[n for n in names if (out/n).exists()]
archive=out/'diagnostic-debug.tgz'
with tarfile.open(archive,'x:gz') as stream:
 for name in names:stream.add(out/name,arcname=name)
files=[dict(remote=str(out/n),name=n,bytes=(out/n).stat().st_size,sha256=hashlib.sha256((out/n).read_bytes()).hexdigest()) for n in names]
value=dict(unix=time.time(),driver_alive=alive,files=files,archive=dict(remote=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),status='Completed passive PV measurement; no candidate or repair',physical=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout,host=psutil.virtual_memory()._asdict())
print(json.dumps(value))

PY
