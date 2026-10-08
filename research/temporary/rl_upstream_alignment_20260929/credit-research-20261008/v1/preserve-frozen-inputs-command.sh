source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
OUT='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/saved-source-mask-audit-20261009-v1'
import hashlib,io,json,tarfile,time
from pathlib import Path
out=Path(OUT);source=out/'result.json';raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='748f7e6bb0eb03702f0c815da5b191f92a37b48142879b29a1bb43e413efcd8a'
audit=json.loads(raw);archive=out/'native-inputs.tgz';assert not archive.exists(),'Do not overwrite a preserved archive'
manifest=[];tick=time.monotonic()
with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
 for task,record in audit['tasks'].items():
  for item in record['files']:
   path=Path(item['path']);data=path.read_bytes();assert len(data)==item['bytes'];assert hashlib.sha256(data).hexdigest()==item['sha256']
   name=task+'/'+path.name;info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data));manifest.append(dict(item,archive_member=name))
 info=tarfile.TarInfo('mask-audit-result.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
value=dict(unix=time.time(),archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),files=manifest,uncompressed_input_bytes=sum(f['bytes'] for f in manifest),elapsed_seconds=time.monotonic()-tick,model_forward=0,DT=0,optimizer=0,scope='64 frozen native input captures used by the mask audit, not all historical training data or model weights')
(out/'native-inputs-manifest.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))

PY
