python3 - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/appworld-row-cuts-finite-20261007-v1')
files=[]
for f in sorted(p.iterdir()):
    if f.is_file():
        raw=f.read_bytes() if f.suffix in ('.json','.cu','.cpp','.sh') or f.name=='libfinite_row_query_starts.so' else None
        row=dict(path=str(f),bytes=f.stat().st_size)
        if raw is not None: row['sha256']=hashlib.sha256(raw).hexdigest()
        if raw is not None and f.suffix in ('.json','.cu','.cpp','.sh') and len(raw)<300000: row['text']=raw.decode()
        files.append(row)
print(json.dumps(dict(scope='Immediate actual production build directory only; no recursive scan or GPU',files=files)))
PY
