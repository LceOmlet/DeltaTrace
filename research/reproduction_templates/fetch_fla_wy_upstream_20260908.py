"""Pin the upstream source that already multiplies before transposing."""
import hashlib
import json
import urllib.request
from pathlib import Path
A=Path(__file__).resolve().parent/'fla_wy_upstream_20260908';A.mkdir(exist_ok=True)
def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'DeltaTrace-source-audit'}),timeout=30) as r:
        return r.read()
raw=fetch('https://api.github.com/repos/fla-org/flash-linear-attention/commits/main')
commit=json.loads(raw)['sha'];(A/'commit.json').write_bytes(raw)
url='https://raw.githubusercontent.com/fla-org/flash-linear-attention/'+commit+'/fla/ops/gated_delta_rule/wy_fast.py'
source=fetch(url);(A/'wy_fast.py').write_bytes(source)
assert b'b_kb = b_k * b_b[:, None]' in source
assert b'tl.trans(tl.dot(tl.trans(b_kb).to(b_dA.dtype), b_dA))' in source
receipt={'commit':commit,'source_url':url,'source_sha256':hashlib.sha256(source).hexdigest(),
         'commit_response_sha256':hashlib.sha256(raw).hexdigest(),
         'issue_url':'https://github.com/fla-org/hybrid-distillation/issues/2',
         'scope':'Backport only the existing two-line multiplication-before-transpose expression; not the changed upstream API or entire kernel.'}
(A/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))
