"""Receipt for the uploaded auxiliary checkpoint against the frozen manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--mlm',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
    plan=json.loads((HERE/'protocol.json').read_bytes());files=[]
    for expected in plan['mlm']['files']:
        path=a.mlm/expected['name'];actual=sha(path)
        assert path.stat().st_size==expected['bytes'] and actual==expected['sha256']
        files.append(dict(name=expected['name'],bytes=path.stat().st_size,sha256=actual,mtime_ns=path.stat().st_mtime_ns))
    receipt=dict(status='all_five_uploaded_assets_verified',repository=plan['mlm']['repository'],revision=plan['mlm']['revision'],
        protocol_sha256=sha(HERE/'protocol.json'),files=files,verified_at=time.time(),verifier_sha256=sha(Path(__file__)))
    a.output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=receipt['status'],files=len(files),receipt_sha256=sha(a.output))))

if __name__=='__main__':main()
