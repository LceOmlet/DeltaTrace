"""Download and hash the author's fixed Longformer MLM dependency for REAGENT."""
import hashlib
import argparse
import json
from pathlib import Path
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor,as_completed

HERE=Path(__file__).resolve().parent
REPOSITORY='allenai/longformer-base-4096'
REVISION='301e6a42cb0d9976a6d6a26a079fef81c18aa895'
FILES=('config.json','merges.txt','pytorch_model.bin','tokenizer.json','vocab.json')

def download_file(url,path,size):
    temporary=path.with_name(path.name+'.partial')
    if size<16*1024*1024:
        with urllib.request.urlopen(url,timeout=60) as response:
            payload=response.read()
        assert len(payload)==size;temporary.write_bytes(payload)
    else:
        chunks=path.parent/(path.name+'.chunks');chunks.mkdir(exist_ok=True)
        step=4*1024*1024
        def chunk(start):
            end=min(size,start+step)-1;dest=chunks/str(start)
            if dest.exists() and dest.stat().st_size==end-start+1:return dest
            for attempt in range(4):
                try:
                    req=urllib.request.Request(url+f'?download=true&range_start={start}',headers={'Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(req,timeout=60) as response:
                        assert response.status==206 and response.headers['Content-Range']==f'bytes {start}-{end}/{size}'
                        payload=response.read(end-start+2)
                    assert len(payload)==end-start+1
                    part=dest.with_name(dest.name+'.partial');part.write_bytes(payload);part.replace(dest);return dest
                except Exception:
                    if attempt==3:raise
                    time.sleep(1+attempt)
        done=0
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures={pool.submit(chunk,start):start for start in range(0,size,step)}
            for future in as_completed(futures):
                future.result();done+=1
                if done%16==0:print(json.dumps(dict(file=path.name,completed_chunks=done,total_chunks=len(futures))),flush=True)
        with temporary.open('wb') as output:
            for start in range(0,size,step):output.write((chunks/str(start)).read_bytes())
        assert temporary.stat().st_size==size
    temporary.replace(path)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--small-only',action='store_true');a=p.parse_args()
    root=HERE/'.assets/longformer';root.mkdir(parents=True,exist_ok=True)
    tree=json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{REPOSITORY}/tree/{REVISION}?recursive=true',timeout=30))
    records={r['path']:r for r in tree};receipts=[]
    for name in FILES:
        source=records[name];path=root/name
        if a.small_only and name=='pytorch_model.bin':
            receipts.append(dict(name=name,bytes=source['size'],sha256=source['lfs']['oid'],git_oid=source['oid'],pending_user_download=True))
            continue
        if not path.exists():
            url=f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}'
            download_file(url,path,source['size'])
        payload=path.read_bytes();assert len(payload)==source['size']
        digest=hashlib.sha256(payload).hexdigest()
        if 'lfs' in source:assert digest==source['lfs']['oid']
        else:assert hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()==source['oid']
        receipts.append(dict(name=name,bytes=len(payload),sha256=digest,git_oid=source['oid']))
        print(json.dumps(dict(file=name,status='hash_verified',bytes=len(payload))),flush=True)
    (HERE/'mlm_identity.json').write_text(json.dumps(dict(repository=REPOSITORY,revision=REVISION,files=receipts),indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()
