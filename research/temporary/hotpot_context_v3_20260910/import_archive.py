"""Verify an immutable downloaded archive before importing its compressed bytes."""
import argparse
import gzip
import json
from pathlib import Path, PurePosixPath
import zipfile
from artifacts_v3 import HERE,sha,byte_sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    a=p.parse_args();assert sha(a.archive)==a.sha256
    dest=HERE/'raw/conditioned_v3';assert not dest.exists()
    with zipfile.ZipFile(a.archive) as z:
        manifest=json.loads(z.read('manifest.json'))
        assert manifest['status']=='complete' and len(manifest['files'])==7
        assert set(z.namelist())=={'manifest.json'}|{r['path']+'.gz' for r in manifest['files']}
        data={}
        for r in manifest['files']:
            relative=PurePosixPath(r['path']+'.gz')
            assert not relative.is_absolute() and '..' not in relative.parts
            packed=z.read(str(relative));unpacked=gzip.decompress(packed)
            assert len(packed)==r['compressed_bytes'] and byte_sha(packed)==r['compressed_sha256']
            assert len(unpacked)==r['bytes'] and byte_sha(unpacked)==r['sha256']
            data[str(relative)]=packed
        data['manifest.json']=z.read('manifest.json')
    for name,value in data.items():
        path=dest/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
    receipt=json.loads(a.receipt.read_bytes())
    receipt['archive_sha256']=a.sha256
    receipt['archive_bytes']=a.archive.stat().st_size
    receipt['manifest_sha256']=sha(dest/'manifest.json')
    (HERE/'execution_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='verified_and_imported',files=7,sha256=a.sha256)))

if __name__=='__main__':main()
