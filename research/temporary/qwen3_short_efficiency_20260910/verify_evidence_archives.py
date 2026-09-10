"""Verify the two raw archives and safely extract unchanged bytes for CPU replay."""
import argparse,hashlib,json,zipfile
from pathlib import Path,PurePosixPath


def verify(study,destination=None):
    reports=[];sha=lambda b:hashlib.sha256(b).hexdigest()
    for receipt_name in ['raw_archive_verification.json','author_archive_verification.json']:
        receipt=json.loads((study/receipt_name).read_bytes());path=study/receipt['archive']
        assert sha(path.read_bytes())==receipt['sha256'] and path.stat().st_size==receipt['bytes']
        with zipfile.ZipFile(path) as z:
            manifest=json.loads(z.read('archive_manifest.json'))
            assert len(manifest['files'])==manifest['files_count']==receipt['files_verified']
            assert set(z.namelist())==set(manifest['files'])|{'archive_manifest.json'}
            for name,row in manifest['files'].items():
                key=PurePosixPath(name);assert not key.is_absolute() and '..' not in key.parts
                data=z.read(name);assert len(data)==row['bytes'] and sha(data)==row['sha256'],name
                if destination:
                    target=destination.joinpath(*key.parts).resolve();assert target.is_relative_to(destination.resolve())
                    if target.exists():assert target.read_bytes()==data,name
                    else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        reports.append({'archive':receipt['archive'],'verified_files':manifest['files_count'],'sha256':receipt['sha256']})
    return reports


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--extract',type=Path)
    a=p.parse_args();print(json.dumps(verify(a.study,a.extract),indent=2))
