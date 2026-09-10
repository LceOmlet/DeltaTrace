"""Validate transfer manifest and atomically import immutable completed cases."""
import argparse
import json
from pathlib import Path,PurePosixPath
import zipfile
from common import HERE,sha,byte_sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
    with zipfile.ZipFile(a.archive) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        assert set(archive.namelist())=={'manifest.json'}|{r['path'] for r in manifest['files']}
        for row in manifest['files']:
            path=PurePosixPath(row['path']);assert not path.is_absolute() and '..' not in path.parts
            payload=archive.read(row['path']);assert len(payload)==row['bytes'] and byte_sha(payload)==row['sha256']
            target=HERE/path;target.resolve().relative_to(HERE.resolve())
            if target.exists() and path.parts[0] not in ('audit',):assert sha(target)==row['sha256']
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                temporary=target.with_name(target.name+'.partial');temporary.write_bytes(payload);temporary.replace(target)
        target=HERE/'transfers'/(a.archive.stem+'.json');target.parent.mkdir(parents=True,exist_ok=True)
        record=dict(archive_sha256=sha(a.archive),**manifest)
        target.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='verified_import',method_cases=len(manifest['method_cases']),files=len(manifest['files']))))

if __name__=='__main__':main()
