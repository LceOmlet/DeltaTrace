"""Losslessly archive the completed, identity-checked three-shard answer run."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

sha = lambda b: hashlib.sha256(b).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    state = json.loads((a.run/'progress.json').read_bytes())
    assert state['status']=='complete' and state['case_count']==48 and state['controls']==8
    files = ['progress.json']
    for item in state['completed']:
        for name in ('results.json', 'vectors.npz'):
            relative = item['chunk']+'/'+name
            assert sha((a.run/relative).read_bytes())==item[name.split('.')[0]+'_sha256']
            files.append(relative)
    assert len(files)==7 and not a.output.exists()
    manifest = dict(format='gzip_original_bytes_v1', status='complete', files=[])
    with zipfile.ZipFile(a.output, 'x', compression=zipfile.ZIP_STORED) as z:
        for relative in files:
            data=(a.run/relative).read_bytes(); packed=gzip.compress(data, compresslevel=9, mtime=0)
            manifest['files'].append(dict(path=relative, bytes=len(data), sha256=sha(data),
                compressed_bytes=len(packed), compressed_sha256=sha(packed)))
            z.writestr(relative+'.gz', packed)
        z.writestr('manifest.json', json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(status='archived', files=7, bytes=a.output.stat().st_size, sha256=sha(a.output.read_bytes()))))

if __name__=='__main__': main()
