"""Verify baseline source bytes against the authors' already fetched Git tree."""
import argparse
import hashlib
import json
from pathlib import Path
from common import HERE,sha,byte_sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--tree',type=Path,required=True)
    a=p.parse_args();tree=json.loads(a.tree.read_bytes());assert tree['sha']=='075e7e44ae4d5acd2ed76e0d2aced57107d02736'
    entries={r['path']:r for r in tree['tree']};files={}
    for path in sorted(a.source.rglob('*.py')):
        relative=path.relative_to(a.source).as_posix();raw=path.read_bytes();normal=raw.replace(b'\r\n',b'\n')
        blob=hashlib.sha1(b'blob '+str(len(normal)).encode()+b'\0'+normal).hexdigest()
        assert blob==entries[relative]['sha'],relative
        files[relative]=dict(normalized_sha256=byte_sha(normal),raw_sha256=byte_sha(raw),git_blob_sha1=blob)
        # Keep the three newly used primitives; the seven existing core files
        # remain available in the already frozen source dependency records.
        if relative in ('perturbation_fast.py','lrp_patches.py','lrp_rules.py'):
            dest=HERE/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(normal)
    assert len(files)==9
    receipt=dict(author_commit=tree['sha'],git_tree_sha256=sha(a.tree),files=files)
    (HERE/'baseline_source_identity.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='verified_original_baseline_sources',files=len(files))))

if __name__=='__main__':main()
