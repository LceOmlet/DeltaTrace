"""Make the nine already verified author source dependencies self-contained."""
import argparse
import json
from pathlib import Path
from common import HERE,byte_sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    identity=json.loads((HERE/'baseline_source_identity.json').read_bytes())
    for name,record in identity['files'].items():
        payload=(a.source/name).read_bytes().replace(b'\r\n',b'\n')
        assert byte_sha(payload)==record['normalized_sha256']
        destination=HERE/'source_snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert destination.read_bytes()==payload
        else:destination.write_bytes(payload)
    print('Archived all nine verified original source files')

if __name__=='__main__':main()
