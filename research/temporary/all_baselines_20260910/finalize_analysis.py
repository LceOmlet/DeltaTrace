"""Resume final CPU checks from the already imported, hash-verified full archive."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import HERE,ROOT,BASELINES,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
    ledger=json.loads((HERE/'transfers'/(a.archive.stem+'.json')).read_bytes())
    assert sha(a.archive)==ledger['archive_sha256']
    assert len(ledger['method_cases'])==2240
    assert Counter(r['method'] for r in ledger['method_cases'])=={m:448 for m in BASELINES}
    env=dict(os.environ,PYTHONPATH=str(HERE.parent/'needle_gap_20260910/.deps'))
    for script in ('scoring_precision_control.py','build_report.py','verify_outputs.py'):
        print(json.dumps(dict(stage=script)),flush=True)
        subprocess.run([sys.executable,'-u',str(HERE/script)],cwd=ROOT,env=env,check=True)
    verification=json.loads((HERE/'verification.json').read_bytes());assert verification['status']=='passed'
    receipt=dict(status='all_baselines_collected_scored_reported_verified',completed=time.time(),archive=a.archive.name,
        archive_sha256=ledger['archive_sha256'],verification_sha256=sha(HERE/'verification.json'),
        finalizer_sha256=sha(Path(__file__)),
        continuation='Resumed from the verified full archive after restoring original HotpotQA CPU scoring precision.')
    (HERE/'collection_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
