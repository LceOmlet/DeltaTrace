"""Technical pilot followed by the fixed full run, with per-case checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('environment','preflight','output','logs'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--methods',nargs='+',required=True);p.add_argument('--mlm',type=Path)
    a=p.parse_args();plan=json.loads((HERE/'protocol.json').read_bytes())
    assert set(a.methods)<=set(plan['new_methods'])
    a.logs.mkdir(parents=True,exist_ok=True)
    receipt=dict(start=time.time(),methods=a.methods,protocol_sha256=sha(HERE/'protocol.json'),
        controller_sha256=sha(Path(__file__)),phases=[])
    receipt_path=a.logs/'controller.json';assert not receipt_path.exists()
    common=[sys.executable,'-u',str(HERE/'evaluate_baselines.py'),'--environment',str(a.environment),
        '--preflight',str(a.preflight),'--output',str(a.output),'--methods',*a.methods,
        '--datasets',*plan['tasks']]
    if a.mlm:common+=['--mlm',str(a.mlm)]
    for phase,extra in [('pilot',['--indices','0']),('full',[])]:
        log_path=a.logs/(phase+'.log');started=time.time()
        phase_receipt=dict(name=phase,start=started,command=common+extra,status='running')
        receipt['phases'].append(phase_receipt)
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(dict(phase=phase,status='started')),flush=True)
        with log_path.open('x') as log:
            child=subprocess.Popen(common+extra,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,MACA_PATH='/opt/maca'))
            phase_receipt['pid']=child.pid;receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
            returncode=child.wait()
        phase_receipt.update(end=time.time(),returncode=returncode,log_sha256=sha(log_path),
            status='complete' if returncode==0 else 'failed')
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(dict(phase=phase,status=phase_receipt['status'],seconds=time.time()-started)),flush=True)
        if returncode:raise SystemExit(returncode)
    receipt.update(end=time.time(),status='complete');receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
