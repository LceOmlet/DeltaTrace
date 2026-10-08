"""Invoke the CPU-only saved-vector diagnostic in the existing runtime."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ROOT,ENTRY,SSH,SCP


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-endpoint-ledger',action='store_true')
    args=parser.parse_args()
    stem='full-credit-endpoint-ledger' if args.include_endpoint_ledger else 'full-credit-bounds'
    folder=HERE/'conditional-mixers-collection-v1'
    out=ROOT+'/receipts/conditional-mixers-collection-textcraft-20261009-v1'
    script=HERE/'analyze_saved_full_credit.py'
    remote_name=script.stem+'-'+hashlib.sha256(script.read_bytes()).hexdigest()[:12]+script.suffix
    for rank in range(2):
        assert json.loads((folder/('textcraft-rank'+str(rank)+'.json')).read_bytes())['phase']=='complete'
    subprocess.run(SCP+[str(script),SSH[-1]+':'+out+'/'+remote_name],check=True,timeout=30)
    records=[out+'/textcraft-v3-rank'+str(i)+'.json' for i in range(2)]
    records += [ROOT+'/receipts/conditional-gdn-collection-textcraft-20261009-v1/results/rank'+str(i)+'.json' for i in range(2)]
    records += [out+'/results/rank'+str(i)+'.json' for i in range(2)]
    body='import hashlib,json,subprocess,os\nfrom pathlib import Path\n'
    body+='p=Path('+repr(out+'/'+remote_name)+')\nassert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(hashlib.sha256(script.read_bytes()).hexdigest())+'\n'
    command=['--include-endpoint-ledger'] if args.include_endpoint_ledger else []
    body+='subprocess.run([os.environ["VENV_PYTHON"],str(p),"--records",*'+repr(records)+',"--output",'+repr(out+'/'+stem+'.json')+',*'+repr(command)+'],check=True)\n'
    body+='q=Path('+repr(out+'/'+stem+'.json')+'); print(json.dumps(dict(remote=str(q),bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest())))\n'
    shell='source '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+body+'\nPY\n'
    (folder/(stem+'-command.sh')).write_bytes(shell.encode())
    result=subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,timeout=75)
    (folder/(stem+'.stderr.txt')).write_bytes(result.stderr)
    (folder/(stem+'.stdout.txt')).write_bytes(result.stdout)
    result.check_returncode()
    subprocess.run(SCP+[SSH[-1]+':'+out+'/'+stem+'.json',str(folder/(stem+'.json'))],check=True,timeout=30)
    receipt=json.loads(result.stdout.splitlines()[-1])
    assert hashlib.sha256((folder/(stem+'.json')).read_bytes()).hexdigest()==receipt['sha256']
    receipt.update(local=str((folder/(stem+'.json')).resolve()),SHA256_verified=True,
        source_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
    (folder/(stem+'-preserved.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(result.stdout.decode())


if __name__=='__main__':main()
