"""Run CPU-only owner/input inventory, not the proposed GPU diagnostic."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from stage_environment_entry import ROOT, ENTRY, SSH, SCP


def main():
    out = ROOT + '/receipts/credit-layer-collection-inputs-20261008-v1'
    files = [HERE/'layer-collection-inputs.json', HERE/'inspect_layer_collection_inputs.py']
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    subprocess.run(SSH+['mkdir', '-p', out], check=True, timeout=30)
    for path in files:
        subprocess.run(SCP+[str(path), SSH[-1]+':'+out+'/'+path.name], check=True, timeout=45)
    body = '''import json,hashlib,os,subprocess,time
from pathlib import Path
out=Path(OUT);root=Path(ROOT)
result={'scope':'CPU exact-owner preparation; no model or GPU work','observed_unix':time.time(),'tasks':{}}
for name,h in HASHES.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h
for task in ('textcraft','appworld'):
 p=root/'runs/direct-target-prefix-runtime-20261007-v1'/task/(task+'-dt')/'source.json'
 source=json.loads(p.read_bytes());env=dict(os.environ,**source['environment'])
 env.pop('MACA_VISIBLE_DEVICES',None);env['CUDA_VISIBLE_DEVICES']='-1'
 env['PYTHONPATH']=':'.join([str(out),source['pythonpath'],str(Path(source['dt_root'])/'clean/qwen35')])
 target=out/(task+'-cpu-inventory.json')
 command=[env['VENV_PYTHON'],str(out/'inspect_layer_collection_inputs.py'),str(out/'layer-collection-inputs.json'),task,str(p),str(target)]
 run=subprocess.run(command,env=env,cwd=out,capture_output=True,timeout=60)
 (out/(task+'-cpu.stderr.txt')).write_bytes(run.stderr)
 result['tasks'][task]={'returncode':run.returncode,'stdout':run.stdout.decode(errors='replace'),'stderr':run.stderr.decode(errors='replace')[-4000:],'command':command}
 if run.returncode==0:
  result['tasks'][task]['receipt']=json.loads(target.read_bytes())
  result['tasks'][task]['sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
print(json.dumps(result))
'''
    body = 'OUT='+repr(out)+'\nROOT='+repr(ROOT)+'\nHASHES='+repr(hashes)+'\n'+body
    shell = 'source '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+body+'\nPY\n'
    (HERE/'layer-input-inventory-command.sh').write_text(shell, encoding='utf-8', newline='\n')
    run = subprocess.run(SSH+['bash', '-s'], input=shell.encode(), capture_output=True, timeout=150)
    (HERE/'layer-input-inventory.stderr.txt').write_bytes(run.stderr)
    run.check_returncode()
    result = json.loads(run.stdout)
    (HERE/'layer-input-inventory.json').write_text(json.dumps(result, indent=2)+'\n')
    for task, value in result['tasks'].items():
        print(task, value['returncode'], value['stdout'], value['stderr'])
    if any(value['returncode'] for value in result['tasks'].values()):
        raise SystemExit('Preserved actual owner/input inventory failure; no model was launched.')


if __name__ == '__main__':
    main()
