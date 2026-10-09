"""Read-only CPU extraction from already completed diagnostic artifacts."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ROOT,ENTRY,SSH,SCP


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    repo=HERE.parents[4]
    completed=repo/'experiments/rl/results_single_background_completed_20261009.json'
    summary=json.loads(completed.read_bytes())
    local=HERE/'single-background-layer-ledger-v1'
    local.mkdir(exist_ok=True)
    target=ROOT+'/receipts/credit-single-background-layer-ledger-20261009-v1'
    protocol=dict(scope='Original completed 165/task diagnostic; no new model or DT calls',
        completed_receipt_sha256=sha(completed),tasks={})
    for task in ('textcraft','appworld'):
        identity=summary['tasks'][task]['cross_cell_analysis']
        assert sha(identity['path'])==identity['sha256']
        points=json.loads(Path(identity['path']).read_bytes())['points']
        assert len(points)==165
        protocol['tasks'][task]=[{k:p[k] for k in ('traj_uid','packed_slot','token_id',
            'initial_state_sha256','cohorts','joint_d','single_d','single_root','native_interval','artifact')} for p in points]
    protocol_path=local/'protocol.json'
    raw=(json.dumps(protocol,indent=2)+'\n').encode()
    if protocol_path.exists():assert protocol_path.read_bytes()==raw
    else:protocol_path.write_bytes(raw)
    assert not (local/'result.json').exists(),'Do not repeat a completed extraction'
    subprocess.run(SSH+['bash','-s'],input=('test ! -e '+target+' && mkdir '+target+'\n').encode(),check=True,timeout=35)
    script=HERE/'inspect_saved_single_background_layers.py'
    subprocess.run(SCP+[str(script),str(protocol_path),SSH[-1]+':'+target+'/'],check=True,timeout=40)
    command='source '+shlex.quote(ENTRY+'/metax-entry.env.sh')+'\n'
    command+='CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$VENV_PYTHON" -u '+shlex.quote(target+'/'+script.name)
    command+=' --protocol '+shlex.quote(target+'/protocol.json')+' --output '+shlex.quote(target+'/result.json')+'\n'
    (local/'command.sh').write_text(command)
    try:
        run=subprocess.run(SSH+['bash','-s'],input=command.encode(),capture_output=True,timeout=180)
    except subprocess.TimeoutExpired:
        # An observation timeout does not authorize a second extraction.
        raise RuntimeError('Inspect the same remote result/process before any further action.')
    (local/'stdout.txt').write_bytes(run.stdout)
    (local/'stderr.txt').write_bytes(run.stderr)
    subprocess.run(SCP+[SSH[-1]+':'+target+'/result.json',str(local/'result.json')],check=True,timeout=45)
    run.check_returncode()
    result=json.loads((local/'result.json').read_bytes())
    assert result['phase']=='complete' and result['source']['sha256']==sha(script)
    assert result['protocol']['sha256']==sha(protocol_path) and not result['cuda_initialized']
    print(run.stdout.decode())


if __name__=='__main__':
    main()
