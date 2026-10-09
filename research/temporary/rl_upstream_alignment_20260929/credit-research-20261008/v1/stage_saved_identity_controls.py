"""Bind already-completed control rows and run one CPU-only extraction."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from stage_environment_entry import ROOT, ENTRY, SSH, SCP


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    collection_path = HERE/'layer-collection-inputs.json'
    collection = json.loads(collection_path.read_bytes())
    ledger_path = HERE/'single-background-layer-ledger-v1/result.json'
    ledger = json.loads(ledger_path.read_bytes())
    protocol = dict(scope='Read saved inactive identical-ID pairs; no model rerun',
        collection_sha256=sha(collection_path), ledger_sha256=sha(ledger_path), tasks={})
    for task, value in ledger['tasks'].items():
        source = collection['tasks'][task]
        entries = {e['traj_uid']:e for e in source['entries']}
        batches = []
        for batch in value['batches']:
            match = re.fullmatch(r'rank[01]-batch(\d+)-round(\d+)\.pt',
                Path(batch['artifact']['path']).name)
            assert match
            index, round_index = map(int, match.groups())
            original = source['batches'][index]
            uids = original['uids']
            padded = uids+[uids[-1]]*(4-len(uids))
            points = {(p['traj_uid'],p['packed_slot']) for p in batch['points']}
            identities = []
            active = set()
            for row, uid in enumerate(padded):
                entry = entries[uid]
                query = entry['queries'][round_index] if row < original['actual_rows'] and round_index < len(entry['queries']) else None
                if query is not None:
                    active.add((uid,query['packed_slot']))
                identities.append(dict(traj_uid=uid,initial_state_sha256=entry['initial_state_sha256'],
                    active_query=query is not None, query=query,
                    padding_identity_control=row >= original['actual_rows'],
                    batch=index,round=round_index))
            assert active == points
            batches.append(dict(artifact=batch['artifact'], rows=identities))
        protocol['tasks'][task] = batches
    local = HERE/'saved-identity-controls-v1'
    local.mkdir(exist_ok=True)
    path = local/'protocol.json'
    raw = (json.dumps(protocol, indent=2)+'\n').encode()
    if path.exists():
        assert path.read_bytes() == raw
    else:
        path.write_bytes(raw)
    assert not (local/'result.json').exists(), 'Never repeat a completed extraction'
    target = ROOT+'/receipts/credit-saved-identity-controls-20261009-v1'
    subprocess.run(SSH+['bash','-s'], input=('test ! -e '+target+' && mkdir '+target+'\n').encode(), check=True, timeout=35)
    script = HERE/'inspect_saved_identity_controls.py'
    subprocess.run(SCP+[str(script),str(path),SSH[-1]+':'+target+'/'], check=True, timeout=45)
    command = 'source '+shlex.quote(ENTRY+'/metax-entry.env.sh')+'\n'
    command += 'CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$VENV_PYTHON" -u '+shlex.quote(target+'/'+script.name)
    command += ' --protocol '+shlex.quote(target+'/protocol.json')+' --output '+shlex.quote(target+'/result.json')+'\n'
    (local/'command.sh').write_text(command)
    try:
        run = subprocess.run(SSH+['bash','-s'], input=command.encode(),capture_output=True,timeout=180)
    except subprocess.TimeoutExpired:
        raise RuntimeError('Re-observe the same remote process/result; do not repeat extraction.')
    (local/'stdout.txt').write_bytes(run.stdout)
    (local/'stderr.txt').write_bytes(run.stderr)
    subprocess.run(SCP+[SSH[-1]+':'+target+'/result.json',str(local/'result.json')], check=True,timeout=45)
    run.check_returncode()
    result = json.loads((local/'result.json').read_bytes())
    assert result['phase']=='complete' and result['source']['sha256']==sha(script)
    assert result['protocol']['sha256']==sha(path) and not result['cuda_initialized']
    print(run.stdout.decode())


if __name__ == '__main__':
    main()
