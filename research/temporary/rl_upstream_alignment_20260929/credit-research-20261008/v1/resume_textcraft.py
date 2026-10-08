"""Release the existing one-shot pre-update hold, without relaunch or patch."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("transport", HERE.parents[1] / "stage_environment_entry.py")
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)

script = '''source {entry}/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
import hashlib,json,psutil,time
from pathlib import Path
root=Path('{root}')
run=root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt'
hold=root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
expected={{2833207:1791370325.16,2838967:1791370371.45,2840776:1791370385.54}}
processes=[]
for pid,birth in expected.items():
    p=psutil.Process(pid)
    assert p.create_time()==birth,(pid,p.create_time(),birth)
    assert p.status()!=psutil.STATUS_ZOMBIE
    processes.append(dict(pid=pid,birth=birth))
source=run/'source.json'
source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
assert source_sha=='2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52'
sidecars=[json.loads((hold/f'rank{{r}}-hold.json').read_text()) for r in range(2)]
for rank,pid in enumerate([2838967,2840776]):
    s=sidecars[rank]
    assert s['pid']==pid and s['rank']==rank and s['actor_saved'] and s['hold_entered']
    assert s['release_file']==str(hold/f'rank{{rank}}-release-update')
receipt=dict(observed_unix=time.time(),authorization='Human: TextCraft可以跑起来，剩下两卡用以调试',
    action='Release original one-shot pending first update',processes=processes,
    source_path=str(source),source_sha256=source_sha,devices=[2,3],
    production_candidate_deployed=False,checkpoint_restored=False,parameters_changed=False,
    pre_release_sidecars=sidecars,release_files=[])
for rank in range(2):
    p=hold/f'rank{{rank}}-release-update'
    if not p.exists():
        with p.open('x') as f:
            json.dump(dict(authorization=receipt['authorization'],created_unix=time.time(),
                source_sha256=source_sha,candidate_deployed=False),f,ensure_ascii=False)
    receipt['release_files'].append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
receipt_path=hold/'resume-20261008.json'
if not receipt_path.exists():
    receipt_path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\\n')
else:
    receipt=json.loads(receipt_path.read_text())
print(json.dumps(receipt,ensure_ascii=False))
PY
'''.format(entry=transport.ENTRY, root=transport.ROOT)
result = subprocess.run(transport.SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
if result.returncode:
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
receipt = json.loads(result.stdout)
(HERE / 'textcraft-resume.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k!='pre_release_sidecars'}, ensure_ascii=False))
