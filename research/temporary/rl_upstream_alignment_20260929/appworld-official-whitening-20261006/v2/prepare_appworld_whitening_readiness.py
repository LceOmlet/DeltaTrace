"""Prepare the existing App whitening sources with verified LOOP readiness."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

LOCAL=Path(__file__).resolve().parent
AUDIT=LOCAL.parents[1]
sys.path.insert(0,str(AUDIT))
from stage_environment_entry import AUDIT,ENTRY,REPO,ROOT,SCP,SSH

OUT=ROOT+'/receipts/appworld-official-whitening-20261006-v2'
V1=ROOT+'/receipts/appworld-official-whitening-20261006-v1'
FORMAL_OUTPUT=ROOT+'/runs/appworld-official-whitening-20261006-v2/appworld-dt'


def main():
    readiness=AUDIT/'appworld-eval-client-routing-20261005/readiness-repair-20261006/v1'
    dated=AUDIT/'textcraft-degradation-20261005/official-whitening-formal-20261006/v2/deployment/runtime-1791290776/current_runtime.json'
    paths={
        'source/prepare_appworld_whitening_readiness.py':Path(__file__).resolve(),
        'source/inspect_appworld_whitening_readiness_cpu.py':LOCAL/'inspect_appworld_whitening_readiness_cpu.py',
        **{'source/inputs/readiness-'+name:readiness/name for name in
            ('prepared.json','native-service-results.json','completed-resource-status.json','verification-review.json')},
        'source/inputs/current-runtime-1791290776.json':dated,
        'source/inputs/current-runtime-20261005-historical.json':REPO/'experiments/rl/current_runtime.json',
        'source/inputs/app-v1-prepared.json':LOCAL.parent/'v1/prepared.json',
    }
    for path in paths.values():
        if path.suffix=='.py':ast.parse(path.read_bytes(),filename=str(path))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    payload=dict(root=ROOT,receipt=OUT,v1_receipt=V1,formal_output=FORMAL_OUTPUT,provisioned_entry=ENTRY,
        repository_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        sources={name:dict(local_path=str(path),sha256=sha(path)) for name,path in paths.items()},
        v1_native_source=dict(path=V1+'/source/inspect_appworld_official_whitening_cpu.py',
            sha256=sha(LOCAL.parent/'v1/inspect_appworld_official_whitening_cpu.py')))
    inp=LOCAL/'preparation-input.json';inp.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    subprocess.run(SSH+['mkdir','-p',OUT+'/source/inputs'],check=True)
    for name,path in paths.items():subprocess.run(SCP+[str(path),f'{SSH[-1]}:{OUT}/{name}'],check=True)
    subprocess.run(SCP+[str(inp),f'{SSH[-1]}:{OUT}/preparation-input.json'],check=True)
    script=('set -e\nsource '+ENTRY+'/metax-entry.env.sh\nexport PYTHONDONTWRITEBYTECODE=1\n'
        'export CUDA_VISIBLE_DEVICES=""\nexport MACA_VISIBLE_DEVICES=""\n'
        '"$VENV_PYTHON" '+OUT+'/source/inspect_appworld_whitening_readiness_cpu.py --input '+OUT+'/preparation-input.json\n')
    (LOCAL/'prepare.sh').write_text(script,encoding='utf-8')
    result=subprocess.run(SSH+['bash','-s'],input=script.encode(),capture_output=True)
    (LOCAL/'prepare.stdout.txt').write_bytes(result.stdout+result.stderr)
    print(result.stdout.decode(errors='replace'));print(result.stderr.decode(errors='replace'))
    result.check_returncode()
    names=('prepared.json','source-binding.json','native-interface-inspection.json','native-interface-inspection.stdout.txt',
        'focused-owner-tests.json','focused-owner-tests.stdout.txt','effective-config.yaml','configuration-comparison.json','launch-plan.json')
    subprocess.run(SCP+[f'{SSH[-1]}:{OUT}/{name}' for name in names]+[str(LOCAL)],check=True)


if __name__=='__main__':main()
