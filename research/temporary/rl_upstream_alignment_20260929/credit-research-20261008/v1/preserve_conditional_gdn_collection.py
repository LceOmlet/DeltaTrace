"""Copy exact completed diagnostic vectors and phase logs, verifying their SHA."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import SCP,SSH,ENTRY


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=('textcraft','appworld'),required=True)
    parser.add_argument('--candidate-kind',choices=('conditional_gdn','conditional_mixers'),default='conditional_gdn')
    args=parser.parse_args()
    folder=HERE/(args.candidate_kind.replace('_','-')+'-collection-v1')
    records=[json.loads((folder/(args.task+'-rank'+str(i)+'.json')).read_bytes()) for i in range(2)]
    assert all(r['phase']=='complete' for r in records)
    files=[]
    for record in records:
        for batch in record['batches']:
            files.extend(batch['variants'].values())
    paths={item['artifact']:item['sha256'] for item in files}
    assert len(paths)==12
    destination=folder/(args.task+'-vectors');destination.mkdir(exist_ok=True)
    missing=[p for p,h in paths.items() if not (destination/Path(p).name).exists()
             or sha(destination/Path(p).name)!=h]
    if missing:
        subprocess.run(SCP+[SSH[-1]+':'+p for p in missing]+[str(destination)],check=True,timeout=60)
    manifest=[]
    for path,digest in paths.items():
        local=destination/Path(path).name
        assert sha(local)==digest
        manifest.append(dict(remote=path,path=str(local.resolve()),sha256=digest,bytes=local.stat().st_size))
    launch=json.loads((folder/(args.task+'-launch.json')).read_bytes())
    remote=str(Path(launch['argv'][1]).parent).replace('\\','/')
    names=['driver.log']+['results/rank'+str(i)+'-phases.jsonl' for i in range(2)]
    names += ['results/'+Path(r['candidate']['phase_log']).name for r in records]
    shell='source '+ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'
    shell+='import hashlib,json\nfrom pathlib import Path\nout=Path('+repr(remote)+')\n'
    shell+='print(json.dumps([{\'remote\':str(out/n),\'sha256\':hashlib.sha256((out/n).read_bytes()).hexdigest(),\'bytes\':(out/n).stat().st_size} for n in '+repr(names)+']))\nPY\n'
    result=subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,check=True,timeout=25)
    logs=json.loads(result.stdout)
    subprocess.run(SCP+[SSH[-1]+':'+item['remote'] for item in logs]+[str(destination)],check=True,timeout=45)
    for item in logs:
        local=destination/Path(item['remote']).name
        assert sha(local)==item['sha256']
        manifest.append(dict(item,path=str(local.resolve())))
    receipt=dict(scope=__doc__,task=args.task,files=manifest,all_SHA256_verified=True,
        bytes=sum(f['bytes'] for f in manifest),
        identity='Actual selected token IDs, source positions, complete signed d vectors, UIDs, detail and exact phase logs; original target/case artifacts stay at their frozen manifest paths and SHA.',
        all_debug_information_local=False,
        exclusion='This is the new small diagnostic artifact copy, not a new claim that all historical native/model captures have been copied locally.')
    (folder/(args.task+'-preserved.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(task=args.task,files=len(manifest),bytes=receipt['bytes'],SHA256_verified=True)))


if __name__=='__main__':main()
