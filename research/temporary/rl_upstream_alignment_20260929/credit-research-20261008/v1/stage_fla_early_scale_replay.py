"""Wire one original B4 replay to a SHA-bound isolated GDN owner.

The original initializer, runner, compiler, producer and saved input plan own
all model work. This only transfers source and supplies its diagnostic seam.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0,str(HERE.parents[1]))
from stage_environment_entry import ENTRY,ROOT,SSH,SCP


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=('textcraft','appworld'))
    args = parser.parse_args()
    local = HERE/'fla-early-scale-review-v1'/('replay-'+args.task)
    local.mkdir(exist_ok=False)
    out = ROOT+'/receipts/fla-early-output-scale-20261009-v1/replay-'+args.task
    prepared = json.loads((HERE/'fla-early-scale-review-v1/prepared.json').read_bytes())
    owner = prepared['owners'][args.task]
    candidate = Path(owner['candidate_path'])
    assert sha(candidate)==owner['candidate_sha256']
    binding = dict(owners={args.task:dict(owner,candidate_path=out+'/diagnostic_gdn_owner.py')})
    (local/'candidate-binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    files = {name:HERE/name for name in ('inspect_single_background_collection.py','layer-collection-inputs.json','passive_nonfinite.py')}
    files.update({'inspect_action_curve.py':HERE.parents[1]/'direct-target-action-author-curve-20261007/v1/inspect_action_curve.py',
        'inspect_extreme_endpoint.py':HERE.parents[1]/'direct-target-extreme-token-endpoint-20261007/v1/inspect_extreme_endpoint.py',
        'diagnostic_gdn_owner.py':candidate,'candidate-binding.json':local/'candidate-binding.json'})
    assert sha(files['inspect_action_curve.py'])=='7277fade4e9b1cb49f825e4cb2fc54453c9ff3ce4859b20350aa2bd67ffaf3a6'
    assert sha(files['inspect_extreme_endpoint.py'])=='8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5'
    hashes = {name:sha(path) for name,path in files.items()}
    commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    with tarfile.open(local/'source.tar','w') as tar:
        for name,path in files.items():
            tar.add(path,arcname=name)
    subprocess.run(SSH+['mkdir','-p',out],check=True,timeout=40)
    subprocess.run(SCP+[str(local/'source.tar'),SSH[-1]+':'+out+'/source.tar'],check=True,timeout=40)
    code = '''import ast,hashlib,json,os,psutil,re,subprocess,time
from pathlib import Path
out=Path(%r);root=Path(%r);task=%r;hashes=%r
assert not (out/'launch.json').exists(),'Do not duplicate this replay'
for name,digest in hashes.items():
 p=out/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
 if name.endswith('.py'):ast.parse(p.read_bytes())
physical=subprocess.check_output(['mx-smi'],text=True)
assert not re.search(r'^\\|\\s*[45]\\s+\\d+\\s+\\S',physical,re.M),'Replay GPUs occupied'
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1'/task/(task+'-dt')/'source.json'
s=json.loads(source_path.read_bytes());env=dict(os.environ,**s['environment'])
env.pop('MACA_VISIBLE_DEVICES',None);env.pop('RAY_ADDRESS',None);env.pop('DT_CAPTURE_FLA_PRECAST',None)
env['CUDA_VISIBLE_DEVICES']='4,5';env['DT_TASK']=s['startup_options']['env.env_name']
env['DT_MAX_STEPS']=str(s['startup_options']['env.max_steps'])
env['DT_SINGLE_BACKGROUND_NONFINITE_REPLAY']='1'
env['DT_DIAGNOSTIC_GDN_OWNER_MANIFEST']=str(out/'candidate-binding.json')
dt=Path(s['dt_root']);qwen=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
official=env.get('DT_OFFICIAL_ROOT') or qwen['official_root']
env['PYTHONPATH']=':'.join([str(out),str(dt),official,str(dt/'clean/qwen35'),s['pythonpath'],qwen['ft_extension_root']])
argv=[env['VENV_PYTHON'],str(out/'inspect_single_background_collection.py'),'--source',str(source_path),'--output',str(out/'results'),'--case',task]
if task=='textcraft':
 evidence=root/'receipts/direct-target-textcraft-author-curve-20261008-v1/textcraft-taskrunner-resolved-training-steps.json'
 assert hashlib.sha256(evidence.read_bytes()).hexdigest()=='57874a6f4491da68e5001fdf4f71a9d787b2dd54ec91a62b7216f1caa58da24d'
 argv+=['--owner-total-training-steps','330','--owner-total-steps-evidence',str(evidence)]
with (out/'driver.log').open('xb') as log:p=subprocess.Popen(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
r=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),unix=time.time(),source_commit=%r,files=hashes,argv=argv,devices=[4,5],physical_before=physical,DT_B4_calls_per_rank=1,optimizer=0,rollout=0,checkpoint_restore=0,production_changed=False)
(out/'launch.json').write_text(json.dumps(r,indent=2)+'\\n');print(json.dumps(r))
'''%(out,ROOT,args.task,hashes,commit)
    shell = 'set -eu\nsource '+ENTRY+'/metax-entry.env.sh\ntar -xf '+out+'/source.tar -C '+out+'\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
    (local/'launch-command.sh').write_text(shell,encoding='utf-8',newline='\n')
    run = subprocess.run(SSH+['bash','-s'],input=shell.encode(),capture_output=True,timeout=45)
    (local/'launch.stdout').write_bytes(run.stdout)
    (local/'launch.stderr').write_bytes(run.stderr)
    run.check_returncode()
    (local/'launch.json').write_bytes(run.stdout)
    print(run.stdout.decode())


if __name__=='__main__':
    main()
