"""Prepare/launch one bounded owner gradient study; never release formal training."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
sys.path.insert(0, str(AUDIT))
from stage_environment_entry import ROOT, ENTRY, SSH, SCP

REMOTE = ROOT + '/receipts/credit-collection-gradients-20261008-v1'
FILES = [HERE / 'inspect_collection_gradients.py',
         AUDIT / 'observe_native_optimizer_minibatch.py',
         AUDIT / 'observe_native_actor_loss_gradients.py',
         AUDIT / 'direct-target-extreme-token-endpoint-20261007/v1/inspect_extreme_endpoint.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    global REMOTE, FILES
    error_mode = '--error-gradients' in sys.argv
    if error_mode:
        REMOTE = ROOT + '/receipts/credit-collection-error-gradients-20261008-v1'
        FILES = FILES + [HERE / 'collection-coefficients.json']
    manifest = json.loads((HERE / 'manifest.json').read_bytes())
    points = json.loads((HERE / 'author-collection-observations/rank0.json').read_bytes())['tail_results']
    development = {uid: group['initial_state_sha256'] for group in manifest['tasks']['textcraft']['groups']
                   if group['split'] == 'development' for uid in group['trajectory_uids']}
    assert len(points) == 37 and sum(p['native_single_d'] >= 0 for p in points) == 24
    assert all(development[p['traj_uid']] == p['initial_state_sha256'] for p in points)
    hashes = {p.name: sha(p) for p in FILES}
    assert hashes['observe_native_actor_loss_gradients.py'] == '94219328ba644a5c0118f4a3bd2561b16f969643f2cd2915047202a7ff085047'
    assert hashes['inspect_extreme_endpoint.py'] == '8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5'
    plan = dict(scope='Actual saved coefficient contributions, not output correction, new credit or corrected-gradient estimation.',
                tail_points=points, frozen_manifest_sha256=sha(HERE / 'manifest.json'),
                tail_observations_sha256=sha(HERE / 'author-collection-observations/rank0.json'),
                diagnostic_code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                observer_extension='Optional named PG coefficient views only; original VERL loss/forward/backward/reductions unchanged.',
                original_observer_baseline_sha256='022466b2bac94617b8e627ea027fb3afdf4490dc4bc2bcaa11944ab444e36214',
                scripts=hashes, original_minibatches=4, planned_native_backward_passes_per_rank=12,
                actual_microbatch=4, lora_rank=8, lora_alpha=16, wall_budget_seconds=1800,
                training_release=False, optimizer_steps=0, scheduler_steps=0, DT=0,
                rollout=0, checkpoint_restore=0, new_GDN_candidate=False)
    if error_mode:
        plan.update(scope='Four original optimizer minibatches at unchanged base LoRA. Full saved PG plus separate fixed-original-scale native-minus-DT errors: predicted-tail development census, bounded uniform sample, and native tails missed by DT. No inverse-inclusion weighting, no extrapolation of sampled errors to the whole bulk, no new whitening, corrected training or GDN candidate.',
                    coefficient_error_receipt=dict(path=REMOTE + '/collection-coefficients.json',
                                                   sha256=sha(HERE / 'collection-coefficients.json')),
                    planned_native_backward_passes_per_rank=13, wall_budget_seconds=2700)
    subprocess.run(SSH + ['bash', '-s'], input=('set -eu\nmkdir -p ' + REMOTE + '\n').encode(),
                   check=True, timeout=30)
    subprocess.run(SCP + [*(str(path) for path in FILES), SSH[-1] + ':' + REMOTE + '/'],
                   check=True, timeout=90)
    body = r'''
import ast,hashlib,json,os,psutil,re,subprocess,sys,time
from pathlib import Path
root=Path(ROOT);out=Path(OUT);plan=PLAN
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (out/'launch.json').exists(),'Do not duplicate or replace this diagnostic launch'
assert psutil.Process(2833207).create_time()==1791370325.16
cap=root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
assert not any((cap/f'rank{rank}-release-update').exists() for rank in (0,1))
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
assert not re.search(r'^\|\s*[45]\s+\d+\s+\S',physical,re.M),'Physical4/5 occupied'
for name,h in plan['scripts'].items():
 assert sha(out/name)==h
 ast.parse((out/name).read_text())
if LAUNCH:
 prepared_plan=json.loads((out/'gradient-inputs.json').read_bytes())
 # Commit metadata may advance after preparation; the measured inputs and
 # imported scripts must remain exactly those already bound by inspection.
 assert all(prepared_plan[k]==plan[k] for k in plan if k!='diagnostic_code_commit')
else:
 (out/'gradient-inputs.json').write_text(json.dumps(plan,indent=2)+'\n')
p=root/'runs/direct-target-prefix-runtime-20261007-v1/textcraft/textcraft-dt/source.json'
assert sha(p)=='2796233e2683f1939896c74b2b578c242dbd7a7f235b9ef61cbedd398f61be52'
source=json.loads(p.read_bytes())
env=dict(os.environ,**source['environment']);env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env['CUDA_VISIBLE_DEVICES']='4,5';env['DT_TASK']=source['startup_options']['env.env_name'];env['DT_MAX_STEPS']=str(source['startup_options']['env.max_steps'])
if 'coefficient_error_receipt' in plan:env['DT_COLLECTION_DIAGNOSTIC_OUT']=str(out)
dt=Path(env['DT_ROOT']);q=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
env['PYTHONPATH']=':'.join([str(out),str(dt),env.get('DT_OFFICIAL_ROOT') or q['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],q['ft_extension_root']])
argv=[env['VENV_PYTHON'],str(out/'inspect_collection_gradients.py')]
if not LAUNCH:
 prepared_env=dict(env,CUDA_VISIBLE_DEVICES='-1')
 prepared=subprocess.run(argv+['--inspect-only'],cwd=out,env=prepared_env,capture_output=True,timeout=120)
 (out/'prepare.stdout.txt').write_bytes(prepared.stdout);(out/'prepare.stderr.txt').write_bytes(prepared.stderr)
 if prepared.returncode:print(prepared.stderr.decode(errors='replace'));prepared.check_returncode()
 receipt={'status':'prepared-only actual owner input mapping; no model/backward', 'inspection':json.loads((out/'input-inspection.json').read_bytes()),'scripts':plan['scripts'],'code_commit':plan['diagnostic_code_commit'],'physical':physical,'host_available':psutil.virtual_memory().available}
 print(json.dumps(receipt));sys.exit(0)
inspection=json.loads((out/'input-inspection.json').read_bytes())
assert inspection['input_plan_sha256']==sha(out/'gradient-inputs.json')
assert inspection['observer']['sha256']==plan['scripts']['observe_native_optimizer_minibatch.py']
assert inspection['tail_points']==37 and inspection['native_sign_flip_points']==24 and inspection['rows']==256
with (out/'driver.log').open('xb') as stream:
 process=subprocess.Popen(argv,cwd=out,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'pid':process.pid,'birth':psutil.Process(process.pid).create_time(),'launched_unix':time.time(),'devices':[4,5],'argv':argv,'code_commit':plan['diagnostic_code_commit'],'scripts':plan['scripts'],'input_plan_sha256':sha(out/'gradient-inputs.json'),'source_path':str(p),'source_sha256':sha(p),'wall_budget_seconds':plan['wall_budget_seconds'],'planned_native_backward_passes_per_rank':plan['planned_native_backward_passes_per_rank'],'scope':inspection['scope'],'host_available':psutil.virtual_memory().available,'training_release':False,'optimizer_steps':0,'scheduler_steps':0,'DT':0,'rollout':0,'checkpoint_restore':0}
(out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');(out/'before-physical.txt').write_text(physical);print(json.dumps(receipt))
'''
    body = 'ROOT=' + repr(ROOT) + '\nOUT=' + repr(REMOTE) + '\nPLAN=' + repr(plan) + '\nLAUNCH=' + repr('--launch' in sys.argv) + '\n' + body
    shell = 'set -eu\nsource ' + ENTRY + '/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n' + body + '\nPY\n'
    mode = 'launch' if '--launch' in sys.argv else 'prepare'
    prefix = 'collection-error-gradient' if error_mode else 'collection-gradient'
    (HERE / f'{prefix}-{mode}-command.sh').write_text(shell, encoding='utf-8', newline='\n')
    run = subprocess.run(SSH + ['bash', '-s'], input=shell.encode(), capture_output=True, timeout=160)
    (HERE / f'{prefix}-{mode}.stderr.txt').write_bytes(run.stderr)
    if run.returncode:
        print(run.stdout.decode(errors='replace')[-6000:]); print(run.stderr.decode(errors='replace')[-3000:])
        run.check_returncode()
    result = json.loads(run.stdout)
    (HERE / f'{prefix}-{mode}.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inspection','physical','argv')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
