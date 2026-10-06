"""Prepare original formal TextCraft configuration; never submit a job.

The frozen launcher owns options and Hydra configuration. The verified entry,
VERL copy, service, assets and cache directories are reused without patching.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, SCP, SSH

OUT = ROOT + '/receipts/textcraft-official-whitening-formal-20261006-v1'
LOCAL = AUDIT / 'textcraft-degradation-20261005/official-whitening-formal-20261006/v1'
FORMAL_OUTPUT = ROOT + '/runs/textcraft-official-whitening-20261006-v1/textcraft-dt'
CHECKPOINT = ROOT + '/runs/official-trajectory-20260930-v7/textcraft-dt/checkpoints/global_step_25'
WHITE = ROOT + '/candidates/textcraft-official-whitening-20261006-v1/verl'


def preparation_script():
    original_path = AUDIT / 'textcraft-late-sampling-audit-20261005/remote-prepared.json'
    white_path = AUDIT / 'textcraft-degradation-20261005/official-whitening-20261006/v1/candidate-source.json'
    originals = json.loads(original_path.read_text(encoding='utf-8'))
    white = json.loads(white_path.read_text(encoding='utf-8'))
    payload = dict(original=originals, white=white, checkpoint=CHECKPOINT,
        formal_output=FORMAL_OUTPUT, receipt=OUT, repository_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        preparation_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_receipts={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (original_path, white_path)})
    body = r'''
from pathlib import Path
import ast,hashlib,json,os,psutil,resource,subprocess,time
payload=json.loads(@PAYLOAD@)
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=payload['original'];white=payload['white'];root=Path(@ROOT@)
receipt=Path(payload['receipt']);receipt.mkdir(parents=True,exist_ok=True)
assert not (receipt/'prepared.json').exists(), 'Preserve the completed preparation'
entry=Path(original['entry']);owner=Path(white['candidate']);dt=Path(original['dt_root'])
assert owner==Path(@WHITE@) and Path(white['baseline'])==Path(original['verl_root'])
for name,expected in original['entry_sha256'].items():assert sha(entry/name)==expected,name
for name,expected in original['dt_source_sha256'].items():assert sha(dt/name)==expected,name
for name,expected in white['all_file_sha256'].items():
 assert sha(owner/name)==expected['candidate'],name
 assert sha(Path(white['baseline'])/name)==expected['baseline'],name
checkpoint=Path(payload['checkpoint']);files=[checkpoint/'data.pt']+[
 checkpoint/'actor'/f'{kind}_world_size_2_rank_{rank}.pt'
 for rank in range(2) for kind in ('model','optim','extra_state')]
assert all(p.is_file() and p.stat().st_size for p in files)
assert not Path(payload['formal_output']).exists(), 'This preparation does not own a prior formal run'
baseline=original['unchanged_formal_options'];agentgym=baseline['+env.textcraft']['owner_root']
data=Path(baseline['data.train_files']).parent
assert all((data/n).is_file() for n in ('train.parquet','validation.parquet','source.json'))
env=os.environ.copy();provisioned_dt_root=Path(env['DT_ROOT'])
dt_paths={str(provisioned_dt_root):str(dt),
 str(provisioned_dt_root/'experiments/rl'):str(dt/'experiments/rl')}
inherited=[dt_paths.get(p,p) for p in env['PYTHONPATH'].split(':')]
env.update(VERL_ROOT=str(owner),DT_ROOT=str(dt),DT_ENTRY_ROOT=str(entry),
 AGENTGYM_RL_ROOT=agentgym,CUDA_VISIBLE_DEVICES='4,5',**original['resource_environment'])
env.pop('MACA_VISIBLE_DEVICES',None)
env['PYTHONPATH']=':'.join([str(entry),str(owner),*inherited])
# The same original launcher is used in config-only and intended formal argv.
# Its dry output is separate, so the future run directory stays uncreated.
cpu_env=env.copy();cpu_env.update(CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='')
code=r"""
from pathlib import Path
import hashlib,json,os,resource,sys,time,torch
import launch_textcraft_native as launcher
import owner_runtime_options as runtime
import verl.trainer.ppo.ray_trainer as trainer
import verl.utils.torch_functional as functional
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
args=json.loads(sys.argv[1]);started=time.time()
formal,sampling=launcher.options_for(Path(args['data']),Path(args['formal_output']),
 resume_from=Path(args['checkpoint']))
dry,_=launcher.options_for(Path(args['data']),Path(args['dry_output']),
 resume_from=Path(args['checkpoint']))
assert not torch.cuda.is_initialized()
record=dict(options=formal,sampling=sampling,dry_options=dry,
 owner_command=runtime.owner_command(formal),cuda_initialized=torch.cuda.is_initialized(),
 pid=os.getpid(),pid_birth=__import__('psutil').Process().create_time(),
 max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
 imports={name:dict(path=module.__file__,sha256=sha(module.__file__)) for name,module in
 [('launcher',launcher),('runtime_options',runtime),('trainer',trainer),('official_helper',functional)]},
 model_initializations=0,rollout_calls=0,DT_calls=0,backward_calls=0,optimizer_steps=0,
 scope='Original options_for and actual imports only; no model or training call',wall_seconds=time.time()-started)
Path(args['inspection']).write_text(json.dumps(record,indent=2)+'\n')
"""
args=dict(data=str(data),formal_output=payload['formal_output'],checkpoint=str(checkpoint),
 dry_output=str(receipt/'config-only'),inspection=str(receipt/'native-interface-inspection.json'))
started=time.time()
with (receipt/'inspection.stdout.txt').open('wb') as log:
 subprocess.run([env['VENV_PYTHON'],'-c',code,json.dumps(args)],cwd=entry,env=cpu_env,
  stdout=log,stderr=subprocess.STDOUT,check=True)
inspection=read(Path(args['inspection']))
assert inspection['imports']['launcher']['sha256']==original['entry_sha256']['launch_textcraft_native.py']
assert inspection['imports']['runtime_options']['sha256']==original['entry_sha256']['owner_runtime_options.py']
assert inspection['imports']['trainer']['path']==str(owner/'verl/trainer/ppo/ray_trainer.py')
assert inspection['imports']['trainer']['sha256']==white['trainer_sha256']
assert inspection['imports']['official_helper']['sha256']==original['owner_sha256']['verl/utils/torch_functional.py']
options=inspection['options']
changes={k:dict(original=baseline.get(k),prepared=options.get(k))
 for k in baseline.keys()|options.keys() if baseline.get(k)!=options.get(k)}
output_keys={'trainer.default_local_dir','trainer.rollout_data_dir','trainer.validation_data_dir'}
assert set(changes)==output_keys|{'trainer.resume_from_path'},changes
assert options['trainer.resume_from_path']==str(checkpoint)
assert inspection['sampling']==original['unchanged_sampling']
argv=[env['VENV_PYTHON'],str(entry/'launch_textcraft_native.py'),'--data',str(data),
 '--output',payload['formal_output'],'--resume-from',str(checkpoint)]
dry_argv=list(argv);dry_argv[dry_argv.index('--output')+1]=args['dry_output'];dry_argv+=['--config-only']
with (receipt/'effective-config.yaml').open('wb') as log:
 subprocess.run(dry_argv,cwd=entry,env=cpu_env,stdout=log,stderr=subprocess.STDOUT,check=True)
dry_launch=read(Path(args['dry_output'])/'launch.json')
assert dry_launch['options']==inspection['dry_options']
dry_changes={k:dict(formal=options.get(k),dry=inspection['dry_options'].get(k))
 for k in options.keys()|inspection['dry_options'].keys() if options.get(k)!=inspection['dry_options'].get(k)}
assert set(dry_changes)==output_keys,dry_changes
assert not Path(payload['formal_output']).exists()
service_path=Path(@ENTRY@)/'textcraft-service.json';service=read(service_path)
service_pid=service['pid'];service_process=psutil.Process(service_pid)
service_identity=dict(pid=service_pid,pid_birth=service_process.create_time(),
 path=str(service_path),sha256=sha(service_path),status=service_process.status(),
 scope='PID/source identity only; no environment creation, service request, restart or cleanup')
launch_plan=dict(argv=argv,working_directory=payload['formal_output'],
 environment={k:env[k] for k in sorted(env) if k.startswith(('DT_','VERL_','FLA_')) or k in
 ('CUDA_VISIBLE_DEVICES','PYTHONPATH','AGENTGYM_RL_ROOT','VENV_PYTHON','TORCHINDUCTOR_CACHE_DIR',
 'TRITON_CACHE_DIR','PYTORCH_CUDA_ALLOC_CONF','OMP_NUM_THREADS','MKL_NUM_THREADS','MACA_TORCH_COMPILE_CONF')},
 environment_source=dict(path=@ENTRY@+'/metax-entry.env.sh',sha256=sha(Path(@ENTRY@)/'metax-entry.env.sh')),
 environment_scope='Source provisioned entry environment, then existing submission path ordering and frozen resource override; not inherited from current diagnostic')
(receipt/'launch-plan.json').write_text(json.dumps(launch_plan,indent=2)+'\n')
record=dict(prepared_unix=time.time(),status='prepared_configuration_only_not_launched',
 upstream_commit=white['upstream_commit'],white_patch_commit='8f52b95658d37fc7a0cb69647de801188f70be02',
 preparation_source_sha256=payload['preparation_source_sha256'],repository_commit=payload['repository_commit'],
 input_receipts=payload['input_receipts'],entry=str(entry),verl_root=str(owner),dt_root=str(dt),
 entry_sha256=original['entry_sha256'],dt_source_sha256=original['dt_source_sha256'],
 owner_source_sha256={name:pair['candidate'] for name,pair in white['all_file_sha256'].items()},
 checkpoint_root=str(checkpoint),actor_checkpoint=str(checkpoint/'actor'),
 checkpoint_files={str(p.relative_to(checkpoint)):p.stat().st_size for p in files},
 data_loader_state_sha256=sha(checkpoint/'data.pt'),configuration_changes=changes,
 dry_vs_formal_output_only_changes=dry_changes,sampling=inspection['sampling'],
 formal_output=payload['formal_output'],formal_output_created=False,service=service_identity,
 intended_entry='Frozen original launcher -> original main_ppo -> native RayPPOTrainer.fit actor_rollout owner',
 old_submit_helper_scope='Not called. Its latest-100 checkpoint guard is not a constraint of native resume_path loader; explicit completed checkpoint25 requested.',
 phase_wall_seconds=time.time()-started,process=dict(pid=os.getpid(),pid_birth=psutil.Process().create_time()),
 no_launch=True,model_initializations=0,optimizer_steps=0,DT_calls=0,rollout_calls=0,
 receipt_files={n:dict(path=str(receipt/n),sha256=sha(receipt/n)) for n in
 ('native-interface-inspection.json','inspection.stdout.txt','effective-config.yaml','launch-plan.json')},
 scope='Source/configuration preparation only. Existing bounded actual-update result remains separate; no formal job or active manifest mutation.')
(receipt/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(status=record['status'],checkpoint=str(checkpoint),
 entry=str(entry),verl_root=str(owner),changes=changes,phase_wall_seconds=record['phase_wall_seconds'])),flush=True)
'''
    body = (body.replace('@PAYLOAD@', repr(json.dumps(payload))).replace('@ROOT@', repr(ROOT))
            .replace('@WHITE@', repr(WHITE)).replace('@ENTRY@', repr(ENTRY)))
    ast.parse(body)
    return 'set -e\nsource ' + ENTRY + '/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n' + body + '\nPY\n'


if __name__ == '__main__':
    LOCAL.mkdir(parents=True, exist_ok=True)
    script = preparation_script()
    (LOCAL / 'prepare.sh').write_text(script, encoding='utf-8')
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
    (LOCAL / 'prepare.stdout.txt').write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
    subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/{name}' for name in
        ('prepared.json','native-interface-inspection.json','inspection.stdout.txt',
         'effective-config.yaml','launch-plan.json')] + [str(LOCAL)], check=True)
