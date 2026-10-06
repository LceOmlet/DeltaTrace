"""Submit the frozen original TextCraft launcher from explicit checkpoint25.

Only the existing process boundary and authoritative job/source schema are
used. This helper does not implement training, stopping, retries or recovery.
Importing it and generating its script perform no remote operation.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, REPO, ROOT, SCP, SSH

PREPARED = ROOT + '/receipts/textcraft-official-whitening-formal-20261006-v2/prepared.json'
PLAN = ROOT + '/receipts/textcraft-official-whitening-formal-20261006-v2/launch-plan.json'
LOCAL = AUDIT / 'textcraft-degradation-20261005/official-whitening-formal-20261006/v2/deployment'
PREPARED_SHA = 'ee3fb085bd5f26f218ee9d6a9b372dc07cdf4a292cd0440e2d3a590cd0420e81'
PLAN_SHA = 'de263511adf4cf469603d1a5f4047232a704282d93c87e14aeea1176f1ed1361'


def submission_script():
    local_root = LOCAL.parent
    for name, expected in [('prepared.json', PREPARED_SHA), ('launch-plan.json', PLAN_SHA)]:
        assert hashlib.sha256((local_root / name).read_bytes()).hexdigest() == expected, name
    plan = json.loads((local_root / 'launch-plan.json').read_text(encoding='utf-8'))
    proofs = {}
    for name, expected in [
        ('textcraft-native25-completed-stop.json', '2a992aed6d3ae081e93ffb81c3f8b0b313085858e8801fa062dd1e8d8c753f9b'),
        ('textcraft-stop-current-completed-stop.json', 'c033a963472eb312af88cb7bb58d733f1f3767cd627ce30d7ab0e0b46a82f279')]:
        path = AUDIT / 'checkpoint-boundary-20261002' / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
        proofs[name] = dict(path=str(path.relative_to(REPO)), sha256=expected,
                           original_record=json.loads(path.read_text(encoding='utf-8')))
    payload = dict(proofs=proofs, repository_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        submission_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        preparation_sha256=PREPARED_SHA, launch_plan_sha256=PLAN_SHA)
    code = r'''
from pathlib import Path
import hashlib,json,os,psutil,re,subprocess,time
payload=json.loads(@PAYLOAD@);root=Path(@ROOT@)
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prepared_path=Path(@PREPARED@);plan_path=Path(@PLAN@)
assert sha(prepared_path)==payload['preparation_sha256']
assert sha(plan_path)==payload['launch_plan_sha256']
prepared=read(prepared_path);plan=read(plan_path)
assert sha(Path(plan['environment_source']['path']))==plan['environment_source']['sha256']
entry=Path(prepared['entry']);owner=Path(prepared['verl_root']);dt=Path(prepared['dt_root'])
for folder,values in [(entry,prepared['entry_sha256']),
 (owner,prepared['owner_source_sha256']),(dt,prepared['dt_source_sha256'])]:
 for name,expected in values.items():assert sha(folder/name)==expected,(str(folder/name),expected)
assert prepared['entry_sha256']['reward_readout.py']=='94a7afbc09da72b62572d31fd32a6534f6e8f3daf656fce1011cdfa68b3c3e2b'
assert prepared['owner_source_sha256']['verl/trainer/ppo/ray_trainer.py']=='7366557b482e604d66f47bdc4841ea147ffaac7c80538fa000544bb4e92eb619'
inspection_record=prepared['receipt_files']['native-interface-inspection.json']
inspection_path=Path(inspection_record['path']);assert sha(inspection_path)==inspection_record['sha256']
inspection=read(inspection_path)
config_record=prepared['receipt_files']['effective-config.yaml']
assert sha(Path(config_record['path']))==config_record['sha256']
checkpoint=Path(prepared['checkpoint_root']);proof=payload['proofs']['textcraft-native25-completed-stop.json']['original_record']
assert str(checkpoint)==proof['checkpoint'] and proof['marker_step']==25
marker=checkpoint.parent/'latest_checkpointed_iteration.txt'
assert marker.read_text().strip()=='25' and sha(marker)==proof['marker_sha256']
assert checkpoint==marker.parent/'global_step_25'
for name,size in proof['checkpoint_files'].items():assert (checkpoint/name).stat().st_size==size,name
assert sha(checkpoint/'data.pt')==prepared['data_loader_state_sha256']
active=read(root/'active-training.json');sources=read(root/'active-source.json')
old=next(j for j in active['jobs'] if j['task']=='TextCraft')
stop=payload['proofs']['textcraft-stop-current-completed-stop.json']['original_record']
assert old['pid']==stop['prior_driver_pid']==3218909
assert old['observed_process_created_unix']==stop['prior_created_unix']
assert not stop['remaining_non_zombie']
old_observation=dict(pid=old['pid'],pid_exists=psutil.pid_exists(old['pid']))
if old_observation['pid_exists']:
 process=psutil.Process(old['pid'])
 old_observation.update(pid_birth=process.create_time(),status=process.status())
 assert process.create_time()==old['observed_process_created_unix']
 assert process.status()==psutil.STATUS_ZOMBIE, 'Old formal job still running; no stop performed'
assert any(item['task']=='TextCraft' for item in sources['jobs'])
prior_source_path=Path(old['source_receipt']);assert sha(prior_source_path)==stop['current_source']['sha256']
prior_source=read(prior_source_path)
argv=plan['argv'];output=Path(plan['working_directory']);base=output.parent
assert str(output)==prepared['formal_output'] and not base.exists(), 'Preserve any prior submission'
assert argv[1]==str(entry/'launch_textcraft_native.py') and '--config-only' not in argv
assert argv[argv.index('--output')+1]==str(output)
assert argv[argv.index('--resume-from')+1]==str(checkpoint)
env=os.environ.copy();env.update(plan['environment']);env.pop('MACA_VISIBLE_DEVICES',None)
assert env['CUDA_VISIBLE_DEVICES']=='4,5'
assert env['VERL_ROOT']==str(owner) and env['DT_ENTRY_ROOT']==str(entry) and env['DT_ROOT']==str(dt)
state=subprocess.check_output(['mx-smi'],text=True)
for device in [4,5]:assert not re.search(r'\|\s*'+str(device)+r'\s+\d+\s+\S',state),state
# Existing provisioned service is observed, never reset or restarted here.
service=prepared['service'];service_process=psutil.Process(service['pid'])
assert service_process.create_time()==service['pid_birth']
assert service_process.status()!=psutil.STATUS_ZOMBIE
assert sha(Path(service['path']))==service['sha256']
output.mkdir(parents=True)
for name in ('active-training.json','active-source.json'):
 (base/('prior-'+name)).write_bytes((root/name).read_bytes())
(base/'physical-before.txt').write_text(state)
source=dict(unix=time.time(),dt_root=str(dt),verl_root=str(owner),agentgym_root=env['AGENTGYM_RL_ROOT'],
 lora_rank=8,lora_alpha=16,entry_sha256=prepared['entry_sha256'],
 verl_sha256={name:sha(owner/name) for name in prior_source['verl_sha256']},
 owner_head_sha256=prepared['owner_source_sha256'],dt_source_sha256=prepared['dt_source_sha256'],
 prepared_receipt=str(prepared_path),prepared_receipt_sha256=sha(prepared_path),
 launch_plan_receipt=str(plan_path),launch_plan_sha256=sha(plan_path),
 submission_repository_commit=payload['repository_commit'],submission_script_sha256=payload['submission_script_sha256'],
 upstream_commit=prepared['upstream_commit'],white_patch_commit=prepared['white_patch_commit'],
 semantic_repair=prepared['semantic_repair'],prior_driver_pid=old['pid'],
 prior_source_receipt=str(prior_source_path),prior_source_sha256=sha(prior_source_path),
 resume_from=str(checkpoint),completed_checkpoint_marker=dict(path=str(marker),value=25,sha256=sha(marker)),
 completed_checkpoint_proofs=payload['proofs'],checkpoint_files=proof['checkpoint_files'],
 data_loader_state_sha256=prepared['data_loader_state_sha256'],pythonpath=env['PYTHONPATH'],
 resume_launcher=dict(path=str(entry/'launch_textcraft_native.py'),sha256=sha(entry/'launch_textcraft_native.py')),
 environment_source=plan['environment_source'],environment=plan['environment'],
 startup_options=inspection['options'],sampling=inspection['sampling'],
 effective_config_preparation=config_record,actual_CPU_imports=inspection['imports'],
 import_scope='CPU-prepared actual imports; new formal worker imports and completed updates are not yet observed',
 prior_driver_observation=old_observation,service_observation=dict(pid=service_process.pid,
 pid_birth=service_process.create_time(),status=service_process.status()),
 physical_before=dict(path=str(base/'physical-before.txt'),sha256=sha(base/'physical-before.txt')),
 host_available_bytes=psutil.virtual_memory().available,
 scope='Explicit complete checkpoint25 restart through frozen original launcher, terminal-reward fix, official whitening and accepted query-clock source. No new training lifecycle or task parameters.')
for name in ('actor_fix_commit','dt_dispatch_commit','actor_padding_sha256',
 'padding_comparison_receipt','padding_comparison_receipt_sha256','rollout_scope_commit',
 'rollout_scope_comparison','rollout_scope_comparison_sha256','completed_overlay_receipt','completed_overlay_sha256'):
 if name in prior_source:source[name]=prior_source[name]
(output/'source.json').write_text(json.dumps(source,indent=2)+'\n')
with (output/'train.log').open('wb') as log:
 proc=subprocess.Popen(argv,env=env,cwd=str(output),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
job=dict(task='TextCraft',method='dt',pid=proc.pid,devices=[4,5],started_unix=time.time(),
 observed_process_created_unix=psutil.Process(proc.pid).create_time(),argv=argv,
 entry=str(entry),verl_root=str(owner),dt_root=str(dt),output=str(output),
 log=str(output/'train.log'),checkpoints=str(output/'checkpoints'),source_receipt=str(output/'source.json'),
 status='formal_resume_submitted_not_yet_verified',lora_rank=8,lora_alpha=16,
 actor_microbatch=4,log_prob_micro_batch_size_per_gpu=4,budget=old['budget'],
 resume_from=str(checkpoint),resume_completed_step=25,prior_driver_pid=old['pid'])
(output/'job.json').write_text(json.dumps(job,indent=2)+'\n')
manifest_path=base/'formal-training.json'
manifest=dict(active,manifest=str(manifest_path),jobs=[job if j['task']=='TextCraft' else j for j in active['jobs']])
manifest.setdefault('retired_jobs',[]).append(dict(old,
 status='replaced_from_explicit_completed_checkpoint25',replacement_pid=proc.pid))
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
(root/'active-training.json').write_text(json.dumps(manifest,indent=2)+'\n')
sources.update(unix=time.time(),manifest=str(manifest_path))
sources['jobs']=[dict(task=job['task'],pid=job['pid'],entry=job['entry'],verl_root=job['verl_root'],
 source_receipt=job['source_receipt'],runtime_override=None,actor_microbatch=4,lora_rank=8,lora_alpha=16)
 if item['task']=='TextCraft' else item for item in sources['jobs']]
(root/'active-source.json').write_text(json.dumps(sources,indent=2)+'\n')
print(json.dumps(job,indent=2),flush=True)
'''
    code = (code.replace('@PAYLOAD@', repr(json.dumps(payload))).replace('@ROOT@', repr(ROOT))
            .replace('@PREPARED@', repr(PREPARED)).replace('@PLAN@', repr(PLAN)))
    ast.parse(code)
    return ('set -e\nsource ' + plan['environment_source']['path']
            + '\n"$VENV_PYTHON" - <<\'PY\'\n' + code + '\nPY\n')


if __name__ == '__main__':
    LOCAL.mkdir(parents=True, exist_ok=True)
    assert not (LOCAL / 'launch.stdout.txt').exists(), 'Inspect an existing submission instead of repeating it'
    script = submission_script()
    (LOCAL / 'submission.sh').write_text(script, encoding='utf-8')
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
    (LOCAL / 'launch.stdout.txt').write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
    output = json.loads((LOCAL.parent / 'launch-plan.json').read_text(encoding='utf-8'))['working_directory']
    base = output.rsplit('/', 1)[0]
    subprocess.run(SCP + [f'{SSH[-1]}:{path}' for path in
        (output+'/job.json',output+'/source.json',base+'/formal-training.json',base+'/physical-before.txt')]
        + [str(LOCAL)],check=True)
