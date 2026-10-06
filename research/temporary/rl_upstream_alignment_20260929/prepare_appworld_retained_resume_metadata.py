"""Compose immutable resume metadata; do not stop, launch, or test a job."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, SCP, SSH


FULL = ROOT + '/receipts/appworld-official-whitening-20261006-v2/prepared.json'
EXPECTED = {
    'full': '83393ed0081b2c88d0fae2b02c3503886df8cd8dac2178d173cf5bc9eaff3f4f',
    'source': '2827c5ae8785c498785b639573769d7db0208d3e3540087a4a0816b6968a34ac',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--efficiency-receipt', required=True)
    parser.add_argument('--efficiency-sha256', required=True)
    parser.add_argument('--entry', required=True)
    parser.add_argument('--receipt-name', required=True,
                        help='New immutable receipt directory under appworld-efficiency-20261006.')
    args = parser.parse_args()
    assert Path(args.receipt_name).name == args.receipt_name
    assert len(args.efficiency_sha256) == 64 and all(c in '0123456789abcdef' for c in args.efficiency_sha256)
    local = AUDIT / 'appworld-efficiency-20261006' / args.receipt_name
    remote = ROOT + '/receipts/appworld-efficiency-20261006/' + args.receipt_name
    expected = dict(EXPECTED, efficiency=args.efficiency_sha256)
    source_bytes = Path(__file__).read_bytes()
    script_sha = hashlib.sha256(source_bytes).hexdigest()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    code = r'''
import base64,copy,hashlib,json,psutil,sys,time
from pathlib import Path
root=Path(@ROOT@);out=Path(@REMOTE@)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
started=time.time()
active_path=root/'active-training.json';active_bytes=active_path.read_bytes()
active=json.loads(active_bytes)
job=next(j for j in active['jobs'] if j['task']=='AppWorld')
assert job['pid']==3592468 and job['observed_process_created_unix']==1791291997.14
assert job['devices']==[2,3]
driver=psutil.Process(job['pid']);assert driver.create_time()==1791291997.14
assert driver.status()!=psutil.STATUS_ZOMBIE
full_path=Path(@FULL@);eff_path=Path(@EFFICIENCY@);source_path=Path(job['source_receipt'])
for label,path in [('full',full_path),('efficiency',eff_path),('source',source_path)]:
 assert sha(path)==@EXPECTED@[label],label
full=read(full_path);eff=read(eff_path);source=read(source_path)
assert eff['entry']==@CANDIDATE_ENTRY@
assert eff['status']=='prepared_only_CPU_interface_tests_passed' and not eff['deployment']
assert eff['cpu_tests']['returncode']==0
assert eff['driver_pid']==job['pid'] and eff['driver_birth']==driver.create_time()
assert eff['source_receipt']==str(source_path) and eff['source_receipt_sha256']==sha(source_path)
assert eff['prior_entry']==job['entry']==full['entry']
assert eff['verl_root']==job['verl_root']==full['verl_root']==source['verl_root']
assert eff['dt_root']==job['dt_root']==full['dt_root']==source['dt_root']
assert full['loop_root']==source['loop_root']
assert full['resource_environment']==source['resource_environment']=={'VERL_RELEASE_UNUSED_HOST_CACHE':'1'}
assert full['entry_sha256']==source['entry_sha256']
owners=full.get('owner_sha256',full.get('owner_head_sha256'))
assert owners==source['owner_head_sha256']
assert full['author_sha256']==source['author_sha256']
assert source['lora_rank']==job['lora_rank']==8 and source['lora_alpha']==job['lora_alpha']==16
assert job['actor_microbatch']==job['log_prob_micro_batch_size_per_gpu']==4
assert source['verl_sha256']['verl/trainer/ppo/ray_trainer.py']=='d35ddd26b7497b153cec22f22e92eb1a08ef70385428dcbdcc37e879d9f16a4f'
assert source['entry_sha256']['reward_readout.py']=='94a7afbc09da72b62572d31fd32a6534f6e8f3daf656fce1011cdfa68b3c3e2b'
entry_hashes=dict(source['entry_sha256'])
assert set(eff['changes'])=={'owner_trajectory_batch.py','dt_training_batch.py'}
for name,change in eff['changes'].items():
 assert entry_hashes[name]==change['before_sha256'],name
 entry_hashes[name]=change['after_sha256']
assert eff['unchanged_entry_sha256']=={n:h for n,h in entry_hashes.items() if n not in eff['changes']}
for directory,files in [(job['entry'],source['entry_sha256']), (eff['entry'],entry_hashes),
 (job['verl_root'],owners),(job['verl_root'],source['verl_sha256']),
 (job['dt_root'],full['dt_source_sha256']),(full['loop_root'],source['author_sha256'])]:
 for name,h in files.items():assert sha(Path(directory)/name)==h,name
assert sha(Path(eff['cpu_tests']['xml']))==eff['cpu_tests']['sha256']
assert sha(Path(full['padding_comparison_receipt']))==full['padding_comparison_receipt_sha256']
assert json.loads(active_path.read_bytes())==active, 'Active identity changed during composition'
assert sha(source_path)==@EXPECTED@['source'] and driver.create_time()==1791291997.14

prepared=copy.deepcopy(full)
# These fields described the previous CPU preparation/checkpoint28, not this
# metadata-only composition. Preserve them explicitly as historical context.
historical_names=('checkpoint_root','actor_checkpoint','checkpoint_files','marker',
 'resume_checkpoint','formal_output','formal_output_created','launch_plan','process',
 'wall_seconds','native_inspection_resource','receipts','no_launch','active_manifest_modified',
 'model_initializations','DT_calls','rollout_calls','backward_calls','optimizer_steps',
 'minimum_completed_checkpoint','prepared_unix','status','scope','preparation_source_sha256',
 'preparation_script_sha256','preparation_repository_commit')
historical={n:prepared.pop(n) for n in historical_names if n in prepared}
prepared.update(prepared_unix=time.time(),status='prepared_only_metadata_composed_not_submitted',
 scope='Existing retained-source entry with current complete white owner. Metadata composition only; no stop, launch, model, tests or numerical change.',
 prior_driver_pid=job['pid'],prior_driver_created_unix=driver.create_time(),
 prior_entry=job['entry'],prior_verl_root=job['verl_root'],
 future_checkpoint_root=job['checkpoints'],entry=eff['entry'],entry_sha256=entry_hashes,
 prior_source_receipt=str(source_path),prior_source_sha256=sha(source_path),
 prior_budget=copy.deepcopy(job['budget']),
 resume_behavior='Existing resume_at_native_checkpoint.py selects the next requested completed native marker, then original submitter/launcher/VERL loader. No checkpoint selected by this metadata preparation.',
 preparation_repository_commit=@REVISION@,preparation_script_sha256=@SCRIPT_SHA@,
 inherited_white_preparation={'path':str(full_path),'sha256':sha(full_path),'historical_fields':historical},
 retained_entry_preparation={'path':str(eff_path),'sha256':sha(eff_path),'changes':eff['changes'],
  'cpu_tests':eff['cpu_tests'],'source_fix_commit':eff['source_fix_commit']},
 current_job=copy.deepcopy(job),
 composition={'changed_entry_files':sorted(eff['changes']),
  'unchanged_entry_files':len(eff['unchanged_entry_sha256']),
  'VERL_DT_LOOP_budget_sampling_async_whitening_preserved':True,
  'active_manifest_sha256':hashlib.sha256(active_bytes).hexdigest(),
  'active_manifest_modified':False,'stop_calls':0,'launch_calls':0,'test_calls':0,
  'GPU_calls':0,'model_initializations':0,'training_calls':0,
  'process':{'pid':psutil.Process().pid,'created_unix':psutil.Process().create_time(),
   'rss_bytes':psutil.Process().memory_info().rss},'torch_imported':'torch' in sys.modules,
  'wall_seconds':time.time()-started})
out.mkdir(parents=True,exist_ok=False)
source_file=out/'prepare_appworld_retained_resume_metadata.py'
with source_file.open('xb') as f:f.write(base64.b64decode(@SCRIPT_BYTES@))
assert sha(source_file)==@SCRIPT_SHA@
prepared['preparation_script_remote_path']=str(source_file)
with (out/'prepared.json').open('x') as f:f.write(json.dumps(prepared,indent=2)+'\n')
result={'prepared':str(out/'prepared.json'),'sha256':sha(out/'prepared.json'),
 'status':prepared['status'],'prior_driver_pid':job['pid'],'prior_driver_birth':driver.create_time(),
 'entry':prepared['entry'],'future_checkpoint_root':prepared['future_checkpoint_root'],
 'composition':prepared['composition']}
with (out/'composition-result.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
'''
    values = {'ROOT': ROOT, 'REMOTE': remote, 'FULL': FULL, 'EFFICIENCY': args.efficiency_receipt,
              'CANDIDATE_ENTRY': args.entry, 'EXPECTED': expected, 'REVISION': revision, 'SCRIPT_SHA': script_sha,
              'SCRIPT_BYTES': base64.b64encode(source_bytes).decode('ascii')}
    for key, value in values.items():
        code = code.replace('@' + key + '@', repr(value))
    ast.parse(code)
    local.mkdir(parents=True, exist_ok=False)
    shell = f"set -e\nsource {ENTRY}/metax-entry.env.sh\n\"$VENV_PYTHON\" - <<'PY'\n{code}\nPY\n"
    (local / 'prepare.sh').write_text(shell, encoding='utf-8')
    result = subprocess.run(SSH + ['bash', '-s'], input=shell.encode(), capture_output=True)
    (local / 'prepare.stdout.txt').write_bytes(result.stdout + result.stderr)
    result.check_returncode()
    for name in ('prepared.json', 'composition-result.json'):
        subprocess.run(SCP + [f'{SSH[-1]}:{remote}/{name}', str(local / name)], check=True)
    print((local / 'composition-result.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
