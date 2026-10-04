"""Prepare the exited AppWorld job's minimal retry-recording resume candidate.

Preparation only: copy its frozen async sources, retain the completed native
host-cache resource patch, and splice the tested RecordedRunner initializer.
No model, engine, GPU test, job stop, loader, or training submission is run.
The existing submit_prepared_appworld_resume.py owns the later submission.
"""
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, remote


BASELINE_COMMIT = "ea2d32a3359c29100704dde1c73efb4641b14669"
FIX_COMMIT = "206fececc54ce3fbf7ef739284f70568ae3599c6"
WORKER_BEFORE_SHA = "2fcd1128cf036e6091a452de165e16dcdf63113d15da669b35e1b13437306abd"
WORKER_AFTER_SHA = "6ad5a3e383d032fdc7f0dd2dab1ee027d8f0ed182f9725d07a3faedb46e54ec9"
HOST_WORKER_SHA = "e5eb4afc42f10fb4608b3ac43046c906d6a2d21a5387ee02e176bc395f1c6f39"
LAUNCHER_BEFORE_SHA = "0f7c3a837ea90665f141d99c86d6d2d5b8a9d4d6c244233a00de36a591c474cc"
ACTOR_SHA = "1f862e8bbdaad6fa116d0670772ad41269529a3a1e4a5b1eb383352d0372e9bd"
SUBMISSION_SHA = "a90911710d87d8da82e639eda5ba2c78ae3f8738136b56487108329da4022252"
NAME = "appworld-native-async-retry-recording-resume-20261004"
TEST_RELATIVE = ("research/temporary/rl_upstream_alignment_20260929/"
                 "appworld-alignment-20261004/test_native_retry_recording.py")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(commit, name):
    return subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=REPO)


def initializer_range(data):
    tree = ast.parse(data)
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "recorded_runner")
    owner_class = next(n for n in factory.body if isinstance(n, ast.ClassDef)
                       and n.name == "RecordedRunner")
    method = next(n for n in owner_class.body if isinstance(n, ast.FunctionDef)
                  and n.name == "__init__")
    lines = data.splitlines(keepends=True)
    return sum(map(len, lines[:method.lineno-1])), sum(map(len, lines[:method.end_lineno]))


def main():
    original = git_blob(BASELINE_COMMIT, "experiments/rl/loop_owner_worker.py")
    fixed = git_blob(FIX_COMMIT, "experiments/rl/loop_owner_worker.py")
    assert sha(original) == WORKER_BEFORE_SHA and sha(fixed) == WORKER_AFTER_SHA
    old_start, old_end = initializer_range(original)
    new_start, new_end = initializer_range(fixed)
    assert original[:old_start] == fixed[:new_start]
    assert original[old_end:] == fixed[new_end:]
    assert len(fixed.splitlines()) - len(original.splitlines()) == 9
    host = (AUDIT / "native-host-cache-owner-patched.py").read_bytes()
    assert sha(host) == HOST_WORKER_SHA
    tests = git_blob(FIX_COMMIT, TEST_RELATIVE)
    submission = AUDIT / "submit_prepared_appworld_resume.py"
    assert sha(submission.read_bytes()) == SUBMISSION_SHA
    accepted_cpu = (AUDIT / "appworld-alignment-20261004/retry-recording-fixed-cpu.json").read_bytes()
    assert json.loads(accepted_cpu)["exit_code"] == 0
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    payload = dict(original=base64.b64encode(original).decode(),
        initializer=base64.b64encode(fixed[new_start:new_end]).decode(),
        fixed_worker_sha256=WORKER_AFTER_SHA, old_worker_sha256=WORKER_BEFORE_SHA,
        fix_commit=FIX_COMMIT, baseline_commit=BASELINE_COMMIT,
        tests=base64.b64encode(tests).decode(), tests_sha256=sha(tests),
        accepted_cpu=base64.b64encode(accepted_cpu).decode(), accepted_cpu_sha256=sha(accepted_cpu),
        host_worker=base64.b64encode(host).decode(), host_worker_sha256=HOST_WORKER_SHA,
        launcher_before_sha256=LAUNCHER_BEFORE_SHA, actor_sha256=ACTOR_SHA,
        submission_script_sha256=SUBMISSION_SHA,
        preparation_repository_commit=revision,
        preparation_script_sha256=sha(Path(__file__).read_bytes()))
    script = r'''set -e
source @ENTRY@/metax-entry.env.sh
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export MACA_VISIBLE_DEVICES=-1
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import ast,base64,hashlib,json,os,psutil,runpy,shutil,sys,time

root=Path(@ROOT@)
payload=json.loads(@PAYLOAD@)
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
digest=lambda b:hashlib.sha256(b).hexdigest()
decode=lambda key:base64.b64decode(payload[key])
active=read(root/'active-training.json')
job=next(j for j in active['jobs'] if j['task']=='AppWorld')
frozen=root/'candidates/appworld-native-async-015-native-futures-20261002'
assert job['pid']==3232113 and job['observed_process_created_unix']==1790924078.76
assert job['devices']==[2,3] and job['entry']==str(frozen/'entry')
assert job['verl_root']==str(frozen/'verl')
if psutil.pid_exists(job['pid']):
    process=psutil.Process(job['pid'])
    assert process.create_time()==job['observed_process_created_unix']
    assert process.status()==psutil.STATUS_ZOMBIE, 'Original job is running; preparation stopped without mutation'
marker=Path(job['checkpoints'])/'latest_checkpointed_iteration.txt'
assert int(marker.read_text().strip())==19
checkpoint=marker.parent/'global_step_19'
checkpoint_files=[checkpoint/'data.pt']+[
    checkpoint/'actor'/f'{kind}_world_size_2_rank_{rank}.pt'
    for rank in range(2) for kind in ('model','optim','extra_state')]
assert all(p.is_file() for p in checkpoint_files)
source_path=Path(job['source_receipt']);source=read(source_path)
prior_path=Path(source['prepared_receipt']);prior=read(prior_path)
assert prior['entry']==job['entry'] and prior['verl_root']==job['verl_root']
assert prior['dt_root']==str(root/'releases/c9cd147')==job['dt_root']
assert prior['prior_budget']==job['budget']
for name,h in source['entry_sha256'].items(): assert sha(frozen/'entry'/name)==h,name
owner_files=dict(prior.get('owner_sha256',prior.get('owner_head_sha256')))
for name,h in owner_files.items(): assert sha(frozen/'verl'/name)==h,name
assert sha(frozen/'verl/verl/workers/actor/dp_actor.py')==payload['actor_sha256']
for name,h in prior['dt_source_sha256'].items(): assert sha(Path(prior['dt_root'])/name)==h,name
worker_path=frozen/'entry/loop_owner_worker.py'
assert sha(worker_path)==payload['old_worker_sha256']
assert worker_path.read_bytes()==decode('original')
launcher_path=frozen/'entry/launch_appworld_native.py'
assert sha(launcher_path)==payload['launcher_before_sha256']
host_path=root/'candidates/native-host-cache-phase-20261002/e5eb4afc42f1/fsdp_workers.py'
assert sha(host_path)==payload['host_worker_sha256']
assert host_path.read_bytes()==decode('host_worker')
cache_receipt=root/'receipts/native-host-cache-phase-20261002/AppWorld-3232113/complete.json'
cache=read(cache_receipt)

def tree_bytes(directory):
    return {str(p.relative_to(directory)):({'symlink':os.readlink(p)} if p.is_symlink()
            else {'sha256':sha(p)}) for p in directory.rglob('*') if p.is_symlink() or p.is_file()}

base=root/'candidates/@NAME@'
assert not base.exists(), 'Immutable candidate already exists; inspect it instead of replacing it'
entry=base/'entry';owner=base/'verl'
base.mkdir()
old_entry_tree=tree_bytes(frozen/'entry');old_owner_tree=tree_bytes(frozen/'verl')
shutil.copytree(frozen/'entry',entry,symlinks=True)
shutil.copytree(frozen/'verl',owner,symlinks=True)
assert tree_bytes(entry)==old_entry_tree and tree_bytes(owner)==old_owner_tree
(base/'prior-active-training.json').write_bytes((root/'active-training.json').read_bytes())
(base/'prior-source.json').write_bytes(source_path.read_bytes())
(base/'prior-prepared.json').write_bytes(prior_path.read_bytes())

def initializer_range(data):
    factory=next(n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name=='recorded_runner')
    cls=next(n for n in factory.body if isinstance(n,ast.ClassDef) and n.name=='RecordedRunner')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    lines=data.splitlines(keepends=True)
    return sum(map(len,lines[:method.lineno-1])),sum(map(len,lines[:method.end_lineno]))

worker=entry/'loop_owner_worker.py';before=worker.read_bytes()
start,end=initializer_range(before);method=decode('initializer')
after=before[:start]+method+before[end:]
ast.parse(after)
assert digest(after)==payload['fixed_worker_sha256']
assert after[:start]==before[:start] and after[start+len(method):]==before[end:]
(base/'loop_owner_worker.before.py').write_bytes(before)
worker.write_bytes(after)
launcher=entry/'launch_appworld_native.py';launcher_before=launcher.read_bytes()
anchor=b"    options.update({'+ray_init.runtime_env.worker_process_setup_hook': 'observe_worker_visibility.install',\n"
addition=b"        '+ray_init.runtime_env.env_vars.VERL_RELEASE_UNUSED_HOST_CACHE': '1',\n"
assert launcher_before.count(anchor)==1
offset=launcher_before.index(anchor)+len(anchor)
launcher_after=launcher_before[:offset]+addition+launcher_before[offset:]
ast.parse(launcher_after)
assert launcher_after[:offset]==launcher_before[:offset]
assert launcher_after[offset+len(addition):]==launcher_before[offset:]
(base/'launch_appworld_native.before-resource-config.py').write_bytes(launcher_before)
launcher.write_bytes(launcher_after)
host_target=owner/'verl/workers/fsdp_workers.py'
(base/'fsdp_workers.before-resource-restore.py').write_bytes(host_target.read_bytes())
shutil.copyfile(host_path,host_target)
owner_files['verl/workers/fsdp_workers.py']=payload['host_worker_sha256']
new_entry_tree=tree_bytes(entry);new_owner_tree=tree_bytes(owner)
entry_changes={name for name in old_entry_tree.keys()|new_entry_tree.keys()
               if old_entry_tree.get(name)!=new_entry_tree.get(name)}
owner_changes={name for name in old_owner_tree.keys()|new_owner_tree.keys()
               if old_owner_tree.get(name)!=new_owner_tree.get(name)}
assert entry_changes=={'loop_owner_worker.py','launch_appworld_native.py'}
assert owner_changes=={'verl/workers/fsdp_workers.py'}

# Reuse the committed CPU owner tests without copying retry or episode logic.
test_path=base/'test_native_retry_recording.py'
test_path.write_bytes(decode('tests'))
assert sha(test_path)==payload['tests_sha256']
test_namespace=runpy.run_path(str(test_path))
test_names=['test_committed_baseline_reproduces_discarded_episode_mismatch',
    'test_no_retry_retains_identical_owner_artifacts',
    'test_native_successful_retries_keep_only_retained_episode',
    'test_original_restart_failure_does_not_erase_recorded_attempt',
    'test_original_restart_arguments_and_return_value_are_preserved']
for name in test_names:
    test_namespace[name].__globals__.update(OWNER=Path(prior['loop_root']),BRIDGE=worker)
pair=(before.decode(),after.decode());cases=[]
for name in test_names:
    if name=='test_native_successful_retries_keep_only_retained_episode':
        for attempts in (1,2,4):
            test_namespace[name](pair,attempts);cases.append(f'{name}[{attempts}]')
    else:
        test_namespace[name](pair);cases.append(name)
assert len(cases)==7
accepted_cpu_path=base/'prior-retry-recording-fixed-cpu.json'
accepted_cpu_path.write_bytes(decode('accepted_cpu'))
assert sha(accepted_cpu_path)==payload['accepted_cpu_sha256']

# Call the original author configuration composer only; do not execute main.
os.environ.update(LOOP_ROOT=prior['loop_root'],VERL_ROOT=str(owner),DT_ROOT=prior['dt_root'],
    DT_ENTRY_ROOT=str(entry),APPWORLD_ROOT=str(root/'receipts/environment-only-20260930/loop-entry/appworld-root'))
sys.path[:0]=[str(entry),str(owner),prior['loop_root']]
old=runpy.run_path(str(launcher_path));new=runpy.run_path(str(launcher))
output=Path('/same-formal-output')
old_options,old_sampling=old['options_for'](output,resume_from=checkpoint)
new_options,new_sampling=new['options_for'](output,resume_from=checkpoint)
assert old_sampling==new_sampling
assert new_options['data.custom_cls.path']==str(entry/'loop_iteration_dataset.py')
new_options_for_comparison=dict(new_options)
new_options_for_comparison['data.custom_cls.path']=old_options['data.custom_cls.path']
changes={k:dict(before=old_options.get(k),after=new_options_for_comparison.get(k))
         for k in old_options.keys()|new_options_for_comparison.keys()
         if old_options.get(k)!=new_options_for_comparison.get(k)}
resource_key='+ray_init.runtime_env.env_vars.VERL_RELEASE_UNUSED_HOST_CACHE'
assert changes=={resource_key:{'before':None,'after':'1'}},changes
assert new_options['actor_rollout_ref.rollout.mode']=='async'
assert new_options['actor_rollout_ref.rollout.chat_scheduler']=='verl.workers.rollout.async_server.ChatCompletionScheduler'
assert new_options['actor_rollout_ref.model.lora_rank']==8
assert new_options['actor_rollout_ref.model.lora_alpha']==16
assert new_options['actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu']==4
assert new_options['actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu']==4
assert new_options['actor_rollout_ref.rollout.max_model_len']==32768
assert new_options['trainer.resume_mode']=='resume_path'
assert new_options['trainer.resume_from_path']==str(checkpoint)
assert tree_bytes(entry)==new_entry_tree and tree_bytes(owner)==new_owner_tree
verification=dict(observed_unix=time.time(),passed=True,
    scope='Exact frozen-source initializer splice, existing CPU author retry tests, and original configuration composition; no model/GPU/loader/throughput test',
    cases=cases,cases_passed=len(cases),test_source_sha256=sha(test_path),
    original_author_retry_source=dict(path=str(Path(prior['loop_root'])/'phi_agents/evals/appworld_evals.py'),
        sha256=sha(Path(prior['loop_root'])/'phi_agents/evals/appworld_evals.py')),
    entry_changed_files=sorted(entry_changes),owner_changed_files=sorted(owner_changes),
    configuration_changes=changes,sampling_unchanged=old_sampling,
    prior_cpu_receipt=dict(path=str(accepted_cpu_path),sha256=sha(accepted_cpu_path)),
    initializer=dict(source_commit=payload['fix_commit'],baseline_commit=payload['baseline_commit'],
        before_sha256=digest(before),after_sha256=digest(after),outside_method_bytes_unchanged=True,added_lines=9),
    launcher=dict(before_sha256=digest(launcher_before),after_sha256=digest(launcher_after),
        outside_single_resource_config_insertion_bytes_unchanged=True),
    restored_resource_owner=dict(path=str(host_path),sha256=sha(host_path),
        completed_receipt=str(cache_receipt),completed_receipt_sha256=sha(cache_receipt),
        resource_environment={'VERL_RELEASE_UNUSED_HOST_CACHE':'1'}),
    options=new_options,checkpoint=dict(path=str(checkpoint),marker=str(marker),value=19,
        marker_sha256=sha(marker),files=[dict(path=str(p),bytes=p.stat().st_size) for p in checkpoint_files]))
verification_path=base/'cpu-verification.json'
verification_path.write_text(json.dumps(verification,indent=2)+'\n')
prepared=dict(prior,prepared_unix=time.time(),status='prepared_only_retry_recording_cpu_owner_verified_not_submitted',
    scope='Retain frozen official async bridge, restore already accepted host-cache resource override, and apply only tested successful-restart recording cleanup; original VERL owns later resume',
    prior_driver_pid=job['pid'],prior_entry=job['entry'],prior_verl_root=job['verl_root'],
    entry=str(entry),verl_root=str(owner),future_checkpoint_root=job['checkpoints'],
    entry_sha256={p.name:sha(p) for p in entry.glob('*.py')},owner_sha256=owner_files,owner_head_sha256=owner_files,
    prior_prepared_receipt=str(prior_path),prior_prepared_receipt_sha256=sha(prior_path),
    prior_source_receipt=str(source_path),prior_source_sha256=sha(source_path),
    preparation_repository_commit=payload['preparation_repository_commit'],
    preparation_script_sha256=payload['preparation_script_sha256'],
    configuration_changes=changes,prior_budget=job['budget'],minimum_completed_checkpoint=19,
    resume_checkpoint=str(checkpoint),resume_launcher=verification['launcher'],
    completed_overlay_receipt=str(cache_receipt),completed_overlay_sha256=sha(cache_receipt),
    resource_environment={'VERL_RELEASE_UNUSED_HOST_CACHE':'1'},
    retry_recording_fix_commit=payload['fix_commit'],retry_recording_comparison=str(verification_path),
    retry_recording_comparison_sha256=sha(verification_path),
    submission_interface=dict(script='submit_prepared_appworld_resume.py',sha256=payload['submission_script_sha256'],
        checkpoint=str(checkpoint),prepared=str(base/'prepared.json')))
(base/'prepared.json').write_text(json.dumps(prepared,indent=2)+'\n')
print(json.dumps(dict(prepared=str(base/'prepared.json'),verification=str(verification_path),
    status=prepared['status'],checkpoint=str(checkpoint),cases_passed=len(cases),
    changed_entry_files=sorted(entry_changes),changed_owner_files=sorted(owner_changes)),indent=2))
PY
'''
    remote(script.replace('@ENTRY@', ENTRY).replace('@ROOT@', repr(ROOT))
           .replace('@NAME@', NAME).replace('@PAYLOAD@', repr(json.dumps(payload))))


if __name__ == '__main__':
    main()
