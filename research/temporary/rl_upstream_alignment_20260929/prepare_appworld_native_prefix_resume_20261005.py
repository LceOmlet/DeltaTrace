"""Prepare AppWorld's isolated prefix-provider resume; never submit training.

Compose pinned owner seams only. Preserve the retry recording repair, actor,
host-cache patch, official task options and checkpoint. Bind the completed
original operator receipts; preparation is not whole-DT or training acceptance.
"""
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess

from stage_environment_entry import ENTRY, REPO, ROOT, remote


NAME = 'appworld-native-prefix-resume-20261005'
SOURCE = ROOT + '/candidates/appworld-native-async-retry-recording-resume-20261004/prepared.json'
SOURCE_SHA = '001078cf260e11a451156b01529406bdce24059f4afeb3a54ca7907c8b305b41'
RETRY_CPU_SHA = 'ee600363f6853531740fe50ef06bab951401cabe1e704c02594796dd11384fdf'
BASE = '99fb5c28d9be064bab22dc8ad949494b3fd85150'
PROVIDER = '8e7dd71258b2173ae0be0784e7783a49c9cd5b94'
BRANCH = '206fececc54ce3fbf7ef739284f70568ae3599c6'
LEASE = '712795d1812b6b29b9626ab2ab88f0bd98564d1e'
ACTOR_SHA = '1f862e8bbdaad6fa116d0670772ad41269529a3a1e4a5b1eb383352d0372e9bd'
HOST_SHA = 'e5eb4afc42f10fb4608b3ac43046c906d6a2d21a5387ee02e176bc395f1c6f39'
RETRY_SHA = '6ad5a3e383d032fdc7f0dd2dab1ee027d8f0ed182f9725d07a3faedb46e54ec9'
SUBMIT_SHA = 'a90911710d87d8da82e639eda5ba2c78ae3f8738136b56487108329da4022252'
SEGMENTS = 'experiments/rl/results_native_prefix_segments_20261005.json'
SEGMENTS_SHA = '8ae9e1c766b23282e727a6d565b9c14e3b5eba38f7b090946a7b5601f2072183'
COVERAGE_SHA = 'df03f9a10bfecbfbe71616993e659800e87bbb5f725b6b39cd36914215fe9e33'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def blob(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=REPO)


def method_range(data):
    cls = next(n for n in ast.parse(data).body
               if isinstance(n, ast.ClassDef) and n.name == '_Qwen35CausalOwnerView')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'synchronize_prefix_start')
    lines = data.splitlines(keepends=True)
    return sum(map(len, lines[:method.lineno-1])), sum(map(len, lines[:method.end_lineno]))


def build_payload():
    """Exact local Git bytes; no connection or model import."""
    segments_path = Path(REPO) / SEGMENTS
    assert digest(segments_path.read_bytes()) == SEGMENTS_SHA
    segments = json.loads(segments_path.read_bytes())
    coverage_path = Path(REPO) / segments['owner_coverage_index']
    assert digest(coverage_path.read_bytes()) == COVERAGE_SHA
    coverage = json.loads(coverage_path.read_bytes())
    operator_evidence = dict(
        scope=segments['official_assertion_scope'],
        aggregate=dict(path=str(segments_path), sha256=SEGMENTS_SHA),
        original_coverage_index=dict(path=str(coverage_path), sha256=COVERAGE_SHA),
        original_coverage_source_bindings=[
            dict(source_path=row['local_path'], sha256=row['sha256'],
                 source_commit=row['last_source_change_commit'])
            for row in coverage['unchanged_prefix_source_bindings']],
        original_segment_receipts=[], whole_dt_acceptance=False,
        formal_training_submitted=False)
    for recorded in segments['ranks']:
        result = recorded['content']
        command = segments['job']['content']['commands'][result['rank']]
        operator_evidence['original_segment_receipts'].append(dict(
            path=command[command.index('--output')+1], sha256=recorded['sha256'],
            rank=result['rank'], status=result['status'],
            original_assertions=result['original_assertions'],
            state_error_ratio=result['state_error_ratio'],
            official_source=result['official_source'],
            original_operator_source=result['original_operator_source'],
            provenance=result['provenance'],
            hf_storage_observation=result['hf_storage_observation']))
    specs = [
        ('entry/reward_readout.py', BASE, PROVIDER, 'experiments/rl/reward_readout.py',
         '8acf94d46cef8ca03ce1e92352723b2490c76e66a075f6b2a5f969163bcdc774'),
        ('entry/deltatrace_credit.py', BASE, PROVIDER, 'experiments/rl/deltatrace_credit.py',
         '8046762ae2fd149b299e29f9a331d8ae1aed665f2de895573e78e61ebc2d8497'),
        ('deltatrace/clean/qwen35/qwen35_dense_finite_runner.py', 'c9cd147', PROVIDER,
         'deltatrace/clean/qwen35/qwen35_dense_finite_runner.py',
         '6348ebef115ff0ea45deeee3df8b65dcfbf0d1ae86619edc67f34d16d92d69aa'),
        ('deltatrace/clean/qwen35/qwen35_native_prefix_artifacts.py', None, PROVIDER,
         'deltatrace/clean/qwen35/qwen35_native_prefix_artifacts.py',
         '06fda7843c4120cb1406b671f06cb19f29921b61b5d524dc28539550446aef16'),
        ('entry/native_prefix_leases.py', None, LEASE, 'experiments/rl/native_prefix_leases.py',
         'd5b539c7ffb8a431374cf79f4b995ef6f4138c3e6e289890409b6b79dbad393e'),
    ]
    changes = []
    for target, old_commit, new_commit, source, expected in specs:
        before = blob(old_commit, source) if old_commit else None
        after = blob(new_commit, source)
        assert digest(after) == expected
        ast.parse(after)
        change = dict(target=target, before=None if before is None else base64.b64encode(before).decode(),
            after=base64.b64encode(after).decode(), before_sha256=None if before is None else digest(before),
            after_sha256=expected, baseline_commit=old_commit, source_commit=new_commit, source_path=source)
        if old_commit:
            change['original_git_diff'] = subprocess.check_output(
                ['git','diff',old_commit,new_commit,'--',source],cwd=REPO,text=True)
        changes.append(change)
    original = blob(BASE, 'experiments/rl/deltatrace_rollout.py')
    reference = blob(BRANCH, 'experiments/rl/deltatrace_rollout.py')
    assert digest(original) == '0ad37a17aede30089fd2ac9a609a42689e6a3a68d00b601520db0a5cff8b8e6e'
    start, end = method_range(original); a, b = method_range(reference)
    replacement = reference[a:b]
    after = original[:start] + replacement + original[end:]
    assert after[:start] == original[:start] and after[start+len(replacement):] == original[end:]
    anchor = b"        self.readout = None if self.readout_options['task'] == 'AppWorld' else EventRatioReadout(\n"
    addition = (b'        from native_prefix_leases import prepare_native_prefix_leases\n'
                b"        self.readout_options['prefix_lease_factory'] = prepare_native_prefix_leases\n")
    assert after.count(anchor) == 1
    offset = after.index(anchor)
    patched = after[:offset] + addition + after[offset:]
    assert patched[:offset] == after[:offset] and patched[offset+len(addition):] == after[offset:]
    ast.parse(patched)
    changes.append(dict(target='entry/deltatrace_rollout.py',before=base64.b64encode(original).decode(),
        after=base64.b64encode(patched).decode(),before_sha256=digest(original),after_sha256=digest(patched),
        source_commit=BRANCH,source_path='experiments/rl/deltatrace_rollout.py',
        outside_original_method_and_factory_insertion_bytes_unchanged=True,
        original_method_sha256=digest(original[start:end]),replacement_method_sha256=digest(replacement)))
    assert digest(Path(__file__).with_name('submit_prepared_appworld_resume.py').read_bytes()) == SUBMIT_SHA
    return dict(changes=changes,source_sha256=SOURCE_SHA,retry_cpu_sha256=RETRY_CPU_SHA,
        actor_sha256=ACTOR_SHA,host_sha256=HOST_SHA,retry_sha256=RETRY_SHA,submission_sha256=SUBMIT_SHA,
        repository_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        script_sha256=digest(Path(__file__).read_bytes()),original_operator_evidence=operator_evidence)


def preparation_script(payload):
    """Return the CPU-only preparation, without invoking any submitter."""
    script = r'''set -e
source @ENTRY@/metax-entry.env.sh
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export MACA_VISIBLE_DEVICES=-1
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import ast,base64,hashlib,json,os,runpy,shutil,sys,time
root=Path(@ROOT@);payload=json.loads(@PAYLOAD@)
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prior_path=Path(@SOURCE@)
assert sha(prior_path)==payload['source_sha256']
prior=read(prior_path)
assert prior['status']=='prepared_only_retry_recording_cpu_owner_verified_not_submitted'
verification=Path(prior['retry_recording_comparison'])
assert sha(verification)==payload['retry_cpu_sha256']==prior['retry_recording_comparison_sha256']
assert read(verification)['passed'] and read(verification)['cases_passed']==7
# Bind existing immutable receipts only; do not re-evaluate or replace their assertions.
for receipt in payload['original_operator_evidence']['original_segment_receipts']:
    assert sha(Path(receipt['path']))==receipt['sha256']
active=read(root/'active-training.json');job=next(j for j in active['jobs'] if j['task']=='AppWorld')
assert job['pid']==prior['prior_driver_pid']==3232113
assert job['entry']==prior['prior_entry'] and job['verl_root']==prior['prior_verl_root']
assert job['devices']==[2,3] and job['budget']==prior['prior_budget']
source_entry=Path(prior['entry']);source_owner=Path(prior['verl_root']);source_dt=Path(prior['dt_root'])
assert source_dt==root/'releases/c9cd147'
for n,h in prior['entry_sha256'].items():assert sha(source_entry/n)==h,n
owner_files=dict(prior.get('owner_sha256',prior['owner_head_sha256']))
for n,h in owner_files.items():assert sha(source_owner/n)==h,n
for n,h in prior['dt_source_sha256'].items():assert sha(source_dt/n)==h,n
assert sha(source_owner/'verl/workers/actor/dp_actor.py')==payload['actor_sha256']
assert sha(source_owner/'verl/workers/fsdp_workers.py')==payload['host_sha256']
assert sha(source_entry/'loop_owner_worker.py')==payload['retry_sha256']
assert prior['resource_environment']=={'VERL_RELEASE_UNUSED_HOST_CACHE':'1'}
marker=Path(job['checkpoints'])/'latest_checkpointed_iteration.txt'
assert int(marker.read_text().strip())==19
checkpoint=marker.parent/'global_step_19';assert str(checkpoint)==prior['resume_checkpoint']
checkpoint_files=[checkpoint/'data.pt']+[
    checkpoint/'actor'/f'{kind}_world_size_2_rank_{rank}.pt'
    for rank in range(2) for kind in ('model','optim','extra_state')]
assert all(p.is_file() for p in checkpoint_files)

def tree_bytes(directory):
    return {str(p.relative_to(directory)):({'symlink':os.readlink(p)} if p.is_symlink()
            else {'sha256':sha(p)}) for p in directory.rglob('*') if p.is_symlink() or p.is_file()}

base=root/'candidates'/@NAME@
assert not base.exists(), 'Immutable candidate already exists; inspect without replacing'
entry=base/'entry';owner=base/'verl';dt=base/'deltatrace';base.mkdir()
old_entry=tree_bytes(source_entry);old_owner=tree_bytes(source_owner);old_dt=tree_bytes(source_dt)
for before,after in ((source_entry,entry),(source_owner,owner),(source_dt,dt)):
    shutil.copytree(before,after,symlinks=True)
assert tree_bytes(entry)==old_entry and tree_bytes(owner)==old_owner and tree_bytes(dt)==old_dt
(base/'prior-prepared.json').write_bytes(prior_path.read_bytes())
(base/'prior-active-training.json').write_bytes((root/'active-training.json').read_bytes())
(base/'prior-retry-cpu-verification.json').write_bytes(verification.read_bytes())
patches=[]
for change in payload['changes']:
    p=base/change['target']
    if change['before'] is None:assert not p.exists()
    else:
        assert p.read_bytes()==base64.b64decode(change['before']) and sha(p)==change['before_sha256']
    p.write_bytes(base64.b64decode(change['after']))
    assert sha(p)==change['after_sha256'];ast.parse(p.read_bytes())
    patches.append({k:v for k,v in change.items() if k not in ('before','after')})
new_entry=tree_bytes(entry);new_dt=tree_bytes(dt)
entry_changes={n for n in old_entry.keys()|new_entry.keys() if old_entry.get(n)!=new_entry.get(n)}
dt_changes={n for n in old_dt.keys()|new_dt.keys() if old_dt.get(n)!=new_dt.get(n)}
assert entry_changes=={'reward_readout.py','deltatrace_credit.py','deltatrace_rollout.py','native_prefix_leases.py'}
assert dt_changes=={'clean/qwen35/qwen35_dense_finite_runner.py','clean/qwen35/qwen35_native_prefix_artifacts.py'}
assert tree_bytes(owner)==old_owner
assert sha(entry/'loop_owner_worker.py')==payload['retry_sha256']

# Import only the original author configuration composer, never its main/loader/model.
os.environ.update(LOOP_ROOT=prior['loop_root'],VERL_ROOT=str(owner),DT_ROOT=str(dt),DT_ENTRY_ROOT=str(entry),
    APPWORLD_ROOT=str(root/'receipts/environment-only-20260930/loop-entry/appworld-root'))
sys.path[:0]=[str(entry),str(owner),prior['loop_root']]
old=runpy.run_path(str(source_entry/'launch_appworld_native.py'))
new=runpy.run_path(str(entry/'launch_appworld_native.py'))
output=Path('/same-formal-output')
old_options,old_sampling=old['options_for'](output,resume_from=checkpoint)
new_options,new_sampling=new['options_for'](output,resume_from=checkpoint)
comparison=dict(new_options);comparison['data.custom_cls.path']=old_options['data.custom_cls.path']
assert comparison==old_options and new_sampling==old_sampling
assert new_options['actor_rollout_ref.model.lora_rank']==8
assert new_options['actor_rollout_ref.model.lora_alpha']==16
assert new_options['actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu']==4
assert new_options['actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu']==4
assert new_options['actor_rollout_ref.rollout.max_model_len']==32768
assert new_options['+ray_init.runtime_env.env_vars.VERL_RELEASE_UNUSED_HOST_CACHE']=='1'
assert tree_bytes(owner)==old_owner and tree_bytes(entry)==new_entry and tree_bytes(dt)==new_dt
cpu=dict(observed_unix=time.time(),passed=True,
    scope='Frozen owner source bytes and unchanged official launch composition only; no GPU/numerical/capacity/throughput acceptance',
    patches=patches,entry_changes=sorted(entry_changes),dt_changes=sorted(dt_changes),owner_changes=[],
    options=new_options,sampling=new_sampling,official_task_options_unchanged=True,
    inherited_retry_cpu=dict(path=str(verification),sha256=sha(verification)),
    checkpoint=dict(path=str(checkpoint),marker=str(marker),value=19,marker_sha256=sha(marker)))
cpu_path=base/'cpu-preparation-verification.json';cpu_path.write_text(json.dumps(cpu,indent=2)+'\n')
prepared=dict(prior,prepared_unix=time.time(),status='prepared_only_original_operator_checks_recorded_not_submitted',
    scope='AppWorld only; pinned prefix artifacts/provider and verified synchronization seam; original retry/actor/task/optimizer preserved; no root tape or new retention policy',
    entry=str(entry),verl_root=str(owner),dt_root=str(dt),
    entry_sha256={p.name:sha(p) for p in entry.glob('*.py')},owner_sha256=owner_files,owner_head_sha256=owner_files,
    dt_source_sha256={str(p.relative_to(dt)):sha(p) for p in dt.rglob('*.py')},
    prior_prefix_prepared_receipt=str(prior_path),prior_prefix_prepared_sha256=sha(prior_path),
    preparation_repository_commit=payload['repository_commit'],preparation_script_sha256=payload['script_sha256'],
    prefix_preparation_verification=str(cpu_path),prefix_preparation_verification_sha256=sha(cpu_path),
    prefix_lifetime='One original reward-denominator DP RPC; no bank retained across policy updates',
    actual_segment_assertion_status='passed_original_interior_segment_ht_assertion_on_saved_actual_operands',
    original_operator_evidence=payload['original_operator_evidence'],
    configuration_changes={},resume_checkpoint=str(checkpoint),minimum_completed_checkpoint=19,
    resume_launcher=dict(prior['resume_launcher'],path=str(entry/'launch_appworld_native.py'),
        sha256=sha(entry/'launch_appworld_native.py')),
    submission_interface=dict(prior['submission_interface'],prepared=str(base/'prepared.json')))
(base/'prepared.json').write_text(json.dumps(prepared,indent=2)+'\n')
print(json.dumps(dict(prepared=str(base/'prepared.json'),sha256=sha(base/'prepared.json'),status=prepared['status'],
    verification=str(cpu_path),verification_sha256=sha(cpu_path),entry_changed_files=sorted(entry_changes),
    dt_changed_files=sorted(dt_changes),unchanged_actor_sha256=sha(owner/'verl/workers/actor/dp_actor.py'),
    unchanged_host_worker_sha256=sha(owner/'verl/workers/fsdp_workers.py'),checkpoint=str(checkpoint)),indent=2))
PY
'''
    return script.replace('@ENTRY@',ENTRY).replace('@ROOT@',repr(ROOT)).replace(
        '@PAYLOAD@',repr(json.dumps(payload))).replace('@SOURCE@',repr(SOURCE)).replace('@NAME@',repr(NAME))


def main():
    remote(preparation_script(build_payload()))


if __name__=='__main__':
    main()
