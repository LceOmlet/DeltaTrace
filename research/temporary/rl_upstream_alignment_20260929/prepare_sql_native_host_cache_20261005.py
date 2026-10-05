"""Prepare SQL's frozen scope candidate with the accepted host-cache boundary.

Preparation only.  The old candidate, job manifests and numerical release stay
untouched.  Reuse the exact accepted resource patch and CPU tests; this script
does not launch a model, training job, engine, allocator probe or GPU test.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, SCP, SSH, remote


NAME = "sql-native-host-cache-20261005-v2"
PRIOR_PREPARED_SHA = "ecf31cc9b98f6a4b73c89a2e39bed3e68ba3ca040610009689547994df607b06"
RESOURCE_COMMIT = "6781bdde60b4873f8336e12dd0e6c2ea1fca22b3"
RESOURCE_RECORDER_COMMIT = "e945acb06a11b1417855efd1eeeda80f2e28d2b1"
WORKER_BEFORE_SHA = "807e51856f9990d408fb2c33998fdcf9b0cbf3f230178677d6a5d2ecc0b7cc0d"
WORKER_AFTER_SHA = "e5eb4afc42f10fb4608b3ac43046c906d6a2d21a5387ee02e176bc395f1c6f39"
PATCH_SHA = "33f61565d271cf9a4a10f80655b44f921e02fad42e21993e36d5efbc0fa361b3"
HOST_TEST_SHA = "7af34c65c2f24d4b6e2e726a5ea14065c883974659213c25f1b5a8f214a9ff4d"
LAUNCHER_SHA = "51a854aed3f0be00fa56ed4d12b0d8702b3c706b46eb337c7a979b23b7ee0980"
ACTOR_SHA = "1f862e8bbdaad6fa116d0670772ad41269529a3a1e4a5b1eb383352d0372e9bd"
TASK_CONFIG_SHA = "69af7f858cd25f2f76f8ccc4c6831d0e9eb51dc072817489257bcb84cd0faafd"
NUMERICAL_LOCK_SHA = "b5d5608ea76d941aca14314eb23811311b189e356cdcc0e478f3282a94c325c5"
RESOURCE_KEY = "+ray_init.runtime_env.env_vars.VERL_RELEASE_UNUSED_HOST_CACHE"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob(name: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{RESOURCE_COMMIT}:{name}"], cwd=REPO)


def build_payload() -> dict:
    """Read pinned bytes only; do not publish other current-worktree files."""
    before = (AUDIT / "native-host-cache-owner-before.py").read_bytes()
    after = (AUDIT / "native-host-cache-owner-patched.py").read_bytes()
    patch = git_blob("experiments/rl/patch_native_host_cache.py")
    tests = git_blob("experiments/rl/test_native_host_cache_boundary.py")
    assert digest(before) == WORKER_BEFORE_SHA
    assert digest(after) == WORKER_AFTER_SHA
    assert digest(patch) == PATCH_SHA and digest(tests) == HOST_TEST_SHA
    # Execute the existing patch module under a non-main name, not a copied
    # patch implementation. Its exact output must be the accepted owner bytes.
    namespace = {"__name__": "_accepted_native_host_patch", "__file__": "patch_native_host_cache.py"}
    exec(compile(patch, namespace["__file__"], "exec"), namespace)
    assert namespace["patch_source"](before.decode("utf-8")).encode("utf-8") == after
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    return dict(
        original_worker=base64.b64encode(before).decode("ascii"),
        patched_worker=base64.b64encode(after).decode("ascii"),
        patch_source=base64.b64encode(patch).decode("ascii"),
        host_tests=base64.b64encode(tests).decode("ascii"),
        preparation_repository_commit=revision,
        preparation_script_sha256=digest(Path(__file__).read_bytes()),
    )


def remote_script(payload: dict) -> str:
    script = r'''set -e
source @ENTRY@/metax-entry.env.sh
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=-1
export MACA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import ast,base64,hashlib,json,os,runpy,shutil,subprocess,sys,time,xml.etree.ElementTree as ET
root=Path(@ROOT@)
payload=json.loads(@PAYLOAD@)
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
decode=lambda k:base64.b64decode(payload[k],validate=True)
resource_key=@RESOURCE_KEY@
prior_path=root/'receipts/owner-b8-dispatch-20260930/sql-rollout-scope/prepared.json'
assert sha(prior_path)==@PRIOR_SHA@
prior=read(prior_path)
assert prior['status']=='prepared_not_launched'
assert prior['entry']==str(root/'candidates/sql-rollout-scope-20261001/entry')
assert prior['verl_root']==str(root/'candidates/sql-rollout-scope-20261001/verl')
assert prior['dt_root']==str(root/'releases/c9cd147')
entry_before=Path(prior['entry']);owner_before=Path(prior['verl_root'])
for name,h in prior['entry_sha256'].items():assert sha(entry_before/name)==h,name
for name,h in prior['owner_sha256'].items():assert sha(owner_before/name)==h,name
for name,h in prior['dt_source_sha256'].items():assert sha(Path(prior['dt_root'])/name)==h,name
assert sha(entry_before/'verified_runtime.json')==@LOCK_SHA@
assert read(entry_before/'verified_runtime.json')['dt_source_sha256']==prior['dt_source_sha256']
assert sha(entry_before/'owner_environment_configs.json')==@TASK_CONFIG_SHA@
assert sha(entry_before/'launch_sql_native.py')==@LAUNCHER_SHA@
assert sha(owner_before/'verl/workers/fsdp_workers.py')==@BEFORE_SHA@
assert sha(owner_before/'verl/workers/actor/dp_actor.py')==@ACTOR_SHA@

# Original manifests are read for source/config identity only. No process or
# device is inspected, stopped, rebound or launched by this preparation.
manifest_paths=[root/n for n in ('active-training.json','active-source.json')]
manifest_before={str(p):sha(p) for p in manifest_paths}
active=read(root/'active-training.json')
job=next(j for j in active['jobs'] if j['task']=='SkyRL-SQL')
assert job['pid']==prior['prior_driver_pid']==552842
assert job['entry']==prior['prior_entry'] and job['verl_root']==prior['prior_verl_root']
assert job['dt_root']==prior['dt_root'] and job['checkpoints']==prior['future_checkpoint_root']
old_launch=read(Path(job['output'])/'launch.json')
prior_source=Path(prior['prior_source_receipt'])
assert sha(prior_source)==prior['prior_source_sha256']
assert not list(Path(job['checkpoints']).rglob('latest_checkpointed_iteration.txt'))

def tree_bytes(directory):
    return {str(p.relative_to(directory)):({'symlink':os.readlink(p)} if p.is_symlink()
            else {'sha256':sha(p)}) for p in directory.rglob('*') if p.is_symlink() or p.is_file()}

base=root/'candidates'/@NAME@
assert not base.exists(),'Immutable candidate already exists; inspect it instead of overwriting or retrying'
old_entry_tree=tree_bytes(entry_before);old_owner_tree=tree_bytes(owner_before)
base.mkdir()
created_candidate=True
entry=base/'entry';owner=base/'verl';audit=base/'audit';audit.mkdir()
shutil.copytree(entry_before,entry,symlinks=True)
shutil.copytree(owner_before,owner,symlinks=True)
assert tree_bytes(entry)==old_entry_tree and tree_bytes(owner)==old_owner_tree
(audit/'prior-prepared.json').write_bytes(prior_path.read_bytes())
(audit/'prior-source.json').write_bytes(prior_source.read_bytes())

patch_path=audit/'patch_native_host_cache.py'
test_path=audit/'test_native_host_cache_boundary.py'
original_path=audit/'fsdp_workers.before.py'
for path,key,h in [(patch_path,'patch_source',@PATCH_SHA@),
                   (test_path,'host_tests',@HOST_TEST_SHA@),
                   (original_path,'original_worker',@BEFORE_SHA@)]:
    path.write_bytes(decode(key));assert sha(path)==h,path
assert original_path.read_bytes()==(owner/'verl/workers/fsdp_workers.py').read_bytes()
patched=decode('patched_worker')
assert hashlib.sha256(patched).hexdigest()==@AFTER_SHA@
patch_namespace=runpy.run_path(str(patch_path),run_name='_accepted_native_host_patch')
assert patch_namespace['patch_source'](original_path.read_text()).encode()==patched
(owner/'verl/workers/fsdp_workers.py').write_bytes(patched)

launcher=entry/'launch_sql_native.py';launcher_before=launcher.read_bytes()
# The pinned launcher SHA contains CRLF here; preserve all original bytes.
anchor=b"    options = {**runtime_options(),\r\n"
addition=b"        '+ray_init.runtime_env.env_vars.VERL_RELEASE_UNUSED_HOST_CACHE': '1',\r\n"
assert launcher_before.count(anchor)==1
offset=launcher_before.index(anchor)+len(anchor)
launcher_after=launcher_before[:offset]+addition+launcher_before[offset:]
ast.parse(launcher_after)
assert launcher_after[:offset]==launcher_before[:offset]
assert launcher_after[offset+len(addition):]==launcher_before[offset:]
(audit/'launch_sql_native.before-resource-config.py').write_bytes(launcher_before)
launcher.write_bytes(launcher_after)
new_entry_tree=tree_bytes(entry);new_owner_tree=tree_bytes(owner)
assert {n for n in old_entry_tree.keys()|new_entry_tree.keys()
        if old_entry_tree.get(n)!=new_entry_tree.get(n)}=={'launch_sql_native.py'}
assert {n for n in old_owner_tree.keys()|new_owner_tree.keys()
        if old_owner_tree.get(n)!=new_owner_tree.get(n)}=={'verl/workers/fsdp_workers.py'}

env=os.environ.copy()
env.update(VERL_ROOT=str(owner),DT_ENTRY_ROOT=str(entry),DT_ROOT=prior['dt_root'],
    SCOPE_OWNER_ROOT=prior['prior_verl_root'],CUDA_VISIBLE_DEVICES='-1',MACA_VISIBLE_DEVICES='-1',
    PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
env['PYTHONPATH']=':'.join([str(entry),str(owner),prior['dt_root']+'/experiments/rl',
    prior['dt_root'],env.get('PYTHONPATH','')])
nodes=['test_two_rank_config_satisfies_native_validator[SkyRL-SQL-dt-formal]',
       'test_native_resume_keeps_original_workload']
cpu_xml=audit/'cpu-tests.xml'
cpu_argv=[env['VENV_PYTHON'],'-m','pytest','-q','--import-mode=importlib','-p','no:cacheprovider',
    *[str(entry/'test_owner_entry_launch.py')+'::'+n for n in nodes],
    str(entry/'test_distributed_credit.py'),str(entry/'test_owner_rollout_scope.py'),
    '--junitxml='+str(cpu_xml)]
print('[SQL host preparation] original 16 CPU owner checks',flush=True)
subprocess.run(cpu_argv,env=env,cwd=entry,check=True)
suites=list(ET.parse(cpu_xml).getroot().iter('testsuite'))
assert sum(int(s.get('tests','0')) for s in suites)==16
assert all(int(s.get(k,'0'))==0 for s in suites for k in ('errors','failures','skipped'))
host_log=audit/'host-boundary-cpu.txt'
host_argv=[env['VENV_PYTHON'],str(test_path),'--owner',str(original_path),'-v']
print('[SQL host preparation] original 3 resource-boundary CPU checks',flush=True)
with host_log.open('wb') as log:
    subprocess.run(host_argv,env=env,cwd=audit,stdout=log,stderr=subprocess.STDOUT,check=True)
assert 'Ran 3 tests' in host_log.read_text() and '\nOK\n' in host_log.read_text()

# Call only the frozen configuration assembler; not its __main__, a model or
# the trainer. Original workload, loss and sampling settings must stay fixed.
os.environ.update({k:env[k] for k in ('VERL_ROOT','DT_ENTRY_ROOT','DT_ROOT')})
sys.path[:0]=[str(entry),str(owner),prior['dt_root']+'/experiments/rl',prior['dt_root']]
from types import SimpleNamespace
arguments=SimpleNamespace(method='dt',phase='formal',
    data=str(Path(old_launch['options']['data.train_files']).parent),output=job['output'])
before=runpy.run_path(str(entry_before/'launch_sql_native.py'),run_name='_old_config_only')['command'](arguments)[1]
after=runpy.run_path(str(launcher),run_name='_new_config_only')['command'](arguments)[1]
path_keys={'data.custom_cls.path':('owner_task_dataset.py'),
           '+data.sql_chat_template':('qwen3_acc_thinking.jinja2')}
old_changes={k:[old_launch['options'].get(k),before.get(k)]
    for k in old_launch['options'].keys()|before.keys() if old_launch['options'].get(k)!=before.get(k)}
assert old_changes==prior['configuration_changes']
changes={k:[before.get(k),after.get(k)] for k in before.keys()|after.keys()
    if before.get(k)!=after.get(k)}
assert set(changes)==set(path_keys)|{resource_key}
assert changes[resource_key]==[None,'1']
for key,name in path_keys.items():
    assert changes[key]==[str(entry_before/name),str(entry/name)]
    assert sha(entry_before/name)==sha(entry/name)
assert {k:after[k] for k in ('actor_rollout_ref.model.lora_rank','actor_rollout_ref.model.lora_alpha',
    'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu','actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu',
    'actor_rollout_ref.rollout.max_model_len')}==dict(zip((
    'actor_rollout_ref.model.lora_rank','actor_rollout_ref.model.lora_alpha',
    'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu','actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu',
    'actor_rollout_ref.rollout.max_model_len'),(8,16,4,4,32768)))
assert tree_bytes(entry)==new_entry_tree and tree_bytes(owner)==new_owner_tree
assert tree_bytes(entry_before)==old_entry_tree and tree_bytes(owner_before)==old_owner_tree
assert {str(p):sha(p) for p in manifest_paths}==manifest_before
assert sha(prior_path)==@PRIOR_SHA@
for name,h in prior['dt_source_sha256'].items():assert sha(Path(prior['dt_root'])/name)==h,name

record=dict(prepared_unix=time.time(),status='prepared_not_launched',entry=str(entry),verl_root=str(owner),
    dt_root=prior['dt_root'],prior_driver_pid=job['pid'],prior_entry=job['entry'],prior_verl_root=job['verl_root'],
    future_checkpoint_root=job['checkpoints'],resumed_checkpoint=None,completed_checkpoints=[],
    prior_prepared_receipt=str(prior_path),prior_prepared_sha256=@PRIOR_SHA@,
    prior_source_receipt=str(prior_source),prior_source_sha256=sha(prior_source),
    supersedes_for_future_start='sql-rollout-scope-20261001',
    preparation_repository_commit=payload['preparation_repository_commit'],
    preparation_script_sha256=payload['preparation_script_sha256'],
    resource_patch_commit=@RESOURCE_COMMIT@,resource_recorder_commit=@RECORDER_COMMIT@,
    resource_environment={'VERL_RELEASE_UNUSED_HOST_CACHE':'1'},
    resource_runtime_env={resource_key:'1'},configuration_changes=changes,prior_budget=job['budget'],
    entry_sha256={n:sha(entry/n) for n in prior['entry_sha256']},
    owner_sha256={n:sha(owner/n) for n in prior['owner_sha256']},dt_source_sha256=prior['dt_source_sha256'],
    unchanged_actor_sha256=@ACTOR_SHA@,unchanged_task_config_sha256=@TASK_CONFIG_SHA@,
    unchanged_numerical_lock_sha256=@LOCK_SHA@,
    resource_sources={p.name:dict(path=str(p),sha256=sha(p)) for p in (patch_path,test_path,original_path)},
    resource_patch_proof=dict(before_sha256=@BEFORE_SHA@,after_sha256=@AFTER_SHA@,
        accepted_patch_output_exact=True,scope='Existing patch_source on actual saved owner bytes; not numerical verification'),
    cpu_tests=dict(path=str(cpu_xml),sha256=sha(cpu_xml),passed=16,argv=cpu_argv,
        scope='Original configuration validator, resume options, return transport, partition ordering and rollout context lifecycle; no model/GPU numerical test'),
    host_boundary_cpu=dict(path=str(host_log),sha256=sha(host_log),passed=3,argv=host_argv,
        scope='Original actual-owner AST/default/failure cleanup tests; allocator behavior remains covered by the prior accepted receipt'),
    old_manifests_unchanged=manifest_before,
    scope='Prepared-only frozen SQL scope plus accepted native phase-end host release. No launch, process/device change, checkpoint loading, new numerical tolerance or training-health claim.')
(base/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ('status','entry','verl_root','dt_root','configuration_changes')},indent=2),flush=True)
PY
'''
    values = {
        "@ENTRY@": ENTRY, "@ROOT@": repr(ROOT), "@NAME@": repr(NAME),
        "@PAYLOAD@": repr(json.dumps(payload)), "@RESOURCE_KEY@": repr(RESOURCE_KEY),
        "@PRIOR_SHA@": repr(PRIOR_PREPARED_SHA), "@LOCK_SHA@": repr(NUMERICAL_LOCK_SHA),
        "@TASK_CONFIG_SHA@": repr(TASK_CONFIG_SHA), "@LAUNCHER_SHA@": repr(LAUNCHER_SHA),
        "@BEFORE_SHA@": repr(WORKER_BEFORE_SHA), "@AFTER_SHA@": repr(WORKER_AFTER_SHA),
        "@ACTOR_SHA@": repr(ACTOR_SHA), "@PATCH_SHA@": repr(PATCH_SHA),
        "@HOST_TEST_SHA@": repr(HOST_TEST_SHA), "@RESOURCE_COMMIT@": repr(RESOURCE_COMMIT),
        "@RECORDER_COMMIT@": repr(RESOURCE_RECORDER_COMMIT),
    }
    for key, value in values.items():
        script = script.replace(key, value)
    # Preserve a failed *new* candidate for inspection; never write into an
    # existing candidate rejected by the immutable-directory guard.
    marker = '"$VENV_PYTHON" - <<\'PY\'\n'
    header, body = script.split(marker, 1)
    assert body.endswith("\nPY\n")
    body = body[:-4]
    failure_handler = '''except BaseException as error:
    if locals().get('created_candidate', False):
        import hashlib,json,time,traceback
        from pathlib import Path
        candidate=Path(@CANDIDATE@)
        if not (candidate/'prepared.json').exists():
            failure=candidate/'preparation-failure.json'
            with failure.open('x') as stream:
                json.dump(dict(observed_unix=time.time(),status='preparation_failed_not_launched',
                    preparation_script_sha256=payload['preparation_script_sha256'],
                    error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
                    scope='Only this new isolated candidate was created; no model/job/device or active manifest changed.'),
                    stream,indent=2)
                stream.write('\\n')
    raise
'''.replace("@CANDIDATE@", repr(ROOT + "/candidates/" + NAME))
    wrapped = "created_candidate=False\ntry:\n" + textwrap.indent(body, "    ") + "\n" + failure_handler
    return header + marker + wrapped + "PY\n"


def main() -> None:
    destination = AUDIT / NAME
    assert not destination.exists(), "Inspect an existing local receipt instead of overwriting it"
    remote(remote_script(build_payload()))
    destination.mkdir(exist_ok=False)
    base = ROOT + "/candidates/" + NAME
    for name, target in [("prepared.json", "prepared.json"),
                         ("audit/cpu-tests.xml", "cpu-tests.xml"),
                         ("audit/host-boundary-cpu.txt", "host-boundary-cpu.txt")]:
        subprocess.run(SCP + [f"{SSH[-1]}:{base}/{name}", str(destination / target)], check=True)


if __name__ == "__main__":
    main()
