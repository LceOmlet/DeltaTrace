"""Stage the existing base-model B8 probe and isolated DI files; never launch it."""
import ast
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[2]
ROW = HERE.parent / 'native-representation-candidate'
REPO = next(p for p in HERE.parents if (p/'experiments/rl/PLAN.md').exists())
sys.path.insert(0, str(AUDIT))
from stage_environment_entry import ENTRY, ROOT, SCP, SSH, remote

PARENT = ROOT + '/receipts/owner-b8-dispatch-20260930/native-prefix-reuse-hot-phase-offset84-20261007-e241e91-current-base'
OUT = ROOT + '/candidates/appworld-row-cuts-finite-20261007-v1/combined-capacity-b8-v1'
LIBRARY = ROOT + '/candidates/appworld-row-cuts-finite-20261007-v1/libfinite_row_query_starts.so'
LIBRARY_SHA = '4f42c391055afec0a0fee9ee698c0820163ff413e42f1c4909b2961ce81e5157'
PARENT_PREPARED_SHA = '7f2252d8f5ba0456aa1f77b6a2e56e681391f876996fbd1c274d0eab4608befd'
FILES = {
    'diagnose_native_prefix_leases.py': (HERE/'diagnose_combined_storage_row_capacity.py', '9b1d1943043ecfea07802651a5ff94f1b5d318fc2e99f35eb6bf0094316381a8'),
    'verify_dt_context_capacity.py': (REPO/'experiments/rl/verify_dt_context_capacity.py', '8570acfb3690d751e2c378d9dabe27b04911ec1bb8bd0cdab52708300d0e75bd'),
    'qwen35_dense_finite_runner_row_candidate.py': (ROW/'candidate/qwen35_dense_finite_runner.py', '5f14bb3cdb491e4d5b3b531607e00936bae76e055e2286ed60e096c6c5d2e555'),
    'vendor_fa_finite_bf16_d256_row_candidate.py': (ROW.parent/'candidate/vendor_fa_finite_bf16_d256.py', '3e1d61037be22a1cc826b004d49f854e3c246a149642a4f9167c204181f34089'),
    'qwen35_native_prefix_artifacts.py': (HERE/'candidate/qwen35_native_prefix_artifacts.py', '37a86074f430efd837c878b5409ce22ff3aac5ebdc984936ea5be5647d7b97d4'),
    'native_prefix_leases.py': (HERE/'candidate/native_prefix_leases.py', 'b94756147cc6f8e59fb39baa1c2737aa655b0d9c6b87091969a65c9071ad0852'),
    'qwen35_answer_finite.py': (ROW/'candidate/qwen35_answer_finite.py', 'd47333ea68fb7a332e7d1dce7913c989d875ea262dfe49cfa4f20f7c35ebe03e'),
}


REMOTE_PREPARE = r'''
import ast,hashlib,importlib.machinery,json,os,pathlib,shutil,tarfile,time
root=pathlib.Path(@ROOT@);parent=pathlib.Path(@PARENT@);out=pathlib.Path(@OUT@)
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
assert out.resolve().is_relative_to((root/'candidates/appworld-row-cuts-finite-20261007-v1').resolve())
assert not (out/'prepared.json').exists() and not (out/'job.json').exists()
assert sha(parent/'prepared.json')==@PARENT_SHA@
previous=json.loads((parent/'prepared.json').read_bytes())
assert previous['base_model_initialization'] is True and previous['checkpoint'] is None
assert previous['selected_observation']['offset']==84 and previous['rows_per_rank']==4
assert previous['native_backward_reference'] is False
assert previous['configuration']==@CONFIGURATION@
assert (parent/'result.json').exists(), 'Reuse only the completed base-model comparison'
formal=previous['current_formal_owner']
for path,expected in formal['source_bindings'].items():
 assert sha(path)==expected,path
baseline_interfaces={
 'answer':dict(path=str(pathlib.Path(formal['dt_root'])/'clean/qwen35/qwen35_answer_finite.py'),sha256='9819cf34333d2ce67224cd49d404d21192040311001429b5c97982d76c727526'),
 'artifact':dict(path=str(pathlib.Path(formal['dt_root'])/'clean/qwen35/qwen35_native_prefix_artifacts.py'),sha256='06fda7843c4120cb1406b671f06cb19f29921b61b5d524dc28539550446aef16'),
 'lease':dict(path=str(pathlib.Path(formal['entry'])/'native_prefix_leases.py'),sha256='d5b539c7ffb8a431374cf79f4b995ef6f4138c3e6e289890409b6b79dbad393e'),
 'runner':dict(path=str(pathlib.Path(formal['dt_root'])/'clean/qwen35/qwen35_dense_finite_runner.py'),sha256='e9c7576486f742c26f895cd4078891a98d94c189e84a6e565fc8042a74aabcab')}
for item in baseline_interfaces.values():assert sha(item['path'])==item['sha256'],item['path']
manifest_raw=(root/'active-training.json').read_bytes()
app=next(j for j in json.loads(manifest_raw)['jobs'] if j['task']=='AppWorld')
assert app['entry']==formal['entry'] and app['verl_root']==formal['verl_root']
source_path=pathlib.Path(app['output'])/'source.json';source_raw=source_path.read_bytes();active_source=json.loads(source_raw)
assert active_source['dt_root']==formal['dt_root']
library=pathlib.Path(@LIBRARY@);assert sha(library)==@LIBRARY_SHA@
copies={
 'verify_native_prefix_artifacts.py':next(v for k,v in previous['source_files'].items() if k.endswith('/verify_native_prefix_artifacts.py')),
 'native-launch-options.json':previous['configuration']['official_launch_sha256'],
 **{pathlib.Path(k).name:v for k,v in previous['literal_request_files'].items()}}
assert copies['verify_native_prefix_artifacts.py']=='22bc698c7ae848581c1572095029c17e3542209589a8faa8d12109b3337ec7e8'
for name,expected in copies.items():
 assert sha(parent/name)==expected,(name,'original bytes changed')
 shutil.copy2(parent/name,out/name)
 assert sha(out/name)==expected,name
shutil.copy2(parent/'prepared.json',out/'parent-prepared.json')
overlay=out/'stage-overlay.tar'
with tarfile.open(overlay) as archive:
 members=archive.getmembers();assert {item.name for item in members}==set(@OVERLAY_NAMES@)
 for item in members:
  assert item.isfile() and pathlib.PurePosixPath(item.name).name==item.name
  raw=archive.extractfile(item).read();(out/item.name).write_bytes(raw)
for name,expected in @OVERLAY_HASHES@.items():assert sha(out/name)==expected,name
assert not (out/'qwen35_dense_finite_runner.py').exists()
assert not (out/'vendor_fa_finite_bf16_d256.py').exists()
for path in out.glob('*.py'):ast.parse(path.read_bytes(),filename=str(path))
# Original linked-owner seam from run_native_prefix_reuse_workload.py418-445.
# producer prepends DT_ROOT/clean/qwen35; make that original import rule select
# only these two isolated candidate interfaces instead of the formal owner.
formal_dt=pathlib.Path(formal['dt_root']);linked=out/'isolated-row-owners'/'deltatrace';linked.mkdir(parents=True)
link_records=[]
def link_children(old,new,exceptions):
 for path in old.iterdir():
  if path.name not in exceptions and path.name!='__pycache__':
   destination=new/path.name;destination.symlink_to(path,target_is_directory=path.is_dir())
   link_records.append(dict(path=str(destination),target=str(path),directory=path.is_dir()))
link_children(formal_dt,linked,{'clean'})
(linked/'clean').mkdir();link_children(formal_dt/'clean',linked/'clean',{'qwen35'})
(linked/'clean/qwen35').mkdir()
link_children(formal_dt/'clean/qwen35',linked/'clean/qwen35',{'qwen35_answer_finite.py','qwen35_native_prefix_artifacts.py'})
link_overrides={}
for name in ('qwen35_answer_finite.py','qwen35_native_prefix_artifacts.py'):
 destination=linked/'clean/qwen35'/name;destination.symlink_to(out/name)
 assert sha(destination)==sha(out/name)
 link_overrides[name]=dict(path=str(destination),target=str(out/name),sha256=sha(destination))
assert sha(linked/'clean/qwen35/qwen35_dense_finite_runner.py')==baseline_interfaces['runner']['sha256']

options=json.loads((out/'native-launch-options.json').read_bytes())
assert options['actor_rollout_ref.model.lora_rank']==8 and options['actor_rollout_ref.model.lora_alpha']==16
assert options['actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu']==4
assert options['actor_rollout_ref.model.path']=='/mnt/si0021787ci2/default/models/Qwen3.5-9B'
env=dict(os.environ)
for key in ('MACA_VISIBLE_DEVICES','RAY_ADDRESS','RAY_TMPDIR','DT_PREFIX_OWNER_SOURCE','DT_PREFIX_ARTIFACT_SOURCE',
 'DT_PREFIX_CHECKPOINT','DT_PREFIX_NATIVE_BACKWARD','DT_PREFIX_HOT_PROFILE','DT_PREFIX_LEDGER_ONLY',
 'DT_PREFIX_PROJECTION_INPUTS','DT_PREFIX_REVERSE_PREFETCH','DT_PREFIX_ROOT_CAPTURE_INVENTORY',
 'DT_PREFIX_ROOT_TAPE','DT_PREFIX_ROOT_TAPE_CPU','DT_PREFIX_ROOT_TAPE_GDN0','DT_PREFIX_ROOT_TAPE_FA3',
 'DT_PREFIX_ROOT_TAPE_HOT','DT_PREFIX_COMPONENT_LAYER','DT_PREFIX_LEASE_COMPONENT_DIAGNOSTIC',
 'DT_PREFIX_NATIVE_CONV_INITIAL_STATES','DT_PREFIX_NATIVE_CONV_CAPACITY','DT_PREFIX_BOUNDARY_ROW_STORAGE'):
 env.pop(key,None)
numerical=previous['configuration']['numerical_environment'];assert sha(numerical['path'])==numerical['sha256']
shell_environment=root/'receipts/environment-only-20260930/entry/metax-entry.env.sh'
assert sha(shell_environment)=='beb9c001cc57db5450ad364b294f1d6ada3f72a2fbe0869d9892507a264bd076'
env.update(VERL_ROOT=formal['verl_root'],DT_ROOT=str(linked),DT_ENTRY_ROOT=formal['entry'],
 LOOP_ROOT=active_source['loop_root'],APPWORLD_ROOT=str(root/'receipts/environment-only-20260930/loop-entry/appworld-root'),
 DT_ENVIRONMENT_JSON=numerical['path'],CUDA_VISIBLE_DEVICES='2,3',DT_PREFIX_PROBE_ROOT=str(out),
 DT_PREFIX_DT_LEASE_DIAGNOSTIC='1',DT_PREFIX_DIAGNOSTIC_ROWS='4',DT_PREFIX_DIAGNOSTIC_OFFSET='84',
 DT_PREFIX_PHASE_ONLY='1',DT_PREFIX_PHASE_WARM='1',DT_PREFIX_CURRENT_FORMAL_OWNER='1',
 DT_PREFIX_BASE_MODEL_PREFETCH='1',DT_PREFIX_INDIVIDUAL_ROW_CANDIDATE='1',
 DT_TASK='AppWorld',DT_MAX_STEPS=str(options['env.max_steps']),DT_MAX_LENGTH='32768',
 DT_SAMPLING_JSON=json.dumps(dict(temperature=options['actor_rollout_ref.rollout.temperature'],max_tokens=options['data.max_response_length'])),
 RAY_TMPDIR='/tmp/dtrcapb1')
inherited_pythonpath=':'.join(part.replace(str(formal_dt),str(linked),1)
 if part==str(formal_dt) or part.startswith(str(formal_dt)+'/') else part
 for part in formal['pythonpath'].split(':'))
env['PYTHONPATH']=':'.join((str(out),str(linked/'clean/qwen35'),inherited_pythonpath))
# Reproduce the existing producer's prepend rule without importing a model.
official_root=env.get('DT_OFFICIAL_ROOT') or json.loads(pathlib.Path(numerical['path']).read_bytes())['qwen35']['official_root']
producer_search=[str(linked),str(official_root),str(linked/'clean/qwen35'),*env['PYTHONPATH'].split(':')]
import_resolution={}
for name,expected in (('qwen35_answer_finite',out/'qwen35_answer_finite.py'),('qwen35_native_prefix_artifacts',out/'qwen35_native_prefix_artifacts.py'),('qwen35_dense_finite_runner',formal_dt/'clean/qwen35/qwen35_dense_finite_runner.py')):
 spec=importlib.machinery.PathFinder.find_spec(name,producer_search)
 assert spec and pathlib.Path(spec.origin).resolve()==expected.resolve(),(name,None if spec is None else spec.origin)
 import_resolution[name]=dict(path=spec.origin,resolved_path=str(pathlib.Path(spec.origin).resolve()),sha256=sha(spec.origin))
assert env['VENV_PYTHON']=='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python'
env_path=out/'run-env.json'
with env_path.open('x') as stream:json.dump(env,stream,indent=2);stream.write('\n')
env_path.chmod(0o600)
owners={key:dict(path=str(out/name),sha256=sha(out/name)) for key,name in @OWNER_NAMES@.items()}
owners['finite_library']=dict(path=str(library),sha256=sha(library))
prepared=dict(role='Prepared only; unchanged original base-model B8 setup with isolated row-cut DI; no launch or numerical acceptance',
 prepared_only=True,gpu_probe_started=False,devices=[2,3],rows_per_rank=4,original_bank_rows_per_rank=88,
 checkpoint=None,base_model_initialization=True,native_backward_reference=False,instrumented_hot_profile=False,isolated_observation_scope='Two sequential stages in one original initialized actor: actual offset84 compact+row cold/warm, then existing exact32768/512 compact+row cold/warm. No operator observers or reference forwards.',combined_storage_row_capacity_entry=True,previous_uncompressed_row_vectors=@PREVIOUS_ROW_VECTOR_FILES@,
 selected_observation=previous['selected_observation'],configuration=previous['configuration'],
 current_formal_owner=formal,parent_prepared=dict(path=str(parent/'prepared.json'),sha256=sha(parent/'prepared.json')),
 active_formal_identity=dict(pid=app['pid'],devices=app['devices'],source_path=str(source_path),source_sha256=hashlib.sha256(source_raw).hexdigest(),manifest_sha256=hashlib.sha256(manifest_raw).hexdigest()),
 individual_row_candidate=owners,literal_request_files={str(out/name):expected for name,expected in copies.items() if name.startswith('actual-requests-rank')},
 original_baseline_interfaces=baseline_interfaces,
 isolated_link_owner=dict(dt_root=str(linked),original_formal_dt_root=str(formal_dt),overrides=link_overrides,original_link_count=len(link_records),import_resolution=import_resolution,scope='Prepared-only linked diagnostic owner; formal sources and runtime remain unchanged'),
 original_setup=dict(path=str(out/'verify_native_prefix_artifacts.py'),sha256=sha(out/'verify_native_prefix_artifacts.py'),bytewise_preserved=True),
 source_files={str(path):sha(path) for path in out.glob('*.py')},
 run_environment_file=dict(path=str(env_path),sha256=sha(env_path),mode='0600',values_printed=False),
 environment_provenance=dict(shell_path=str(shell_environment),shell_sha256=sha(shell_environment),
  original_hot_environment_file_names=[p.name for p in parent.glob('*env*') if p.is_file()],
  method='Original provisioned shell environment plus the current-base hot-probe formal paths, numerical-environment hash and native-launch options; only isolated linked DT_ROOT/out/ray/device/profiler/individual-row dispatch differ. No checkpoint-probe environment is used.'),
 initialized_optimizer_scope='Original actor-role init constructs official optimizer; probe performs no optimizer step/update, no actor backward and no checkpoint load.',
 observed_unix=time.time())
cpu_contracts=root/'candidates/appworld-row-cuts-finite-20261007-v1/combined-storage-row-v1/cpu-contracts.json'
assert sha(cpu_contracts)=='a3b469124e7384d6287a775d37ec4d70702aabb7d1e13ed88c03edf51fa29d01'
prepared['combined_storage_row_cpu_contracts']=dict(path=str(cpu_contracts),sha256=sha(cpu_contracts))
prepared['sequential_stages']=['actual_b8','exact32768_b8']
for item in prepared['previous_uncompressed_row_vectors'].values():assert sha(item['path'])==item['sha256']
(out/'prepared.json').write_text(json.dumps(prepared,indent=2)+'\n')
for stage_name in prepared['sequential_stages']:
 stage_out=out/stage_name;stage_out.mkdir()
 for name in (*[pathlib.Path(p).name for p in prepared['literal_request_files']], 'prepared.json','native-launch-options.json'):
  shutil.copy2(out/name,stage_out/name)
  assert sha(stage_out/name)==sha(out/name)

print(json.dumps(dict(out=str(out),prepared_sha256=sha(out/'prepared.json'),prepared_only=True,gpu_probe_started=False,
 original_setup=prepared['original_setup'],individual_row_candidate=owners,literal_request_files=prepared['literal_request_files'],
 original_baseline_interfaces=baseline_interfaces,isolated_link_owner=prepared['isolated_link_owner'],environment_provenance=prepared['environment_provenance'],
 run_environment_file=prepared['run_environment_file'],launch_entry=str(out/'launch_real_b8_probe.py'))))
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline_path = AUDIT / 'appworld-efficiency-20261007/current-base-hot-profile-v1/prepared.json'
    assert digest(baseline_path) == PARENT_PREPARED_SHA
    previous = json.loads(baseline_path.read_bytes())
    files = dict(FILES)
    launcher = ROW / 'launch_real_b8_probe_v2.py'
    files['launch_real_b8_probe.py'] = (launcher, digest(launcher))
    for name, (path, expected) in files.items():
        assert digest(path) == expected, (name, 'Candidate changed; review instead of mixing versions')
    bundle = HERE / 'combined-capacity-b8-stage-overlay.tar'
    with tarfile.open(bundle, 'w') as archive:
        for name, (path, _) in files.items():
            archive.add(path, arcname=name)
    remote('test ! -e ' + shlex.quote(OUT + '/prepared.json') + ' && test ! -e ' +
           shlex.quote(OUT + '/job.json') + ' && mkdir -p ' + shlex.quote(OUT) + '\n')
    subprocess.run(SCP + [str(bundle), SSH[-1] + ':' + OUT + '/stage-overlay.tar'], check=True)
    replacements = {
        '@ROOT@': repr(ROOT), '@PARENT@': repr(PARENT), '@OUT@': repr(OUT),
        '@LIBRARY@': repr(LIBRARY), '@LIBRARY_SHA@': repr(LIBRARY_SHA),
        '@PARENT_SHA@': repr(PARENT_PREPARED_SHA),
        '@CONFIGURATION@': repr(previous['configuration']),
        '@OVERLAY_NAMES@': repr(list(files)),
        '@OVERLAY_HASHES@': repr({name: expected for name, (_, expected) in files.items()}),
        '@PREVIOUS_ROW_VECTOR_FILES@': "{'0': {'path': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/appworld-row-cuts-finite-20261007-v1/real-b8-v3/prefix-lease-vectors-rank0.pt', 'sha256': '04a7723ffba52bed49ef1b655f4f66ef8c8a3c32c5d6411f1d7a115012fcb306'}, '1': {'path': '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/appworld-row-cuts-finite-20261007-v1/real-b8-v3/prefix-lease-vectors-rank1.pt', 'sha256': '9efae60d0a1c5ed73195d6a8153393a78d5f2f8e9bb6dd492e6576e783af43ac'}}",
        '@OWNER_NAMES@': repr(dict(
            runner='qwen35_dense_finite_runner_row_candidate.py',
            finite_wrapper='vendor_fa_finite_bf16_d256_row_candidate.py',
            artifact='qwen35_native_prefix_artifacts.py', lease='native_prefix_leases.py',
            answer='qwen35_answer_finite.py')),
    }
    script = REMOTE_PREPARE
    for needle, value in replacements.items():
        script = script.replace(needle, value)
    ast.parse(script, filename='<remote prepare-only source>')
    shell = ('set -e\nsource ' + shlex.quote(ENTRY + '/metax-entry.env.sh') +
             '\n"$VENV_PYTHON" - <<\'PY\'\n' + script + '\nPY\n')
    completed = subprocess.run(SSH + ['bash', '-s'], input=shell.encode(),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
    (HERE / 'combined-capacity-b8-stage.stderr.txt').write_bytes(completed.stderr)
    (HERE / 'combined-capacity-b8-stage.stdout.json').write_bytes(completed.stdout)
    completed.check_returncode()
    result = json.loads(completed.stdout)
    result['stager_source'] = dict(path=str(Path(__file__).resolve()), sha256=digest(Path(__file__)))
    result['remote_prepare_script_sha256'] = hashlib.sha256(script.encode()).hexdigest()
    result['scope'] = 'Prepared only; no model/GPU/optimizer/profiler/checkpoint operation or formal change.'
    (HERE / 'combined-capacity-b8-stage-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
