"""CPU-only source/configuration preparation for one fixed fresh AppWorld run.

This does not stop or submit a process, instantiate a model, call DT, or read a
checkpoint. The existing frozen launcher still owns all training behaviour.
"""
from pathlib import Path
import copy
import hashlib
import json
import os
import subprocess
import time

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs = json.loads((HERE / 'inputs.json').read_bytes())
assert sha(HERE / 'inputs.json') == '7559b0905df5b59804cf151529252143eb655dca0ed71036e16c968071b93480'
root = Path(inputs['remote_root'])
prior_path = Path(inputs['formal_source']['path'])
assert sha(prior_path) == inputs['formal_source']['sha256']
prior = json.loads(prior_path.read_bytes())
prod_path = Path(inputs['production_prepared']['path'])
assert sha(prod_path) == inputs['production_prepared']['sha256']
prod = json.loads(prod_path.read_bytes())
imports_path = Path(inputs['production_cpu_imports']['path'])
assert sha(imports_path) == inputs['production_cpu_imports']['sha256']
imports = json.loads(imports_path.read_bytes())
assert imports['status'] == 'CPU_import_interface_passed_prepared_only'
assert not imports['cuda_initialized_before'] and not imports['cuda_initialized_after']
assert not imports['model_instantiated'] and not imports['DT_called']
assert sha(inputs['provisioning_environment']['path']) == inputs['provisioning_environment']['sha256']
frozen_prep_path = Path(prior['prepared_receipt'])
assert sha(frozen_prep_path) == prior['prepared_receipt_sha256']
frozen_prep = json.loads(frozen_prep_path.read_bytes())
assert prior['resume_mode'] == frozen_prep['checkpoint_resume_mode'] == 'disable'
assert not prior['checkpoint_restore_requested'] and not frozen_prep['checkpoint_restore_requested']
assert prod['inherited_verl_root'] == prior['verl_root']
assert prod['inherited_loop_root'] == prior['loop_root']
baseline = inputs['baseline_source_bindings']
assert len(baseline) == 1550
for path, digest in baseline.items():
    assert sha(path) == digest, path
for label, location, key in inputs['original_inventories']:
    for name, digest in prior[key].items():
        assert sha(Path(location) / name) == digest, (label, name)
for item in [prior['candidate_environment'], prior['canonical_HF_owner'], *prior['installed_verified_files']]:
    assert sha(item['path']) == item['sha256'], item['path']
lock = json.loads((root / 'receipts/environment-only-20260930/entry/verified_runtime.json').read_bytes())
for item in lock['native_fla_files'].values():
    assert sha(item['path']) == item['expected'], item['path']
for item in prod['import_resolution'].values():
    assert sha(item['path']) == item['sha256'], item['path']
for item in imports['actual_imported_sources'].values():
    assert sha(item['path']) == item['sha256'], item['path']
selected = inputs['selected_source_bindings']
assert len(selected) == 1550
binding_changes = []
for old_path, digest in baseline.items():
    item = inputs['binding_mapping'][old_path]
    assert selected[item['path']] == item['sha256']
    assert sha(item['path']) == item['sha256'], item['path']
    if item['sha256'] != digest:
        binding_changes.append(old_path)
assert set(binding_changes) == set(inputs['allowed_binding_sha_changes'])

entry = Path(prod['entry'])
dt = Path(prod['dt_root'])
old_entry = Path(inputs['formal_entry'])
old_dt = Path(prior['dt_root'])
launcher = entry / 'launch_appworld_native.py'
assert sha(launcher) == inputs['launcher_sha256']
assert sha(old_entry / launcher.name) == inputs['launcher_sha256']
old_environment = json.loads(Path(prior['candidate_environment']['path']).read_bytes())
new_environment = json.loads(Path(prod['environment']['path']).read_bytes())
changed_environment = copy.deepcopy(old_environment)
changed_environment['qwen35'].update({key: value['after'] for key, value in prod['environment']['changed_qwen35_fields'].items()})
assert changed_environment == new_environment
assert set(prod['environment']['changed_qwen35_fields']) == {
    'finite_library', 'finite_library_sha256', 'individual_prefixes', 'boundary_row_storage'}
new_pythonpath = ':'.join(
    str(dt) + value[len(str(old_dt)):] if value == str(old_dt) or value.startswith(str(old_dt) + '/')
    else str(entry) + value[len(str(old_entry)):] if value == str(old_entry) or value.startswith(str(old_entry) + '/')
    else value for value in prior['pythonpath'].split(':'))
env = dict(os.environ)
env.update(prior.get('resource_environment', {}))
env.update(frozen_prep['resource_environment'])
for key in ['MACA_VISIBLE_DEVICES', 'RAY_ADDRESS', 'RAY_TMPDIR', 'DT_PREFIX_CHECKPOINT',
            'DT_PREFIX_NATIVE_CONV_INITIAL_STATES', 'DT_CONV_ISOLATED_IMPORT_ROOT']:
    env.pop(key, None)
env.update(VERL_ROOT=prior['verl_root'], DT_ROOT=str(dt), DT_ENTRY_ROOT=str(entry),
    LOOP_ROOT=prior['loop_root'], DT_ENVIRONMENT_JSON=prod['environment']['path'],
    CUDA_VISIBLE_DEVICES='4,5', APPWORLD_ROOT=str(root / 'receipts/environment-only-20260930/loop-entry/appworld-root'),
    PYTHONPATH=new_pythonpath)
assert env['MODEL_PATH'] == prior['fresh_base_model']
original_env = dict(env)
original_env.update(DT_ROOT=str(old_dt), DT_ENTRY_ROOT=str(old_entry),
    DT_ENVIRONMENT_JSON=prior['candidate_environment']['path'], PYTHONPATH=prior['pythonpath'])
assert {key for key in env if env.get(key) != original_env.get(key)} == {
    'DT_ROOT', 'DT_ENTRY_ROOT', 'DT_ENVIRONMENT_JSON', 'PYTHONPATH'}

config_reader = '''import json,os,sys
from pathlib import Path
import launch_appworld_native
options,sampling=launch_appworld_native.options_for(Path(sys.argv[1]))
initialized=bool('torch' in sys.modules and sys.modules['torch'].cuda.is_initialized())
assert not initialized
print('OWNER_CONFIG_JSON='+json.dumps(dict(options=options,sampling=sampling,launcher=launch_appworld_native.__file__,cuda_initialized=initialized)))
'''
def compose(environment, output):
    cpu = dict(environment, CUDA_VISIBLE_DEVICES='', MACA_VISIBLE_DEVICES='-1')
    completed = subprocess.run([cpu['VENV_PYTHON'], '-c', config_reader, str(output)],
        env=cpu, capture_output=True, text=True, timeout=60, check=True)
    return json.loads(next(line.removeprefix('OWNER_CONFIG_JSON=')
        for line in completed.stdout.splitlines() if line.startswith('OWNER_CONFIG_JSON=')))

old_output = Path(inputs['formal_output'])
output = Path(inputs['new_output'])
old_config = compose(original_env, old_output)
new_config = compose(env, output)
original_launch = json.loads((old_output / 'launch.json').read_bytes())
assert old_config['options'] == original_launch['options'], 'Original options no longer match the actual v3 launch'
assert new_config['sampling'] == old_config['sampling']
option_changes = {key: dict(before=old_config['options'].get(key), after=value)
    for key, value in new_config['options'].items() if value != old_config['options'].get(key)}
assert set(option_changes) == set(inputs['allowed_launcher_option_changes'])
for key, item in option_changes.items():
    assert item['after'] == inputs['allowed_launcher_option_changes'][key], key
assert new_config['options']['trainer.resume_mode'] == 'disable'
assert 'trainer.resume_from_path' not in new_config['options']
for key, value in inputs['fixed_options'].items():
    assert new_config['options'][key] == value, key

new_source = copy.deepcopy(prior)
new_source.update(unix=time.time(), dt_root=str(dt), pythonpath=new_pythonpath,
    prior_driver_pid=inputs['formal_pid'], prior_source_receipt=str(prior_path),
    prior_source_sha256=inputs['formal_source']['sha256'],
    submission_repository_commit=inputs['repository_commit'],
    source_scope='Same original VERL/LOOP PPO workload and whole-mask whiten; measured storage and per-row prefix representation through official DT owner interfaces; fresh base with no checkpoint loading',
    row_prefix_production_preparation=inputs['production_prepared'],
    row_prefix_production_cpu_imports=inputs['production_cpu_imports'],
    row_prefix_capacity_result=inputs['combined_capacity_result'],
    source_bindings=selected, checkpoint_restore_requested=False, resume_mode='disable')
new_source['candidate_environment'] = prod['environment']
new_source['resource_environment'] = dict(prior['resource_environment'],
    DT_ROOT=str(dt), DT_ENVIRONMENT_JSON=prod['environment']['path'])
for name, digest in inputs['allowed_entry_changes'].items():
    new_source['entry_sha256'][name] = digest
for name, digest in inputs['allowed_dt_changes'].items():
    new_source['dt_source_sha256'][name] = digest
for key in ['resume_from', 'completed_checkpoint_marker', 'resume_launcher',
            'unfinished_rollout_restart', 'initialization_retry']:
    new_source.pop(key, None)
for label, location, key in [('entry', str(entry), 'entry_sha256'),
        ('VERL', prior['verl_root'], 'verl_sha256'), ('LOOP', prior['loop_root'], 'author_sha256'),
        ('DT', str(dt), 'dt_source_sha256')]:
    for name, digest in new_source[key].items():
        assert sha(Path(location) / name) == digest, (label, name)
assert sha(inputs['combined_capacity_result']['path']) == inputs['combined_capacity_result']['sha256']
prepared_path = HERE / 'prepared.json'
assert not prepared_path.exists(), 'Preserve the prior preparation and inspect its receipt'
(HERE / 'source-template.json').write_text(json.dumps(new_source, indent=2) + '\n')
(HERE / 'run-env.json').write_text(json.dumps(env, indent=2) + '\n')
os.chmod(HERE / 'run-env.json', 0o600)
(HERE / 'new-launch-options.json').write_text(json.dumps(new_config, indent=2) + '\n')
receipt = dict(status='prepared_only_not_stopped_not_submitted', observed_unix=time.time(),
    inputs_sha256=sha(HERE / 'inputs.json'),
    source_template=dict(path=str(HERE / 'source-template.json'), sha256=sha(HERE / 'source-template.json')),
    run_environment=dict(path=str(HERE / 'run-env.json'), sha256=sha(HERE / 'run-env.json'), mode='0600'),
    options=dict(path=str(HERE / 'new-launch-options.json'), sha256=sha(HERE / 'new-launch-options.json')),
    baseline_binding_count=len(baseline), selected_binding_count=len(selected),
    binding_sha_changes=binding_changes, entry_changes=inputs['allowed_entry_changes'],
    dt_changes=inputs['allowed_dt_changes'], launcher_option_changes=option_changes,
    inherited_budget=inputs['budget'], model=False, GPU=False, DT=False, checkpoint=False,
    stop=False, submit=False, source_mutation=False,
    scope='CPU original configuration composition and literal source binding checks only; capacity and numerical evidence retain their original scope')
prepared_path.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(prepared=str(prepared_path), sha256=sha(prepared_path),
    binding_count=len(selected), changed_binding_count=len(binding_changes),
    changed_options=list(option_changes), stop=False, submit=False)))
