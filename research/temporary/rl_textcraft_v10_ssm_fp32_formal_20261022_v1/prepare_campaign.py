"""Prepare one fresh run by reusing the deployed v10 and resource owners.

Only run identities, native fresh-start settings, and the existing SSM cache
option change. This script does not execute a model, environment or optimizer.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/textcraft-v10-native-SSM-fp32-20261022-v1'
PRIOR_AUDIT = ROOT / 'receipts/textcraft-v10-eight-gpu-generation-aligned-20261022-v1'
PRIOR_RUN = ROOT / 'runs/textcraft-dtpo-v10-target-logprob-8gpu-formal-20261022-v2'
VERSION = 'textcraft-dtpo-v10-target-logprob-ssm-fp32-8gpu-formal-20261022-v1'
RUN = ROOT / 'runs' / VERSION
CANDIDATE = ROOT / 'candidates/textcraft-v10-native-SSM-fp32-20261022-v1'
CHECKPOINTS = ROOT.parent.parent / 'deltatrace_backups/textcraft-8gpu-20261022' / VERSION / 'checkpoints'
MODE = 'task_sample_turn_v1'
STATUS = 'v10_ssm_fp32_formal_configuration_resolved'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    assert not CANDIDATE.exists() and not RUN.exists() and not CHECKPOINTS.exists()
    assert not (AUDIT / 'dt-prepared.json').exists(), 'Inspect existing preparation; do not replace it'
    original = PRIOR_RUN / 'source.json'
    assert sha(original) == 'f62b2b1fc720d164635202e9dba378dee79935f6ec5e1ac5045d5477bf4435d7'
    parent = json.loads(original.read_text())
    row = copy.deepcopy(parent)
    old_entry = Path(parent['entry'])
    entry = CANDIDATE / 'entry'
    entry.mkdir(parents=True)
    for child in old_entry.iterdir():
        if child.name != '__pycache__':
            (entry / child.name).symlink_to(child.resolve())
    config = entry / 'owner_environment_configs.json'
    settings = json.loads(config.read_text())
    settings['TextCraft']['train']['actor_rollout_ref']['rollout']['rollout_log_dir'] = str(RUN / 'native-executer-logs/train')
    settings['TextCraft']['eval']['rollout']['rollout_log_dir'] = str(RUN / 'native-executer-logs/eval')
    config.unlink()
    config.write_text(json.dumps(settings, indent=2) + '\n')
    for name, digest in {
        'reward_readout.py': '812ef4662771055d1045d78ed76cc8a1d2d8bd3ae607d1b6e9ec809a49ee51cd',
        'dt_training_batch.py': '006419cd4c186a44f4bbbc895994eff032e01136f5e70d9036915750266dcfc4',
        'textcraft_owner_rollout.py': 'da35841f5319a427f079bb6f6a297dce741157e1d4fba3c20f0577de193fd7b9',
        'accepted_dt_storage_profile.py': '763143c13e589612408a8712dbb3a917fe38475fe11b7041a6591afdb82bea22',
    }.items():
        assert sha(entry / name) == digest, name
    environment = row['effective_environment']
    accepted_DT = ROOT / 'candidates/dt-native-replay-host-lifetime-20261017-v3/deltatrace'
    assert environment['DT_ROOT'] == str(accepted_DT)
    assert sha(accepted_DT / 'clean/qwen35/qwen35_dense_finite_runner.py') == '6a112224b3d4e63d047a576d7df68ff1ee65e4bbacfcb0014f84c17bd82b8520'
    assert environment['DT_NATIVE_HOST_CACHE_BATCH'] == '1'
    assert environment['DT_NATIVE_HOST_CACHE_REPLAY_LAYER'] == '0'
    assert environment['DT_ACCEPTED_STORAGE_PROFILE'] == 'root_pageable_replay_pinned_resident16384_v1'
    old_run = str(PRIOR_RUN)
    for key, value in list(environment.items()):
        if isinstance(value, str):
            environment[key] = value.replace(old_run, str(RUN)).replace(str(old_entry), str(entry))
    environment.update(DT_ENTRY_ROOT=str(entry), VERL_ROOT=row['verl_root'], PWD=row['verl_root'],
                       CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
    changes = {
        'trainer.experiment_name': VERSION,
        'trainer.default_local_dir': str(CHECKPOINTS),
        'trainer.rollout_data_dir': str(RUN / 'textcraft-dt/rollouts'),
        'trainer.validation_data_dir': str(RUN / 'textcraft-dt/validation'),
        'trainer.resume_mode': 'disable',
        'trainer.resume_from_path': None,
        '+ray_init.runtime_env.env_vars.DT_FSDP_EVENT_WAIT_RECEIPTS': environment['DT_FSDP_EVENT_WAIT_RECEIPTS'],
        '+ray_init.runtime_env.env_vars.DT_ACCEPTED_STORAGE_RECEIPTS': environment['DT_ACCEPTED_STORAGE_RECEIPTS'],
    }
    addition = '+actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_ssm_cache_dtype'
    assert addition not in row['options']
    changes[addition] = 'float32'
    seen = set()
    argv = []
    for argument in row['argv']:
        key = argument.split('=', 1)[0]
        if key in changes:
            assert key not in seen, key
            argument = key + '=' + json.dumps(changes[key])
            seen.add(key)
        argv.append(argument)
    assert seen == set(changes) - {addition}
    argv.append(addition + '=' + json.dumps(changes[addition]))
    row['argv'] = argv
    row['options'].update(changes)
    assert all(row['options'][key] == value for key, value in parent['options'].items() if key not in changes)
    bindings = {}
    for path, digest in parent['source_bindings'].items():
        assert sha(path) == digest, path
        mapped = path.replace(str(old_entry) + '/', str(entry) + '/', 1) if path.startswith(str(old_entry) + '/') else path
        bindings[mapped] = sha(mapped)
    row['source_bindings'] = bindings
    prepared_patch = json.loads((PRIOR_AUDIT / 'original-owner-SSM-dtype-deployment-preparation-v1.json').read_text())
    row['required_installed_source_bindings'] = {x['original']: x['candidate_SHA256'] for x in prepared_patch['files']}
    for key in (
        'pid', 'birth', 'started_unix', 'observed_process_created_unix', 'source_receipt', 'log',
        'stdout', 'stderr', 'exit_file', 'active', 'alive', 'ray_session', 'completed_iterations',
        'phase', 'progress_collected', 'optional_observation_unavailable', 'previous_job_identity',
        'resumed_completed_iteration', 'same_logical_run', 'resource_resume_bridge_source_commit',
        'failed_prior_eight_GPU_execution', 'failed_prior_executions',
    ):
        row.pop(key, None)
    row.update(version=VERSION, status='prepared_only_fresh_v10_native_SSM_FP32',
        run=str(RUN), output=str(RUN / 'textcraft-dt'), checkpoints=str(CHECKPOINTS),
        entry=str(entry), dt_root=str(accepted_DT), fresh_base_model=True, resume_checkpoint=None,
        fresh_completed_iteration=0, execution_source_commit=None, local_patch_commit=None,
        adapter_source_commit=None, source_assembly_uncommitted=True,
        training_logic_parent_commit='4545c55807e7d002287717b6a4254f9b083dd5fc',
        method_source_commit='53b421c7a6d4bdb045806a49cd22f5e00dce7769',
        original_fresh_execution_source_commit=parent['execution_source_commit'],
        original_fresh_base_model_start_preserved=True,
        native_SSM_cache_option='float32', generation_activation_dtype='bfloat16', convolution_cache_dtype='auto',
        prior_execution_preserved=dict(run=str(PRIOR_RUN), source=str(original), source_SHA256=sha(original),
                                      stopped_at_complete_iteration=30, history_not_joined=True),
        source_reuse=dict(entry=str(old_entry), VERL=row['verl_root'], DT=str(accepted_DT),
                          scope='Exact accepted v10/readout/resource/generation/PPO source reuse; new log identities and existing native SSM option.'),
        source_change_scope='Original installed GDN SSM argument bridge; no kernels, sampler, DT, advantage or PPO change.',
        changed_argv_keys=sorted(changes), initial_natural_iteration_pending=True)
    row['verification_receipts'].update(
        SSM_quality=str(PRIOR_AUDIT / 'native-SSM-dtype-complete-quality-review-v1.json'),
        original_owner_SSM_installation=str(PRIOR_AUDIT / 'original-owner-SSM-dtype-installed-v1/deployment.json'),
        accepted_resource_actual_binding=str(PRIOR_RUN / 'runtime-overrides/actual-worker-imports-after-bound-resource-DT-v1.json'))
    row['orchestration_source_bindings'] = {str(p): sha(p) for p in (AUDIT / 'source').glob('*.py')}
    assert row['budget']['total_iterations'] == 330 and row['options']['trainer.total_epochs'] == 30
    assert row['options']['trainer.total_training_steps'] is None
    put(AUDIT / 'dt-prepared.json', row)
    put(AUDIT / 'campaign-prepared.json', dict(status=STATUS, order=['dt'], jobs=[row]))
    for name in ('formal-training.json', 'active-training.json', 'active-source.json'):
        assert not (AUDIT / name).exists()
        put(AUDIT / name, dict(jobs=[], source_commit=None, updated_unix=time.time(), scope=VERSION))
    record = dict(status='prepared_only_not_submitted', prepared_unix=time.time(), run=str(RUN),
        source_bindings=bindings, required_installed_source_bindings=row['required_installed_source_bindings'],
        changed_argv_keys=sorted(changes), budget=row['budget'], fresh_base_model=True,
        parent_source_SHA256=sha(original), credit_training_and_resource_sources_unchanged=True,
        model_calls=0, environment_calls=0, DT_calls=0, optimizer_updates=0)
    put(AUDIT / 'fresh-preparation.json', record)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
