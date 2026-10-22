"""Configure the accepted v5 branch of the existing corrected native stack.

Only credit mode and fresh run identities change. No training owner is copied.
"""
import copy
import hashlib
import json
from pathlib import Path
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/textcraft-v5-native-SSM-fp32-20261022-v1'
PRIOR_AUDIT = ROOT / 'receipts/textcraft-v10-eight-gpu-generation-aligned-20261022-v1'
PARENT_AUDIT = ROOT / 'receipts/textcraft-v10-native-SSM-fp32-20261022-v1'
VERSION = 'textcraft-dtpo-v5-ssm-fp32-8gpu-formal-20261022-v1'
RUN = ROOT / 'runs' / VERSION
CANDIDATE = ROOT / 'candidates/textcraft-v5-native-SSM-fp32-20261022-v1'
CHECKPOINTS = ROOT.parent.parent / 'deltatrace_backups/textcraft-8gpu-20261022' / VERSION / 'checkpoints'
STATUS = 'original_v5_corrected_SSM_FP32_fresh_configuration_resolved'
MODE = 'reward_target_exponential_v5'
METHOD_COMMIT = '4545c55807e7d002287717b6a4254f9b083dd5fc'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    assert not CANDIDATE.exists() and not RUN.exists() and not CHECKPOINTS.exists()
    assert not (AUDIT / 'dt-prepared.json').exists()
    original = PARENT_AUDIT / 'dt-prepared.json'
    parent = json.loads(original.read_text())
    assert all(sha(p) == h for p, h in parent['source_bindings'].items())
    row = copy.deepcopy(parent)
    old_entry, old_run = Path(parent['entry']), parent['run']
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

    def identity(value):
        return value.replace(old_run, str(RUN)).replace(str(old_entry), str(entry)) if isinstance(value, str) else value

    environment = row['effective_environment']
    environment.update({k: identity(v) for k, v in environment.items()})
    environment.update(DT_ENTRY_ROOT=str(entry), DT_CREDIT_MODE=MODE, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
    changes = {k: identity(v) for k, v in row['options'].items() if identity(v) != v}
    changes.update({
        'trainer.experiment_name': VERSION,
        'trainer.default_local_dir': str(CHECKPOINTS),
        '+ray_init.runtime_env.env_vars.DT_CREDIT_MODE': MODE,
    })
    seen, argv = set(), []
    for argument in row['argv']:
        key = argument.split('=', 1)[0]
        if key in changes:
            assert key not in seen
            argument = key + '=' + json.dumps(changes[key])
            seen.add(key)
        argv.append(argument)
    assert seen == set(changes)
    row['argv'] = argv
    row['options'].update(changes)
    assert all(row['options'][k] == v for k, v in parent['options'].items() if k not in changes)
    row['source_bindings'] = {identity(p): sha(identity(p)) for p in parent['source_bindings']}
    assert all(sha(entry / n) == sha(old_entry / n) for n in ('reward_readout.py', 'dt_training_batch.py', 'textcraft_owner_rollout.py', 'accepted_dt_storage_profile.py'))
    for key in ('pid', 'birth', 'started_unix', 'observed_process_created_unix', 'source_receipt', 'log', 'stdout', 'stderr', 'exit_file', 'active', 'alive', 'ray_session', 'completed_iterations', 'phase', 'progress_collected', 'optional_observation_unavailable', 'previous_job_identity', 'resumed_completed_iteration', 'same_logical_run', 'resource_resume_bridge_source_commit', 'failed_prior_eight_GPU_execution', 'failed_prior_executions', 'prior_execution_preserved'):
        row.pop(key, None)
    row.update(version=VERSION, status='prepared_only_original_v5_corrected_SSM_FP32', run=str(RUN), output=str(RUN / 'textcraft-dt'), checkpoints=str(CHECKPOINTS), entry=str(entry), fresh_base_model=True, resume_checkpoint=None, fresh_completed_iteration=0, execution_source_commit=None, local_patch_commit=None, adapter_source_commit=None, source_assembly_uncommitted=True, method_source_commit=METHOD_COMMIT, training_logic_parent_commit=METHOD_COMMIT, target_credit_rule='original_same_event_self_target_Q_b_V_zero_A_b; TextCraft_b_equals_official_group_g', prior_source_credit_rule='original_unscaled_DT_d_then_negative_expm1', source_change_scope='Existing accepted v5 mode and independent run identities only; corrected generation, SSM FP32, DT resource owner and native VERL scheduling unchanged.', source_reuse=dict(entry=str(old_entry), VERL=row['verl_root'], DT=row['dt_root']), changed_argv_keys=sorted(changes), initial_natural_iteration_pending=True)
    row['verification_receipts'].update(original_v5_CPU=str(ROOT / 'receipts/textcraft-v5-credit-integration-20261020-v1/CPU-behavior-verification.json'), original_v5_native=str(ROOT / 'receipts/textcraft-v5-credit-integration-20261020-v1/native-behavior-status.json'), unchanged_default_v5=str(ROOT / 'receipts/textcraft-v10-target-logprob-20261021-v1/CPU-behavior-verification.json'))
    row['orchestration_source_bindings'] = {str(p): sha(p) for p in (AUDIT / 'source').glob('*.py')}
    assert row['budget']['total_iterations'] == 330 and row['options']['trainer.total_epochs'] == 30
    assert row['options']['trainer.total_training_steps'] is None
    assert row['options']['trainer.resume_mode'] == 'disable' and row['options']['trainer.resume_from_path'] is None
    put(AUDIT / 'dt-prepared.json', row)
    put(AUDIT / 'campaign-prepared.json', dict(status=STATUS, order=['dt'], jobs=[row]))
    for name in ('formal-training.json', 'active-training.json', 'active-source.json'):
        assert not (AUDIT / name).exists()
        put(AUDIT / name, dict(jobs=[], source_commit=None, updated_unix=time.time(), scope=VERSION))
    cleanup = AUDIT / 'source/resume_at_native_checkpoint.py'
    if not cleanup.exists():
        cleanup.symlink_to(PARENT_AUDIT / 'source/resume_at_native_checkpoint.py')
    assert sha(cleanup) == 'b5be347a4ba7deb2f02b7c892f375642b91d4992290d64f1d7b7c2659b211f0f'
    record = dict(status='prepared_only_not_submitted', prepared_unix=time.time(), run=str(RUN), mode=MODE, parent_source=str(original), parent_source_SHA256=sha(original), source_bindings=row['source_bindings'], required_installed_source_bindings=row['required_installed_source_bindings'], changed_argv_keys=sorted(changes), budget=row['budget'], fresh_base_model=True, model_calls=0, environment_calls=0, DT_calls=0, optimizer_updates=0)
    put(AUDIT / 'fresh-preparation.json', record)
    print(json.dumps(dict(status=record['status'], run=str(RUN), mode=MODE, changed_argv_keys=sorted(changes), budget=row['budget'])))


if __name__ == '__main__':
    main()
