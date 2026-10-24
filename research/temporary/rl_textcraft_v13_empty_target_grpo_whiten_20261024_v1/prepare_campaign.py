"""Configure v13: v5 attribution, target-free GRPO, then batch whitening.

Only the requested fallback and fresh identities change. The native training owner is reused.
"""
import copy
import hashlib
import json
from pathlib import Path
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/textcraft-v13-empty-target-grpo-whiten-20261024-v1'
PRIOR_AUDIT = ROOT / 'receipts/textcraft-v10-eight-gpu-generation-aligned-20261022-v1'
PARENT_AUDIT = ROOT / 'receipts/textcraft-v12-attribution-then-group-20261024-v1'
VERSION = 'textcraft-dtpo-v13-empty-target-grpo-whiten-ssm-fp32-8gpu-formal-20261024-v1'
RUN = ROOT / 'runs' / VERSION
CANDIDATE = ROOT / 'candidates/textcraft-v13-empty-target-grpo-whiten-20261024-v1'
CHECKPOINTS = ROOT.parent.parent / 'deltatrace_backups/textcraft-8gpu-20261024' / VERSION / 'checkpoints'
STATUS = 'v13_empty_target_grpo_whiten_only_configuration_resolved'
MODE = 'reward_target_exponential_whiten_v13'
METHOD_COMMIT = '4545c55807e7d002287717b6a4254f9b083dd5fc'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    assert not RUN.exists() and not CHECKPOINTS.exists()
    assert not (AUDIT / 'dt-prepared.json').exists()
    original = PARENT_AUDIT / 'dt-prepared.json'
    parent = json.loads(original.read_text())
    assert all(sha(p) == h for p, h in parent['source_bindings'].items())
    row = copy.deepcopy(parent)
    old_entry, old_run = Path(parent['entry']), parent['run']
    old_verl = parent['verl_root']
    new_verl = str(CANDIDATE/'verl-owner')
    row['verl_root'] = new_verl
    entry = CANDIDATE / 'entry'
    entry.mkdir(parents=True, exist_ok=True)
    for child in old_entry.iterdir():
        if child.name != '__pycache__' and not (entry / child.name).exists():
            (entry / child.name).symlink_to(child.resolve())
    config = entry / 'owner_environment_configs.json'
    settings = json.loads(config.read_text())
    settings['TextCraft']['train']['actor_rollout_ref']['rollout']['rollout_log_dir'] = str(RUN / 'native-executer-logs/train')
    settings['TextCraft']['eval']['rollout']['rollout_log_dir'] = str(RUN / 'native-executer-logs/eval')
    config.unlink()
    config.write_text(json.dumps(settings, indent=2) + '\n')

    def identity(value):
        return value.replace(old_run, str(RUN)).replace(str(old_entry), str(entry)).replace(old_verl, new_verl) if isinstance(value, str) else value

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
    row['required_installed_source_bindings'] = {identity(p): sha(identity(p)) for p in parent['required_installed_source_bindings']}
    assert sha(entry / 'accepted_dt_storage_profile.py') == sha(old_entry / 'accepted_dt_storage_profile.py')
    assert json.loads((AUDIT / 'CPU-behavior-verification.json').read_text())['status'] == 'passed'
    for key in ('pid', 'birth', 'started_unix', 'observed_process_created_unix', 'source_receipt', 'log', 'stdout', 'stderr', 'exit_file', 'active', 'alive', 'ray_session', 'completed_iterations', 'phase', 'progress_collected', 'optional_observation_unavailable', 'previous_job_identity', 'resumed_completed_iteration', 'same_logical_run', 'resource_resume_bridge_source_commit', 'failed_prior_eight_GPU_execution', 'failed_prior_executions', 'prior_execution_preserved'):
        row.pop(key, None)
    for key in ('group_statistic', 'group_std_correction', 'group_epsilon'):
        row.pop(key, None)
    row.update(version=VERSION, status='prepared_only_v13_empty_target_grpo_whiten', run=str(RUN), output=str(RUN / 'textcraft-dt'), checkpoints=str(CHECKPOINTS), entry=str(entry), fresh_base_model=True, resume_checkpoint=None, fresh_completed_iteration=0, execution_source_commit=None, local_patch_commit=None, adapter_source_commit=None, source_assembly_uncommitted=True, method_source_commit=None, method_parent_commit=METHOD_COMMIT, training_logic_parent_commit=METHOD_COMMIT, target_credit_rule='Original v5 target g; target-free rollout native g broadcast; one subsequent batch masked_whiten', prior_source_credit_rule='Original v5 g times negative_expm1 d; unchanged masks and scatter; one batch masked_whiten', source_change_scope='v11 raw credit plus native masked_whiten shift_mean=True. Original DT/PPO/generation/optimizer/scheduling unchanged.', source_reuse=dict(entry=str(old_entry), VERL=row['verl_root'], DT=row['dt_root']), changed_argv_keys=sorted(changes), initial_natural_iteration_pending=True)
    row.update(method_parent_commit='b7216e2357db50547703ffb5ced0c6419ad60d46',
               source_reuse=dict(entry=str(old_entry), VERL=old_verl, DT=row['dt_root']),
               advantage_statistic='whole native training batch retained response_mask tokens',
               variance_owner='verl.utils.torch_functional.masked_var unbiased=True', variance_epsilon=1e-8,
               pre_attribution_GRPO=True, shift_mean=True, second_advantage_normalization=False)
    row['verification_receipts'].update(original_v5_CPU=str(ROOT / 'receipts/textcraft-v5-credit-integration-20261020-v1/CPU-behavior-verification.json'), original_v5_native=str(ROOT / 'receipts/textcraft-v5-credit-integration-20261020-v1/native-behavior-status.json'), unchanged_default_v5=str(AUDIT / 'CPU-behavior-verification.json'))
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
