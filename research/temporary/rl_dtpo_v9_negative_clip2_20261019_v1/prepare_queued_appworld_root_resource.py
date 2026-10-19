"""Persist the accepted original root-cache option for the queued AppWorld DT.

Use owner_command and Hydra's original CPU configuration resolver. No running
process, numeric kernel, task workload or other algorithm is changed.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'
SEGMENT = ROOT / 'receipts/appworld-eight-gpu-v9-negative-clip2-20261019-v4'
RUN = ROOT / 'runs/textcraft-dtpo-v9-negative-clip2-8gpu-200-20261019-v4'
VERSION = 'root-pageable-replay-pinned-20261019-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    p = Path(path)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, p)


def main():
    from omegaconf import OmegaConf
    record_path = SEGMENT / 'resource-overrides' / (VERSION + '-prepared.json')
    if record_path.exists():
        saved = json.loads(record_path.read_bytes())
        for path, expected in saved['prepared_files_after'].items():
            assert sha(path) == expected
        print(json.dumps(dict(status=saved['status'], receipt=str(record_path))))
        return
    assert not (SEGMENT / 'campaign-state.json').exists(), 'AppWorld has started; do not edit its preparation'
    proof_path = AUDIT / 'root-resource-production-1792409305/production-window-receipt.json'
    proof = json.loads(proof_path.read_bytes())
    assert proof['status'] == 'completed_read_only_production_window'
    assert proof['latest_completed'] >= 14 and proof['model_calls'] == 0
    override = json.loads((RUN / 'runtime-overrides/root-pageable-resource-override-completed.json').read_bytes())
    assert override['effective_root_flags'] == dict(pin_root_host=False, pin_replay_host=True)
    assert len(override['actual_workers']) == 8
    assert all(r['override']['native_runner_SHA256'] == '6a112224b3d4e63d047a576d7df68ff1ee65e4bbacfcb0014f84c17bd82b8520'
               for r in override['actual_workers'])
    row_path = SEGMENT / 'dt-prepared.json'
    campaign_path = SEGMENT / 'campaign-prepared.json'
    config_path = SEGMENT / 'dt-effective-config.yaml'
    row = json.loads(row_path.read_bytes())
    campaign = json.loads(campaign_path.read_bytes())
    assert row['task'] == 'AppWorld' and row['method'] == 'dt'
    assert row['fresh_base_model'] and row['resume_checkpoint'] is None
    assert row['options']['trainer.total_training_steps'] == 200
    assert row['options']['trainer.save_freq'] == 2
    assert row['effective_environment']['DT_EXPONENTIAL_CREDIT_NEGATIVE_CLIP'] == '2'
    assert not Path(row['run']).exists(), 'Prepared run unexpectedly exists'
    assert all(sha(p) == expected for p, expected in row['source_bindings'].items())
    previous = {str(p): sha(p) for p in (row_path, campaign_path, config_path)}
    controls = {str(SEGMENT / (m + suffix)): sha(SEGMENT / (m + suffix))
                for m in ('ppo', 'grpo') for suffix in ('-prepared.json', '-effective-config.yaml')}

    resource = AUDIT / VERSION
    resource.mkdir(exist_ok=True)
    old_env_path = Path(row['effective_environment']['DT_ENVIRONMENT_JSON'])
    old_env = json.loads(old_env_path.read_bytes())
    assert old_env['qwen35']['dt_pin_root_host'] is True
    assert old_env['qwen35']['dt_pin_replay_host'] is True
    env = copy.deepcopy(old_env)
    env['qwen35']['dt_pin_root_host'] = False
    env_path = resource / 'environment.json'
    if env_path.exists():
        assert json.loads(env_path.read_bytes()) == env
    else:
        write(env_path, env)
    check = copy.deepcopy(env)
    check['qwen35']['dt_pin_root_host'] = True
    assert check == old_env

    backup = resource / 'AppWorld-preparation-before'
    backup.mkdir(exist_ok=True)
    for p in (row_path, campaign_path, config_path):
        target = backup / p.name
        if target.exists():
            assert target.read_bytes() == p.read_bytes()
        else:
            target.write_bytes(p.read_bytes())
    new = copy.deepcopy(row)
    new['effective_environment']['DT_ENVIRONMENT_JSON'] = str(env_path)
    old_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    old_vars = old_cfg['ray_init']['runtime_env']['env_vars']
    option = ('+' if 'DT_ENVIRONMENT_JSON' not in old_vars else '') + 'ray_init.runtime_env.env_vars.DT_ENVIRONMENT_JSON'
    assert option not in new['options']
    new['options'][option] = str(env_path)
    cpu_env = dict(os.environ, **new['effective_environment'])
    cpu_env['CUDA_VISIBLE_DEVICES'] = ''
    cpu_env.pop('MACA_VISIBLE_DEVICES', None)
    command = ('import json,sys; from owner_runtime_options import owner_command; '
               'print(json.dumps(owner_command(json.loads(sys.stdin.read()))))')
    new['argv'] = json.loads(subprocess.check_output(
        [cpu_env['VENV_PYTHON'], '-c', command], input=json.dumps(new['options']),
        cwd=new['entry'], env=cpu_env, text=True))
    assert new['argv'][0] == new['effective_environment']['VENV_PYTHON']
    resolved = subprocess.run(new['argv'] + ['--cfg', 'job', '--resolve'],
                              cwd=new['entry'], env=cpu_env, capture_output=True, timeout=120)
    (resource / 'AppWorld-DT-CPU-config.stderr').write_bytes(resolved.stderr)
    (resource / 'AppWorld-DT-effective-config.yaml').write_bytes(resolved.stdout)
    resolved.check_returncode()
    cfg = OmegaConf.to_container(OmegaConf.create(resolved.stdout.decode()), resolve=True)
    assert cfg['ray_init']['runtime_env']['env_vars'].pop('DT_ENVIRONMENT_JSON') == str(env_path)
    if 'DT_ENVIRONMENT_JSON' in old_vars:
        cfg['ray_init']['runtime_env']['env_vars']['DT_ENVIRONMENT_JSON'] = old_vars['DT_ENVIRONMENT_JSON']
    assert cfg == old_cfg, 'Unexpected effective configuration change'
    new['effective_config_SHA256'] = hashlib.sha256(resolved.stdout).hexdigest()
    new['source_bindings'][str(env_path)] = sha(env_path)
    new.setdefault('completed_preparation_resource_overrides', []).append(dict(
        version=VERSION, status='prepared_not_started', receipt=str(record_path),
        effective_root_flags=override['effective_root_flags'],
        production_verification=str(proof_path), numeric_kernels_changed=False))
    new['resource_version'] = VERSION
    new_campaign = copy.deepcopy(campaign)
    assert sum(j['method'] == 'dt' for j in new_campaign['jobs']) == 1
    new_campaign['jobs'] = [new if j['method'] == 'dt' else j for j in new_campaign['jobs']]
    new_campaign['source_bindings'] = new['source_bindings']
    new_campaign.setdefault('completed_preparation_resource_overrides', []).append(dict(
        method='dt', version=VERSION, receipt=str(record_path), status='prepared_not_started'))
    assert not (SEGMENT / 'campaign-state.json').exists()
    assert all(sha(p) == expected for p, expected in previous.items())
    assert all(sha(p) == expected for p, expected in controls.items())
    write(row_path, new)
    config_path.write_bytes(resolved.stdout)
    write(campaign_path, new_campaign)
    record = dict(status='prepared_queued_AppWorld_original_root_storage_option', recorded=time.time(),
        version=VERSION, task='AppWorld', method='dt', run=row['run'],
        production_verification=str(proof_path), actual_training_execution_source=row['local_patch_commit'],
        original_environment=dict(path=str(old_env_path), SHA256=sha(old_env_path)),
        selected_environment=dict(path=str(env_path), SHA256=sha(env_path)),
        environment_changes={'qwen35.dt_pin_root_host': [True, False]},
        replay_pinned_preserved=True, effective_config_change='Only explicit Ray DT_ENVIRONMENT_JSON propagation',
        original_preparation_backup=str(backup), prepared_files_before=previous,
        prepared_files_after={str(p): sha(p) for p in (row_path, campaign_path, config_path)},
        untouched_PPO_GRPO=controls, AppWorld_launched=False,
        AppWorld_production_efficiency_verified=False, current_TextCraft_changed=False,
        model_calls=0, optimizer_calls=0, training_restarts=0,
        verification_scope='Same original DT runner, prior same-input eight-rank preservation and original B4/32768 per-rank capacity receipts; AppWorld actual worker bindings and production efficiency remain pending its native launch.')
    record_path.parent.mkdir(exist_ok=True)
    write(record_path, record)
    assert all(sha(p) == expected for p, expected in controls.items())
    print(json.dumps(dict(status=record['status'], receipt=str(record_path),
                          environment=str(env_path), model_calls=0, AppWorld_launched=False)))


if __name__ == '__main__':
    main()
