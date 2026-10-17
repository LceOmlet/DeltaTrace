"""Rebind six accepted native preparations to fresh runs after efficiency tuning.

Only process/output identities and accepted native allocator resources change.
The pinned owner still constructs argv and resolves every Hydra configuration.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/six-run-fresh-after-efficiency-20261017-v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def main():
    from omegaconf import OmegaConf
    assert not (AUDIT / 'six-combination-manifest.json').exists()
    resources = json.loads((AUDIT / 'native-resource-owner-prepared.json').read_bytes())
    all_jobs, segments = [], []
    for task in ('TextCraft', 'AppWorld'):
        old = ROOT / 'receipts' / (task.lower() + '-eight-gpu-sequential-20261017-v1')
        new = ROOT / 'receipts' / (task.lower() + '-eight-gpu-sequential-20261017-v2')
        new.mkdir(parents=True, exist_ok=False)
        (new / 'source').mkdir()
        original_stop = ROOT / 'receipts/textcraft-eight-gpu-sequential-20261017-v1/source/resume_at_native_checkpoint.py'
        assert sha(original_stop) == 'b5be347a4ba7deb2f02b7c892f375642b91d4992290d64f1d7b7c2659b211f0f'
        (new / 'source/resume_at_native_checkpoint.py').write_bytes(original_stop.read_bytes())
        parent = json.loads((old / 'campaign-prepared.json').read_bytes())
        configs, rows = {}, []
        for method in ('dt', 'ppo', 'grpo'):
            old_row = json.loads((old / (method + '-prepared.json')).read_bytes())
            version = old_row['version'].removesuffix('-v1') + '-v2'
            row = json.loads(json.dumps(old_row).replace(old_row['version'], version))
            assert not Path(row['run']).exists() and not Path(row['checkpoints']).exists()
            row['effective_environment']['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False,pinned_use_cuda_host_register:True'
            row['options']['+ray_init.runtime_env.env_vars.PYTORCH_CUDA_ALLOC_CONF'] = row['effective_environment']['PYTORCH_CUDA_ALLOC_CONF']
            assert row['options']['trainer.resume_mode'] == 'disable'
            assert row['options']['trainer.resume_from_path'] is None
            cpu_env = dict(os.environ, **row['effective_environment'])
            cpu_env['CUDA_VISIBLE_DEVICES'] = ''
            command = ('import json,sys; from owner_runtime_options import owner_command; '
                       'print(json.dumps(owner_command(json.loads(sys.stdin.read()))))')
            row['argv'] = json.loads(subprocess.check_output([cpu_env['VENV_PYTHON'], '-c', command],
                input=json.dumps(row['options']), env=cpu_env, text=True))
            assert row['argv'][0] == row['effective_environment']['VENV_PYTHON']
            result = subprocess.run(row['argv'] + ['--cfg', 'job', '--resolve'],
                cwd=row['entry'], env=cpu_env, capture_output=True)
            (new / (method + '-effective-config.yaml')).write_bytes(result.stdout)
            (new / (method + '-CPU-config.stderr')).write_bytes(result.stderr)
            result.check_returncode()
            cfg = OmegaConf.to_container(OmegaConf.create(result.stdout.decode()), resolve=True)
            prior = OmegaConf.to_container(OmegaConf.load(old / (method + '-effective-config.yaml')), resolve=True)
            for section in ('data', 'env', 'actor_rollout_ref', 'critic', 'algorithm'):
                assert cfg[section] == prior[section], (task, method, section)
            assert cfg['trainer']['total_training_steps'] == 200
            assert cfg['trainer']['resume_mode'] == 'disable' and cfg['trainer']['resume_from_path'] is None
            assert cfg['trainer']['n_gpus_per_node'] == 8
            assert cfg['actor_rollout_ref']['rollout']['max_model_len'] == 32768
            assert cfg['actor_rollout_ref']['actor']['ppo_micro_batch_size_per_gpu'] == 4
            row.update(status='prepared_not_started', fresh_base_model=True, resume_checkpoint=None,
                local_patch_commit=os.environ['RL_CAMPAIGN_COMMIT'],
                effective_config_SHA256=sha(new / (method + '-effective-config.yaml')),
                resource_version='native-host-register-and-MRoPE-row-copy-20261017-v1',
                native_resource_receipt=str(AUDIT / 'native-resource-owner-deployed.json'),
                prior_tuning_preparation=str(old / (method + '-prepared.json')),
                prior_tuning_source_SHA256=sha(old / (method + '-prepared.json')),
                no_tuning_weights_or_logs_reused=True)
            row['source_bindings'][resources['owner_path']] = resources['candidate_SHA256']
            configs[method] = cfg
            write(new / (method + '-prepared.json'), row)
            rows.append(row)
        for method in ('ppo', 'grpo'):
            for section in ('data', 'env', 'actor_rollout_ref'):
                assert configs[method][section] == configs['dt'][section], (task, method, section)
        receipt = deepcopy(parent)
        receipt.update(version=new.name, collected=time.time(), jobs=rows,
            local_patch_commit=os.environ['RL_CAMPAIGN_COMMIT'],
            source_bindings=rows[0]['source_bindings'],
            prior_tuning_history_excluded=True,
            status='three_native_CPU_configs_resolved_and_common_workload_equal')
        write(new / 'campaign-prepared.json', receipt)
        segments.append(dict(task=task, state=str(new / 'campaign-state.json'), prepared=str(new / 'campaign-prepared.json')))
        all_jobs.extend(rows)
    manifest = dict(version=AUDIT.name, status='prepared_not_started', collected=time.time(),
        source_commit=os.environ['RL_CAMPAIGN_COMMIT'], segments=segments,
        combinations=[{k:j[k] for k in ('task','method','version','run','output','checkpoints','budget')} for j in all_jobs],
        tuning_history=str(ROOT / 'receipts/textcraft-eight-gpu-sequential-20261017-v1/campaign-state.json'),
        all_six_start_from_base=True, tuning_iterations_spliced=0,
        completion_owner='Pinned run_campaign: native exit0, metrics1..200 and eight-rank full checkpoint200',
        resource_receipt=str(AUDIT / 'native-resource-owner-deployed.json'))
    write(AUDIT / 'six-combination-manifest.json', manifest)
    print(json.dumps(dict(status=manifest['status'], combinations=[(j['task'],j['method'],j['version']) for j in all_jobs])))


if __name__ == '__main__':
    main()
