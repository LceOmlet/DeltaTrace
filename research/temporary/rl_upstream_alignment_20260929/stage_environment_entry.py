"""Transfer selected official assets and a separate, auditable entry candidate."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922'
AUDIT = Path(__file__).resolve().parent
REPO = AUDIT.parents[2]
SSH = ['ssh', '-oBatchMode=yes', '-oConnectTimeout=12', '-oStrictHostKeyChecking=yes',
       '-oServerAliveInterval=15', '-oServerAliveCountMax=3',
       '-oHostKeyAlias=[ssh.v5000-prod-gw.nhss.zhejianglab.com]:32036', '-p', '30821',
       'root@ssh.v5000-prod-gw.nhss.zhejianglab.com']
SCP = ['scp', '-oBatchMode=yes', '-oStrictHostKeyChecking=yes',
       '-oHostKeyAlias=[ssh.v5000-prod-gw.nhss.zhejianglab.com]:32036', '-P', '30821']
ENTRY = ROOT + '/receipts/environment-only-20260930/entry'

def remote(script):
    return subprocess.run(SSH+['bash', '-s'], input=script.encode(), check=True)

if __name__ == '__main__':
    # Small integration source overlay only; numerical DT stays in c9cd147.
    names = ['owner_environment_transport.py', 'patch_verl_environment_entry.py',
        'sql_environment_entry.py', 'loop_environment_entry.py', 'textcraft_environment_entry.py',
        'launch_sql_native.py', 'owner_task_dataset.py', 'owner_environment_configs.json',
        'reward_readout.py', 'dt_training_batch.py', 'deltatrace_rollout.py',
        'test_owner_environment_transport.py', 'test_sql_owner_rollout.py',
        'sql_owner_rollout.py', 'owner_trajectory_batch.py', 'patch_owner_trajectory_entry.py',
        'test_owner_trajectory_batch.py',
        'test_reward_readout.py', 'test_distributed_credit.py', 'test_loop_reward_readout.py']
    bundle = AUDIT / 'environment-entry-source.tar'
    with tarfile.open(bundle, 'w') as tar:
        for name in names:
            tar.add(REPO/'experiments/rl'/name, arcname=name)
        tar.add(AUDIT/'loop-project-environment-comparison.json', arcname='loop-project-environment-comparison.json')
    subprocess.run(SCP+[str(bundle), f'{SSH[-1]}:{ENTRY}/source.tar'], check=True)
    remote(f'tar -xf {ENTRY}/source.tar -C {ENTRY}\n')
    data_bundle = AUDIT/'sql-selected-data.tar.gz'
    if not data_bundle.exists():
        with tarfile.open(data_bundle, 'w:gz') as tar:
            for name in ['train.parquet', 'validation.parquet', 'source.json', 'db_files']:
                tar.add(AUDIT/'sql-data'/name, arcname=name)
    remote(f'mkdir -p {ROOT}/datasets/skyrl-sql-7e5e665\n')
    subprocess.run(SCP+[str(data_bundle), f'{SSH[-1]}:{ROOT}/datasets/skyrl-sql-7e5e665/source.tar.gz'], check=True)
    remote(f'tar -xzf {ROOT}/datasets/skyrl-sql-7e5e665/source.tar.gz -C {ROOT}/datasets/skyrl-sql-7e5e665\n')
    print(json.dumps(dict(entry=ENTRY, source_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        database_archive_bytes=data_bundle.stat().st_size)))
