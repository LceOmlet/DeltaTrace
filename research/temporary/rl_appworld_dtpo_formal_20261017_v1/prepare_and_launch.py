"""Compose and launch the verified AppWorld owner at its formal workload.

Configuration and provenance only. Native LOOP owns sampling and environment;
native VERL owns all updates, checkpoints, data cursor and termination.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION = 'appworld-dtpo-reward-target-exponential-formal-20261017-v1'
BASE = ROOT / 'candidates/appworld-sql-native-speed-20261016-v3/AppWorld/prepared.json'
DEST = ROOT / 'candidates' / VERSION
AUDIT = ROOT / 'receipts' / VERSION
RUN = ROOT / 'runs' / VERSION
OUTPUT = RUN / 'appworld-dt'
SHARED = Path('/mnt/si0021787ci2/default/lzq/deltatrace_backups/appworld-20261017') / VERSION / 'checkpoints'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def prepare():
    assert not DEST.exists(), 'Read existing preparation; never overwrite its source'
    baseline = read(BASE)
    assert baseline['task'] == 'AppWorld'
    assert all(sha(path) == digest for path, digest in baseline['source_bindings'].items())
    old_entry = baseline['entry']
    entry = DEST / 'entry'
    entry.mkdir(parents=True)
    for path in Path(old_entry).iterdir():
        if path.is_file():
            shutil.copyfile(path, entry / path.name)
    env = dict(baseline['effective_environment'])
    for key, value in list(env.items()):
        if isinstance(value, str):
            env[key] = value.replace(old_entry, str(entry)).replace(baseline['run'], str(OUTPUT))
    env.pop('DT_LOOP_OVERRIDES_JSON', None)
    for key in ('RAY_ADDRESS', 'RAY_TMPDIR', 'MACA_VISIBLE_DEVICES'):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES='6,7', DT_ENTRY_ROOT=str(entry),
               DT_FSDP_EVENT_WAIT_RECEIPTS=str(OUTPUT / 'worker-runtime-overrides'))
    assert env.get('FLASH_ATTENTION_DETERMINISTIC', '0') == '0'
    cpu = dict(env, CUDA_VISIBLE_DEVICES='')
    # The original entry resolves the author's README/Hydra recipe. Restore
    # formal workload by removing only the old small-trial override above.
    compose = '''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import launch_appworld_native as owner
options,sampling=owner.options_for(Path(sys.argv[2]))
print(json.dumps(dict(options=options,sampling=sampling)))
'''
    native = json.loads(subprocess.check_output(
        [env['VENV_PYTHON'], '-c', compose, str(entry), str(OUTPUT)], env=cpu, text=True))
    options = native['options']
    assert options['data.train_batch_size'] == options['+data.loop_train_groups'] == 40
    assert options['env.rollout.n'] == 6
    assert options['trainer.total_training_steps'] == options['trainer.total_epochs'] == 200
    assert options['actor_rollout_ref.actor.ppo_mini_batch_size'] == 32
    assert options['actor_rollout_ref.actor.ppo_epochs'] == 2
    # Keep the interface/graph/event-wait settings already executed successfully
    # in the two native AppWorld iterations; no alternative trainer is used.
    options.update({
        'trainer.experiment_name': VERSION,
        'trainer.default_local_dir': str(SHARED),
        'trainer.save_freq': 2,
        'trainer.resume_mode': 'disable', 'trainer.resume_from_path': None,
        'actor_rollout_ref.rollout.mode': 'sync',
        'actor_rollout_ref.rollout.enforce_eager': False,
        'actor_rollout_ref.rollout.free_cache_engine': False,
        '+ray_init.runtime_env.worker_process_setup_hook': 'metax_fsdp_event_wait.worker_setup',
        '+ray_init.runtime_env.env_vars.DT_FSDP_EVENT_WAIT_RECEIPTS': env['DT_FSDP_EVENT_WAIT_RECEIPTS'],
        '+ray_init.runtime_env.env_vars.DT_CREDIT_MODE': 'reward_target_exponential_v5',
    })
    options.pop('+ray_init.runtime_env.env_vars.DT_WORKER_VISIBILITY_DIR', None)
    env.update(DT_MAX_STEPS=str(options['env.max_steps']), DT_MAX_LENGTH='32768',
               DT_SAMPLING_JSON=json.dumps(native['sampling']))
    command = '''import json,sys
sys.path.insert(0,sys.argv[1])
from owner_runtime_options import owner_command
print(json.dumps(owner_command(json.loads(sys.stdin.read()))))
'''
    argv = json.loads(subprocess.check_output([env['VENV_PYTHON'], '-c', command, str(entry)],
        input=json.dumps(options), env=cpu, text=True))
    assert argv[0] == env['VENV_PYTHON']
    check = subprocess.run(argv + ['--cfg', 'job', '--resolve'], cwd=entry, env=cpu, capture_output=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / 'effective-config.yaml').write_bytes(check.stdout)
    (AUDIT / 'CPU-config.stderr.log').write_bytes(check.stderr)
    check.check_returncode()
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(check.stdout.decode())
    assert cfg.algorithm.adv_estimator == 'deltatrace'
    assert cfg.trainer.save_freq == 2 and cfg.trainer.max_actor_ckpt_to_keep == 2
    assert cfg.trainer.default_local_dir == str(SHARED)
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 4
    assert cfg.actor_rollout_ref.rollout.max_model_len == 32768
    assert cfg.trainer.total_training_steps == 200 and cfg.trainer.test_freq == 5
    SHARED.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.TemporaryFile(dir=SHARED) as probe:
        probe.write(b'native checkpoint destination writable\n');probe.flush();os.fsync(probe.fileno())
    bindings = {path.replace(old_entry, str(entry)): digest for path, digest in baseline['source_bindings'].items()}
    assert all(sha(path) == digest for path, digest in bindings.items())
    record = dict(version=VERSION, task='AppWorld', method='dt', entry=str(entry), run=str(RUN), output=str(OUTPUT),
        checkpoints=str(SHARED), argv=argv, options=options, effective_environment=env,
        source_bindings=bindings, baseline_verified_preparation=str(BASE),
        method_source_commit='4545c55807e7d002287717b6a4254f9b083dd5fc',
        upstream_commits=dict(VERL='20bd331bdbc9026a5668e11362178e10ab7400c8',
                             LOOP='f14107a976e5793990329d3193df4742076c5a1d'),
        local_patch_commit=os.environ['APPWORLD_FORMAL_SOURCE_COMMIT'], physical_gpus=[6,7],
        budget=dict(iterations=200, groups_per_iteration=40, requested_rollouts_per_group=6,
                    native_cancellation_preserved=True, optimizer_minibatch=32, ppo_epochs=2,
                    checkpoint_interval=2, actor_checkpoints_to_keep=2, evaluation_interval=5),
        status='CPU_configuration_verified_not_started', numerical_method_changed=False,
        model_DT_optimizer_calls_in_preparation=0)
    write(DEST / 'prepared.json', record)
    write(AUDIT / 'CPU-verification.json', dict(status='passed', effective_config_SHA256=sha(AUDIT/'effective-config.yaml'),
        original_author_workload_verified=True, unchanged_source_bindings=len(bindings),
        checkpoint_interval=2, checkpoint_path=str(SHARED), checkpoint_destination_writable=True,
        checkpoint_created=False, new_numerical_tests=False, reused_verified_method_commit=record['method_source_commit']))
    print(json.dumps({k:record[k] for k in ('version','status','physical_gpus','budget','checkpoints')}))


def launch():
    import psutil
    record = read(DEST / 'prepared.json')
    assert record['local_patch_commit'] == os.environ['APPWORLD_FORMAL_SOURCE_COMMIT']
    assert not (RUN / 'launch.json').exists(), 'Inspect current identity instead of starting a duplicate'
    assert all(sha(path) == digest for path, digest in record['source_bindings'].items())
    smi = subprocess.check_output(['mx-smi'], text=True)
    assert not any(re.match(r'\|\s*(?:6|7)\s+\d+\s+\S+', line) for line in smi.splitlines()), 'Physical GPU6/7 occupied'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ);env.update(record['effective_environment'])
    env.pop('MACA_VISIBLE_DEVICES', None)
    log = OUTPUT / 'train.log'
    with log.open('ab', buffering=0) as stream:
        process = subprocess.Popen(record['argv'], cwd=record['entry'], env=env,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    birth = psutil.Process(process.pid).create_time()
    record.update(pid=process.pid, birth=birth, observed_process_created_unix=birth,
        started_unix=time.time(), log=str(log), status='submitted_not_yet_healthy', devices=[6,7])
    write(RUN / 'launch.json', record)
    write(RUN / 'source.json', record)
    write(OUTPUT / 'job.json', dict(record, source_receipt=str(RUN/'source.json')))
    job = {key:record[key] for key in ('task','method','pid','birth','observed_process_created_unix',
        'started_unix','devices','output','log','checkpoints','budget','version','status','entry')}
    job.update(source_receipt=str(RUN/'source.json'), verl_root=env['VERL_ROOT'],dt_root=env['DT_ROOT'],
               local_patch_commit=record['local_patch_commit'], method_source_commit=record['method_source_commit'])
    receipt = AUDIT / 'formal-training.json'
    write(receipt, dict(jobs=[job], source_receipt=str(RUN/'source.json')))
    for name in ('formal-training.json','active-training.json','active-source.json'):
        path = ROOT / name
        manifest = read(path)
        write(AUDIT / ('prior-' + name), manifest)
        prior = manifest['jobs']
        preserved = [j for j in prior if not (j.get('task')=='AppWorld' and j.get('method')=='dt')]
        manifest['jobs'] = preserved + [job]
        manifest['appworld_formal_training_current'] = str(receipt)
        manifest['appworld_formal_updated_unix'] = time.time()
        write(path, manifest)
    write(AUDIT / 'launch-identity.json', dict(pid=process.pid,birth=birth,version=VERSION,
        local_patch_commit=record['local_patch_commit'],method_source_commit=record['method_source_commit'],
        physical_gpus=[6,7],checkpoints=str(SHARED),status='submitted_not_yet_healthy'))
    print(json.dumps(dict(pid=process.pid,birth=birth,status=record['status'],budget=record['budget'],checkpoints=str(SHARED))))


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--launch', action='store_true')
args=parser.parse_args()
launch() if args.launch else prepare()
