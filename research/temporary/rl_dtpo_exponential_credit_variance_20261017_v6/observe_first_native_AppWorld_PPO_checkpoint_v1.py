"""Wait for the live native PPO marker, then call the existing CPU verifier.

This observes one already running job. It never saves/restores a checkpoint,
submits training, sends a signal, or changes a training option. The original
trainer writes loop_sampler.pt before its completed-checkpoint marker.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/dtpo-exponential-credit-variance-20261017-v6'
CAMPAIGN = ROOT / 'receipts/appworld-formal-training-20261017-v1'
VERIFIER = CAMPAIGN / 'source/verify_completed_checkpoint.py'
VERIFIER_SHA256 = 'd5a6f3ee52e822356e5cb4d3df73215f3818ede9e033eda9e4b53719636ae865'
STATE = AUDIT / 'native-AppWorld-PPO-checkpoint2-observer-v1-job.json'


def write_state(value):
    temporary = STATE.with_name(STATE.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(STATE)


def main():
    with (AUDIT / 'native-AppWorld-PPO-checkpoint2-observer-v1.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(status='existing_checkpoint_observer_retained', state=str(STATE))))
            return
        assert hashlib.sha256(VERIFIER.read_bytes()).hexdigest() == VERIFIER_SHA256
        manifest = json.loads((ROOT / 'formal-training.json').read_bytes())
        job, = [j for j in manifest['jobs'] if j.get('task') == 'AppWorld' and j.get('method') == 'ppo']
        source = json.loads(Path(job['source_receipt']).read_bytes())
        environment = dict(os.environ)
        environment.update(source['effective_environment'])
        environment['CUDA_VISIBLE_DEVICES'] = ''
        environment.pop('MACA_VISIBLE_DEVICES', None)
        marker = Path(job['checkpoints']) / 'latest_checkpointed_iteration.txt'
        process = psutil.Process()
        state = dict(status='waiting_for_native_completed_checkpoint_marker',
                     pid=process.pid, birth=process.create_time(), started=time.time(),
                     target_pid=job['pid'], target_birth=job['birth'], target_run=job['run'],
                     target_version=job['version'], marker=str(marker),
                     source=str(Path(__file__)),
                     source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     observer_source_commit=os.environ.get('RL_OBSERVER_SOURCE_COMMIT'),
                     original_verifier=str(VERIFIER), original_verifier_SHA256=VERIFIER_SHA256,
                     training_actions=0, signals=0, model_calls=0, checkpoint_save_calls=0,
                     checkpoint_restore_calls=0, optimizer_updates=0, goal_status='active',
                     scope='First native shared checkpoint metadata only; not completion of the 200-iteration budget')
        print(json.dumps(state), flush=True)
        while True:
            try:
                driver = psutil.Process(job['pid'])
                alive = driver.create_time() == job['birth'] and driver.status() != psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                alive = False
            state['last_identity_verified_unix'] = time.time()
            if not alive:
                state.update(status='native_driver_terminal_before_checkpoint_verification', completed=time.time())
                write_state(state)
                print(json.dumps(state), flush=True)
                return
            # Empty content can be observed during the owner's short marker
            # write. It is a pending observation, not a training failure.
            content = marker.read_text().strip() if marker.exists() else ''
            if content:
                step = int(content)
                assert step > 0
                state.update(status='reading_completed_native_checkpoint_with_original_verifier',
                             observed_native_step=step, verification_started=time.time())
                write_state(state)
                result = subprocess.run([environment['VENV_PYTHON'], '-X', 'utf8', str(VERIFIER), 'ppo'],
                                        env=environment, capture_output=True, text=True)
                stdout = AUDIT / 'native-AppWorld-PPO-checkpoint2-verifier-v1.stdout'
                stderr = AUDIT / 'native-AppWorld-PPO-checkpoint2-verifier-v1.stderr'
                stdout.write_text(result.stdout)
                stderr.write_text(result.stderr)
                state.update(verifier_exit_code=result.returncode, stdout=str(stdout), stderr=str(stderr),
                             verification_finished=time.time())
                if result.returncode:
                    state['status'] = 'checkpoint_observation_verifier_failed_training_not_changed'
                else:
                    receipt = CAMPAIGN / f'{job["version"]}-checkpoint-{step}-verified.json'
                    verified = json.loads(receipt.read_bytes())
                    assert verified['status'] == 'passed_native_shared_checkpoint_state_metadata'
                    assert verified['step'] == step
                    state.update(status='completed_original_native_checkpoint_metadata_verification',
                                 original_verification_receipt=str(receipt),
                                 original_verification_receipt_SHA256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
                                 first_checkpoint_step_matches_save2=(step == 2))
                write_state(state)
                print(json.dumps(state), flush=True)
                return
            write_state(state)
            time.sleep(30)


if __name__ == '__main__':
    main()
