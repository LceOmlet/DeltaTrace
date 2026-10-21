"""Archive the failed attempt, then submit one verified native continuation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from run_native_resume5 import AUDIT, RUN, PROCESS_OWNER, VERSION, ready, controller_source, sha


def put(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    import psutil
    row = ready()
    assert os.environ['RL_CAMPAIGN_COMMIT']
    assert not (AUDIT / 'resume5-controller-job.json').exists(), 'Inspect existing submission; never duplicate it'
    archive = AUDIT / 'attempt2-native-DT-host-allocation-failure'
    assert not archive.exists(), 'Inspect any partial arming before continuing; no automatic retry'
    assert json.loads((AUDIT / 'campaign-state.json').read_text())['status'] == 'needs_attention_no_failed_combination_skipped'
    assert (RUN / 'exit-code.txt').read_text().strip() == '1'
    archive.mkdir()
    saved = []
    audit_files = ['campaign-state.json', 'controller-job.json', 'controller.stdout', 'controller.stderr',
                   'dt-prepared.json', 'campaign-prepared.json', 'dt-final-source-snapshot.json',
                   'dt-latest-source-snapshot.json', 'formal-training.json', 'active-training.json', 'active-source.json',
                   'source/run_campaign.py', 'source/composed-native-process-owner.py', 'source/prepare_campaign.py']
    run_files = ['source.json', 'launch.json', 'exit-code.txt', 'driver.stderr.log',
                 'textcraft-dt/train.log', 'physical-GPU-before-launch.txt']
    for base, names, prefix in ((AUDIT, audit_files, 'audit'), (RUN, run_files, 'run')):
        for name in names:
            source = base / name
            if not source.is_file():
                continue
            destination = archive / prefix / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)  # Logs must not share an inode with later append writes.
            assert sha(source) == sha(destination)
            saved.append(dict(original=str(source), preserved=str(destination), SHA256=sha(destination)))
    put(archive / 'preservation-receipt.json', dict(status='original_failed_attempt_preserved_before_native_resume',
        preserved_unix=time.time(), files=saved, completed_iterations=5, native_exit_code=1,
        checkpoint5_unchanged=True))

    # Use the original exact-session owner for any CPU helpers left after exit.
    namespace = dict(__name__='original_failed_session_release_owner', __file__=str(PROCESS_OWNER))
    exec(compile(PROCESS_OWNER.read_text(), str(PROCESS_OWNER), 'exec'), namespace)
    cleanup = namespace['release_finished_session'](2654698, 1792605592.43)
    assert not cleanup['remaining_non_zombie']
    put(archive / 'original_failed_session_release.json', cleanup)

    commit = os.environ['RL_CAMPAIGN_COMMIT']
    row.update(local_patch_commit=commit, adapter_source_commit=commit, execution_source_commit=commit,
               execution_source_commit_scope='Same-run native checkpoint5 continuation plus accepted resource initialization; unchanged v10 mathematics',
               initial_fresh_execution_source_commit='bad0ae1d6a2209862fb13ff86bba6bbc1062f9af',
               resource_resume_bridge_source_commit=commit)
    put(AUDIT / 'resume5-dt-prepared.json', row)
    campaign = dict(status='v10_eight_GPU_formal_configuration_resolved', order=['dt'], jobs=[row])
    put(AUDIT / 'resume5-campaign-prepared.json', campaign)
    put(AUDIT / 'dt-prepared.json', row)
    put(AUDIT / 'campaign-prepared.json', campaign)
    os.replace(AUDIT / 'campaign-state.json', archive / 'campaign-state.from-active-path.json')
    os.replace(RUN / 'exit-code.txt', archive / 'exit-code.from-active-path.txt')
    put(AUDIT / 'resume5-armed.json', dict(status='preserved_failed_attempt_and_armed_one_native_resume',
        armed_unix=time.time(), archive=str(archive), run=str(RUN), checkpoint=row['resume_checkpoint'],
        source_commit=commit, training_driver_started=False))
    assert sha(AUDIT / 'source/composed-native-resume5-resource-owner.py') == \
        hashlib.sha256(controller_source().encode()).hexdigest()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    stdout, stderr = AUDIT / 'resume5-controller.stdout', AUDIT / 'resume5-controller.stderr'
    with stdout.open('xb') as out, stderr.open('xb') as err:
        child = subprocess.Popen([row['effective_environment']['VENV_PYTHON'],
                                  str(AUDIT / 'source/run_native_resume5.py')],
                                 cwd=AUDIT / 'source', env=environment, stdin=subprocess.DEVNULL,
                                 stdout=out, stderr=err, start_new_session=True)
    job = dict(status='submitted_one_native_same_run_resume5_controller', pid=child.pid,
               birth=psutil.Process(child.pid).create_time(), submitted_unix=time.time(),
               source_commit=commit, source=str(AUDIT / 'source/run_native_resume5.py'),
               source_SHA256=sha(AUDIT / 'source/run_native_resume5.py'),
               stdout=str(stdout), stderr=str(stderr), run=str(RUN),
               original_iteration_budget=330, resumed_completed_iteration=5,
               actual_native_worker_restoration_pending=True, no_automatic_retry_loop=True)
    put(AUDIT / 'resume5-controller-job.json', job)
    put(AUDIT / 'controller-job.json', job)
    print(json.dumps(job))


if __name__ == '__main__':
    main()
