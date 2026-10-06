"""Explicitly launch the prepared isolated probe; the preparer never calls this."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import psutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = Path(__file__).resolve().parent
    prepared = json.loads((out / 'prepared.json').read_bytes())
    assert prepared['prepared_only'] and not prepared['gpu_probe_started']
    assert not (out / 'job.json').exists() and not (out / 'result.json').exists()
    for path, expected in prepared['source_files'].items():
        assert digest(path) == expected, path
    for path, expected in prepared['literal_request_files'].items():
        assert digest(path) == expected, path
    library = prepared['individual_row_candidate']['finite_library']
    assert digest(library['path']) == library['sha256']
    env_path = out / 'run-env.json'
    assert digest(env_path) == prepared['run_environment_file']['sha256']
    env = json.loads(env_path.read_bytes())
    linked = prepared['isolated_link_owner']
    assert env['DT_ROOT'] == linked['dt_root']
    for item in linked['overrides'].values():
        assert Path(item['path']).resolve() == Path(item['target']).resolve()
        assert digest(item['path']) == item['sha256']
    assert env['CUDA_VISIBLE_DEVICES'] == '2,3'
    assert env['DT_PREFIX_PROBE_ROOT'] == str(out)
    assert env['DT_PREFIX_DIAGNOSTIC_OFFSET'] == '84'
    assert env['DT_PREFIX_DIAGNOSTIC_ROWS'] == '4'
    assert env['DT_PREFIX_INDIVIDUAL_ROW_CANDIDATE'] == '1'
    assert not env.get('DT_PREFIX_CHECKPOINT')
    assert all(env.get(key) != '1' for key in (
        'DT_PREFIX_HOT_PROFILE', 'DT_PREFIX_NATIVE_BACKWARD',
        'DT_PREFIX_ROOT_TAPE_HOT', 'DT_PREFIX_REVERSE_PREFETCH'))
    physical = subprocess.check_output(['mx-smi'], text=True)
    # Same occupancy expression used by the existing native probe stager.
    assert not any(re.search(r'^\|\s+' + str(device) + r'\s+\d+\s+',
                            physical.split('| Process:')[-1], re.M)
                   for device in (2, 3)), 'Selected GPUs 2/3 are occupied'
    log_path = out / 'probe.log'
    with log_path.open('xb') as log:
        child = subprocess.Popen(
            [env['VENV_PYTHON'], '-u', str(out / 'verify_native_prefix_artifacts.py')],
            cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    receipt = dict(
        pid=child.pid, pid_birth=psutil.Process(child.pid).create_time(),
        started_unix=time.time(), devices=[2, 3], log=str(log_path),
        prepared_sha256=digest(out / 'prepared.json'),
        command_owner='Unchanged original verify_native_prefix_artifacts.py',
        checkpoint_restore=False, optimizer_step=False, profiler=False,
        scope='Isolated saved-input B8 DT comparison; no formal deployment')
    (out / 'job.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
