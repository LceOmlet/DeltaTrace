"""CPU process-lifecycle tests for the exact one-time observer script.

The checkpoint files are metadata placeholders. No checkpoint contents,
VERL loader, model, training formula or GPU behavior are tested here.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
try:
    import psutil
except ModuleNotFoundError:
    psutil = None


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HELPER = HERE / 'resume_at_native_checkpoint.py'
OLD_REVISION = 'e87c5935a781a4bb7a91a97b173cdf1d612e4204'
FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def generated_python(source, root, *, stop_only=False):
    """Capture the actual main() output without contacting a server."""
    fixture_source = root / 'observer-source.py'
    fixture_source.write_text(source, encoding='utf8')
    namespace = {'__name__': 'observer_fixture', '__file__': str(fixture_source)}
    exec(compile(source, str(fixture_source), 'exec'), namespace)
    captured = []

    class Captured(Exception):
        pass

    def capture(script):
        captured.append(script)
        raise Captured

    namespace.update(ROOT=root.as_posix(), REPO=REPO, remote=capture)
    before = sys.argv
    sys.argv = ['observer_fixture', '--task', 'TextCraft', '--minimum-step', '1',
                '--prepared', str(root / 'prepared.json'),
                '--receipt', str(root / 'receipt')]
    sys.argv += ['--stop-only'] if stop_only else ['--run-dir', str(root / 'run')]
    try:
        with pytest.raises(Captured):
            namespace['main']()
    finally:
        sys.argv = before
    assert len(captured) == 1
    return captured[0].split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]


def run_observer_case(root, source, event, *, stop_only=False):
    if psutil is None:
        pytest.skip('Existing process-lifecycle tests require provisioned psutil; no replacement or installation is used')
    root.mkdir()
    tracked = []

    def spawn(code):
        p = subprocess.Popen([sys.executable, '-X', 'utf8', '-c', code],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             creationflags=FLAGS)
        tracked.append((p.pid, psutil.Process(p.pid).create_time()))
        return p

    def alive(p):
        return p.poll() is None

    try:
        child_file = root / 'child.txt'
        own = spawn('import pathlib,subprocess,sys,time; '
                    'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(120)"],'
                    f'creationflags={FLAGS}); '
                    f'pathlib.Path({str(child_file)!r}).write_text(str(p.pid));time.sleep(120)')
        until = time.monotonic() + 8
        while not child_file.exists():
            assert alive(own) and time.monotonic() < until
            time.sleep(.01)
        child = psutil.Process(int(child_file.read_text()))
        tracked.append((child.pid, child.create_time()))
        other = spawn('import time;time.sleep(120)')

        def job(task, p):
            return dict(task=task, pid=p.pid,
                        observed_process_created_unix=psutil.Process(p.pid).create_time(),
                        entry=str(root / 'entry'), verl_root=str(root / 'owner'),
                        dt_root=str(root / 'dt'), source_receipt=str(root / 'source.json'),
                        checkpoints=str(root / 'checkpoints'))

        own_job, other_job = job('TextCraft', own), job('AppWorld', other)
        active = root / 'active-training.json'
        active.write_text(json.dumps(dict(jobs=[own_job, other_job])))
        hashes = {}
        for name in ('entry', 'owner', 'dt'):
            directory = root / name
            directory.mkdir()
            file = directory / 'fixture.py'
            file.write_text('# CPU metadata fixture only\n')
            hashes[name] = {'fixture.py': hashlib.sha256(file.read_bytes()).hexdigest()}
        prepared = dict(prior_driver_pid=own.pid, prior_entry=own_job['entry'],
                        prior_verl_root=own_job['verl_root'],
                        future_checkpoint_root=own_job['checkpoints'],
                        entry=own_job['entry'], verl_root=own_job['verl_root'],
                        dt_root=str(root / 'dt'), entry_sha256=hashes['entry'],
                        owner_sha256=hashes['owner'], dt_source_sha256=hashes['dt'])
        if stop_only:
            # A resumed job uses prepared.entry; prepared.prior_* describes
            # its predecessor and must not be used to identify the current tree.
            prepared.update(prior_driver_pid=-1, prior_entry='previous-entry',
                            prior_verl_root='previous-owner', future_checkpoint_root='previous-checkpoints')
            current_source = dict(dt_root=own_job['dt_root'], verl_root=own_job['verl_root'],
                                  entry_sha256=hashes['entry'], verl_sha256=hashes['owner'],
                                  owner_head_sha256=hashes['owner'])
            if event == 'wrong_current_source':
                current_source['dt_root'] = 'different-root'
            (root / 'source.json').write_text(json.dumps(current_source))
        if event == 'wrong_birth':
            own_job['observed_process_created_unix'] -= 60
            active.write_text(json.dumps(dict(jobs=[own_job, other_job])))
        if event == 'wrong_source':
            prepared['entry_sha256']['fixture.py'] = '0' * 64
        (root / 'prepared.json').write_text(json.dumps(prepared))
        checkpoint = root / 'checkpoints/global_step_1'
        (checkpoint / 'actor').mkdir(parents=True)
        files = [checkpoint / 'data.pt'] + [checkpoint / 'actor' / f'{kind}_world_size_2_rank_{rank}.pt'
                 for kind in ('model', 'optim', 'extra_state') for rank in range(2)]
        for file in files:
            file.write_bytes(b'CPU metadata fixture, not a PyTorch checkpoint')
        if event == 'incomplete':
            files[-1].write_bytes(b'')

        code = generated_python(source, root, stop_only=stop_only)
        observer = subprocess.Popen([sys.executable, '-X', 'utf8', '-c', code],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, encoding='utf8', creationflags=FLAGS)
        tracked.append((observer.pid, psutil.Process(observer.pid).create_time()))
        replacement = None
        if event not in ('wrong_source', 'wrong_current_source', 'wrong_birth'):
            until = time.monotonic() + 8
            while not (root / 'receipt/waiting.json').exists():
                assert observer.poll() is None and time.monotonic() < until
                time.sleep(.01)
            assert alive(own)  # No stop before the original completion marker.
            if event in ('replaced', 'exited'):
                other.terminate()
                other.wait(timeout=4)
            if event == 'replaced':
                replacement = spawn('import time;time.sleep(120)')
                active.write_text(json.dumps(dict(jobs=[own_job, job('AppWorld', replacement)])))
            (root / 'checkpoints/latest_checkpointed_iteration.txt').write_text('1')
        stdout, stderr = observer.communicate(timeout=28)
        stopped = root / 'receipt/completed-stop.json'
        result = dict(returncode=observer.returncode, stdout=stdout, stderr=stderr,
                      own_alive=alive(own), child_alive=child.is_running(),
                      other_alive=alive(replacement or other),
                      checkpoint_files_preserved=all(file.is_file() for file in files),
                      stopped=json.loads(stopped.read_text()) if stopped.exists() else None,
                      helper_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      scope='CPU process lifecycle only; placeholder checkpoint metadata')
        (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
    finally:
        for pid, created in reversed(tracked):
            try:
                p = psutil.Process(pid)
                if abs(p.create_time() - created) < .02:
                    p.kill()
                    p.wait(timeout=4)
            except psutil.NoSuchProcess:
                pass


@pytest.mark.parametrize('event', ['unchanged', 'replaced', 'exited', 'wrong_source'])
def test_exact_observer_lifecycle(tmp_path, event):
    if event == 'replaced':
        old = subprocess.check_output(['git', 'show', f'{OLD_REVISION}:{HELPER.relative_to(REPO).as_posix()}'],
                                      cwd=REPO).decode('utf8')
        baseline = run_observer_case(tmp_path / 'old', old, event)
        assert baseline['returncode'] != 0
        assert 'Another original job changed' in baseline['stderr']
        assert not baseline['own_alive'] and baseline['other_alive']
    result = run_observer_case(tmp_path / 'current', HELPER.read_text(encoding='utf8'), event)
    if event == 'wrong_source':
        assert result['returncode'] != 0 and result['own_alive']
        assert result['stopped'] is None
    else:
        assert result['returncode'] == 0, result['stderr']
        assert not result['own_alive'] and not result['child_alive']
        assert result['stopped']['remaining_non_zombie'] == []
        assert result['stopped']['other_jobs_unchanged']['AppWorld'] == (event != 'exited')
        assert result['other_alive'] == (event != 'exited')


@pytest.mark.parametrize('event', ['unchanged', 'wrong_source', 'wrong_current_source', 'wrong_birth', 'incomplete'])
def test_stop_only_current_prepared_identity(tmp_path, event):
    result = run_observer_case(tmp_path / 'current', HELPER.read_text(encoding='utf8'), event,
                               stop_only=True)
    assert result['checkpoint_files_preserved']
    assert result['other_alive']
    if event == 'unchanged':
        assert result['returncode'] == 0, result['stderr']
        assert not result['own_alive'] and not result['child_alive']
        assert result['stopped']['stop_only'] is True
        assert result['stopped']['marker_step'] == 1
        assert result['stopped']['reused_loaded_checkpoint'] is False
        assert result['stopped']['remaining_non_zombie'] == []
    else:
        assert result['returncode'] != 0 and result['own_alive'] and result['child_alive']
        assert result['stopped'] is None


def test_stop_only_returns_without_submission(tmp_path, monkeypatch):
    source = HELPER.read_text(encoding='utf8')
    namespace = {'__name__': 'observer_fixture', '__file__': str(HELPER)}
    exec(compile(source, str(HELPER), 'exec'), namespace)
    scripts, calls = [], []
    audit = tmp_path / 'audit'
    audit.mkdir()

    def fake_run(argv, **kwargs):
        calls.append(argv)
        assert argv[:len(namespace['SCP'])] == namespace['SCP'], 'Stop-only must not submit a job'
        Path(argv[-1]).write_text(json.dumps({'checkpoint': 'original/global_step_20'}))

    namespace.update(AUDIT=audit, remote=scripts.append,
                     subprocess=SimpleNamespace(check_output=lambda *a, **k: 'a' * 40,
                                                run=fake_run))
    monkeypatch.setattr(sys, 'argv', ['observer_fixture', '--task', 'AppWorld', '--minimum-step', '20',
                                    '--prepared', 'prepared.json', '--receipt', 'receipt', '--stop-only'])
    namespace['main']()
    assert len(scripts) == len(calls) == 1


@pytest.mark.parametrize('flags', [[], ['--stop-only', '--reuse-loaded-checkpoint']])
def test_checkpoint_observer_cli_rejects_invalid_modes(monkeypatch, flags):
    namespace = {'__name__': 'observer_fixture', '__file__': str(HELPER)}
    exec(compile(HELPER.read_text(encoding='utf8'), str(HELPER), 'exec'), namespace)
    namespace['remote'] = lambda _: pytest.fail('Invalid CLI must not contact the server')
    monkeypatch.setattr(sys, 'argv', ['observer_fixture', '--task', 'AppWorld', '--minimum-step', '20',
                                    '--prepared', 'prepared.json', '--receipt', 'receipt', *flags])
    with pytest.raises(SystemExit) as failure:
        namespace['main']()
    assert failure.value.code == 2
