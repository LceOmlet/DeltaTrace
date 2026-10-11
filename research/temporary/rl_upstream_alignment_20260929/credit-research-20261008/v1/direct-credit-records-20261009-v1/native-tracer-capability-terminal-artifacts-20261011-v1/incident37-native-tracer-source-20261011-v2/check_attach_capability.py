"""Exercise the installed mcTracer attach/stop interface; no model or PPO."""
import hashlib
import json
import os
from pathlib import Path
import pty
import signal
import subprocess
import sys
import time
import traceback

import psutil


def main():
    root = Path(sys.argv[1])
    root.mkdir(exist_ok=False)
    assert 'no process found' in subprocess.check_output(['mx-smi'], text=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='4')
    code = ('import torch,time; x=torch.empty(1024,device="cuda"); '
            'print("ready",flush=True); '
            '\nfor i in range(40):\n x.fill_(i); time.sleep(.25)\n')
    with (root / 'child.log').open('wb') as output:
        child = subprocess.Popen([sys.executable, '-u', '-c', code], env=env,
                                 stdout=output, stderr=subprocess.STDOUT)
    child_birth = psutil.Process(child.pid).create_time()
    tracer = None
    master = slave = None
    report = dict(child_pid=child.pid, child_birth=child_birth,
                  model_calls=0, production_changes=0)
    try:
        for _ in range(100):
            if b'ready' in (root / 'child.log').read_bytes():
                break
            if child.poll() is not None:
                raise RuntimeError('Capability child exited before ready')
            time.sleep(.1)
        else:
            raise TimeoutError('Capability child did not initialize')
        master, slave = pty.openpty()
        # The installed tool prepends its cwd even to an absolute odname.
        # Keep the first failed capability artifacts; launch v2 in a new dir.
        command = ['/opt/maca/bin/mcTracer', '--odname', 'trace',
                   '--name', 'capability', '--attach', str(child.pid)]
        with (root / 'tracer.log').open('wb') as output:
            tracer = subprocess.Popen(command, stdin=slave, stdout=output,
                                      stderr=subprocess.STDOUT, start_new_session=True,
                                      cwd=root)
        report.update(command=command, tracer_pid=tracer.pid,
                      tracer_birth=psutil.Process(tracer.pid).create_time())
        time.sleep(3)
        os.write(master, b'\x14')  # Installed help: ctrl+t stops attach.
        report['stop_key_unix'] = time.time()
        try:
            report['tracer_returncode'] = tracer.wait(timeout=12)
        except subprocess.TimeoutExpired:
            report['tracer_stop_timeout'] = True
            tracer.terminate()
            report['tracer_returncode'] = tracer.wait(timeout=5)
        report['child_returncode'] = child.wait(timeout=12)
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        for process in (tracer, child):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        for fd in (master, slave):
            if fd is not None:
                os.close(fd)
        report['files'] = [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size,
                               sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                           for p in root.rglob('*') if p.is_file()]
        (root / 'capability.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report))


if __name__ == '__main__':
    main()
