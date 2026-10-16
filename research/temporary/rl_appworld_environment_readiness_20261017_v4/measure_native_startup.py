"""Measure original AppWorld service startup; no rollout, model, or optimizer."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import site
import sys
from threading import Lock
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/appworld-environment-readiness-20261017-v4'
source = json.loads((ROOT / 'runs/appworld-dtpo-reward-target-exponential-formal-20261017-v3/source.json').read_bytes())
os.environ.update(source['effective_environment'])
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.pop('MACA_VISIBLE_DEVICES', None)
sys.path[:0] = os.environ['PYTHONPATH'].split(os.pathsep)
site.addsitedir(os.environ['LOOP_EXTRAS'])

import psutil
import requests
from phi_agents.appworld.interface import DUMMY_TASK_ID, get_appworld_root
from phi_agents.appworld.server import launch_appworld_environment_server, stop_appworld_environment_server
from phi_agents.utils.network_utils import get_free_port

AUDIT.mkdir(parents=True, exist_ok=True)
started = time.time()
servers, measurements = [], []
server_lock = Lock()
timeout = 100  # Existing AppWorldInterface timeout_seconds default, not a new production setting.
record = dict(started=started, status='measuring', native_service_count=64,
              baseline_source=source['source_receipt'] if 'source_receipt' in source else str(ROOT/'runs'/source['version']/'source.json'),
              native_service_launcher=launch_appworld_environment_server.__module__,
              observation_deadline_seconds=timeout, model_calls=0, optimizer_updates=0, rollouts=0)


def measure(item):
    process, port, launched = item
    result = dict(pid=process.pid, port=port, launched=launched)
    # This only measures the original endpoint. The production owner state
    # machine is not used or replaced, and all processes are closed below.
    while time.monotonic() - launched < timeout:
        if process.poll() is not None:
            result.update(status='service_exited', returncode=process.returncode)
            return result
        try:
            response = requests.get(f'http://localhost:{port}/tasks/{DUMMY_TASK_ID}', timeout=timeout)
            if response.status_code == 200 and response.json()['output']['task_id'] == DUMMY_TASK_ID:
                result.update(status='ready', ready_seconds=time.monotonic()-launched)
                return result
        except requests.ConnectionError:
            pass
        except requests.Timeout:
            result.update(status='endpoint_timeout')
            return result
        time.sleep(1)
    result.update(status='not_ready_by_native_timeout')
    return result


try:
    # Native production starts two sets of 32 servers. Preserve that service
    # count here, without running a collector or sending any LLM request.
    def start_and_measure():
        port = get_free_port(verbose=False)
        launched = time.monotonic()
        process = launch_appworld_environment_server(port, False, get_appworld_root(), False)
        item = (process, port, launched)
        with server_lock:
            servers.append(item)
        return measure(item)
    with ThreadPoolExecutor(max_workers=64) as executor:
        for future in as_completed([executor.submit(start_and_measure) for _ in range(64)]):
            measurements.append(future.result())
            (AUDIT/'native-startup-measurement.json').write_text(json.dumps({**record, 'services':measurements},indent=2)+'\n')
    assert len({port for _, port, _ in servers}) == len(servers), 'Native free-port collision recorded'
    ready = [m['ready_seconds'] for m in measurements if m['status']=='ready']
    record.update(status='measurement_complete', services=measurements,
                  ready_count=len(ready), above_original_five_second_window=sum(x>5 for x in ready),
                  minimum_ready_seconds=min(ready,default=None), maximum_ready_seconds=max(ready,default=None))
finally:
    def close(item):
        process, port, _ = item
        if process.poll() is None:
            stop_appworld_environment_server(process)
        return dict(pid=process.pid,port=port,returncode=process.poll(),alive=psutil.pid_exists(process.pid))
    with ThreadPoolExecutor(max_workers=64) as executor:
        record['cleanup'] = list(executor.map(close,servers))
    record['finished'] = time.time()
    (AUDIT/'native-startup-measurement.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ('services','cleanup')}))
