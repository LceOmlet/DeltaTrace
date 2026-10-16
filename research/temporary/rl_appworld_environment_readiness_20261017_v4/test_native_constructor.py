"""Exercise all 64 original interface constructors; no sampling or model calls."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import site
import sys
from threading import Lock
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/appworld-environment-readiness-20261017-v4'
patch = json.loads((AUDIT/'owner-patch.json').read_bytes())
source = json.loads((ROOT/'runs/appworld-dtpo-reward-target-exponential-formal-20261017-v3/source.json').read_bytes())
os.environ.update({k:v.replace(patch['baseline'],patch['owner_root']) for k,v in source['effective_environment'].items()})
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.pop('MACA_VISIBLE_DEVICES',None)
sys.path[:0]=os.environ['PYTHONPATH'].split(os.pathsep)
site.addsitedir(os.environ['LOOP_EXTRAS'])
import psutil
from phi_agents.appworld import interface
assert Path(interface.__file__) == Path(patch['owner_root'])/'phi_agents/appworld/interface.py'
assert hashlib.sha256(Path(interface.__file__).read_bytes()).hexdigest()==patch['changes']['phi_agents/appworld/interface.py']['after_SHA256']
worlds, results, mutex = [], [], Lock()
record = dict(started=time.time(),status='testing_native_constructors', interface=interface.__file__,
              interface_SHA256=hashlib.sha256(Path(interface.__file__).read_bytes()).hexdigest(),
              model_calls=0,optimizer_updates=0,rollouts=0)


def construct(index):
    started=time.monotonic()
    world=interface.AppWorldInterface(stdout_to_devnull=False)
    with mutex:worlds.append(world)
    return dict(index=index,pid=world.server.pid,port=world.port,ready_seconds=time.monotonic()-started,
                max_wait_tries=world._max_wait_tries,wait_seconds=world._wait_seconds,
                max_restarts=world._max_restarts_on_error,timeout_seconds=world.timeout_seconds)


try:
    with ThreadPoolExecutor(max_workers=64) as executor:
        for future in as_completed([executor.submit(construct,i) for i in range(64)]):
            results.append(future.result())
    assert len(results)==64
    record.update(status='passed',constructors=results,ready_count=len(results),
                  minimum_ready_seconds=min(x['ready_seconds'] for x in results),
                  maximum_ready_seconds=max(x['ready_seconds'] for x in results))
finally:
    def close(world):
        pid=world.server.pid
        world.close_server()
        return dict(pid=pid,alive=psutil.pid_exists(pid),clean=world.clean)
    with ThreadPoolExecutor(max_workers=64) as executor:
        record['cleanup']=list(executor.map(close,worlds))
    record['finished']=time.time()
    (AUDIT/'native-constructor-verification.json').write_text(json.dumps(record,indent=2)+'\n')
assert not any(x['alive'] or not x['clean'] for x in record['cleanup'])
print(json.dumps({k:v for k,v in record.items() if k not in ('constructors','cleanup')}))
