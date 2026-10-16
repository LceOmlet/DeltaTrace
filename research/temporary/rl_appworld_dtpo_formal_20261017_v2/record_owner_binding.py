"""Passive deployment receipt in the existing native LOOP environment process."""


def record_owner_binding(rank):
    import hashlib
    import json
    import os
    from pathlib import Path
    import sys
    import time
    import psutil

    modules = {}
    for name in ('phi_agents.evals.appworld_evals', 'phi_agents.appworld.interface',
                 'phi_agents.rl.appworld_scenario_runner'):
        path = getattr(sys.modules.get(name), '__file__', None)
        modules[name] = None if path is None else dict(path=path, SHA256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    directory = Path(os.environ['DT_FSDP_EVENT_WAIT_RECEIPTS'])
    record = dict(pid=os.getpid(), birth=psutil.Process().create_time(), rank=rank,
                  collected=time.time(), modules=modules, scope='Passive actual environment owner imports; no model or task calls')
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'loop-owner-{os.getpid()}.json').write_text(json.dumps(record, indent=2) + '\n')
    except OSError as exc:
        print(f'LOOP deployment receipt unavailable: {type(exc).__name__}', flush=True)
