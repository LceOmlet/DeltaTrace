"""Attach the saved passive observer through the original worker generic RPC.

Only this diagnostic's existing workers are touched. No restart, new rollout,
optimizer replacement or production source change. Stop after the captured
native update rather than permitting a second training iteration.
"""
import json
import os
from pathlib import Path
import time

import psutil
import ray


OUT = Path(os.environ['DT_ACTOR_INCIDENT_OUT'])


def install(worker):
    from types import MethodType
    from observe_native_first_iteration import install_native_observer, WorkerObservation

    result = install_native_observer(worker)

    def captured_update(owner, data):
        output = WorkerObservation.update_actor(owner, data)
        (OUT/f'rank{owner.rank}-diagnostic-stop.json').write_text(json.dumps(dict(
            unix=time.time(), rank=owner.rank,
            status='Native update captured; intentional diagnostic stop',
            production_repair=False, metrics=output.meta_info['metrics']),indent=2)+'\n')
        raise RuntimeError('INTENTIONAL_DIAGNOSTIC_STOP_AFTER_CAPTURED_NATIVE_UPDATE')

    for owner in worker.worker_dict.values():
        if getattr(owner,'_is_actor',False):
            owner.update_actor = MethodType(captured_update,owner)
    return result


def main():
    launch = json.loads((OUT/'launch.json').read_bytes())
    driver = psutil.Process(launch['pid'])
    assert driver.create_time() == launch['birth']
    assert not list(OUT.glob('rank*-actual-update-input.pt')), 'Update already started; do not replace a live update observer'
    assert not (OUT/'attached-observer.json').exists(), 'Do not duplicate installation'
    assert not list(OUT.glob('rank*-initial.json')), 'Observer already attached; preserve its records instead'
    ray.init(address='10.200.112.15:62721',logging_level='ERROR')
    expected = {264116,265959}
    actors = ray._private.state.actors()
    owned = {a['Name']:a for a in actors.values() if a.get('Pid') in expected}
    assert set(a['Pid'] for a in owned.values()) == expected
    named = {a['name']:a for a in ray.util.list_named_actors(all_namespaces=True)}
    handles = [ray.get_actor(name,namespace=named[name]['namespace']) for name in sorted(owned)]
    references = [handle.execute_with_func_generator.remote(install) for handle in handles]
    results = ray.get(references,timeout=240)
    value = dict(unix=time.time(),driver_pid=launch['pid'],driver_birth=launch['birth'],
        workers={name:{key:value[key] for key in ('Name','Pid','ActorID','JobID','State')}
                 for name,value in owned.items()},results=results,
        reason='Trainer subclass observation did not install; reuse existing native workers before their first update.',
        stop='Intentional RPC failure only after both native rank updates complete and save their records.',
        changed_policy=False,changed_optimizer=False,changed_credit=False,
        restarted=False,formal_training=False)
    (OUT/'attached-observer.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))
    ray.shutdown()


if __name__ == '__main__':
    main()
