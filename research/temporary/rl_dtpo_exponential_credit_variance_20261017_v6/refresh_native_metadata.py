"""Record exact native Ray sessions and budget aliases for the original collector."""
from pathlib import Path
import json
import psutil
import time

from prepare_formal import ROOT, AUDIT, write

facts = {}
manifest = json.loads((ROOT / 'formal-training.json').read_bytes())
for job in manifest['jobs']:
    if job.get('task') not in ('TextCraft', 'AppWorld') or job.get('method') not in ('dt', 'ppo'):
        continue
    driver = psutil.Process(job['pid'])
    assert driver.create_time() == job['birth']
    gcs = next((p for p in driver.children(recursive=True) if p.name() == 'gcs_server'), None)
    if gcs is None:
        continue
    log_dir = next(a.split('=', 1)[1] for a in gcs.cmdline() if a.startswith('--log_dir='))
    session = str(Path(log_dir).parent)
    budget = job['budget'].get('total_iterations', job['budget'].get('iterations'))
    facts[(job['task'], job['method'])] = dict(pid=job['pid'], birth=job['birth'], ray_session=session,
                                              paper_budget={'iterations': int(budget)})
for name in ('formal-training.json', 'active-training.json', 'active-source.json',
             'receipts/appworld-formal-training-20261017-v1/formal-training.json'):
    path = ROOT / name
    current = json.loads(path.read_bytes())
    for job in current['jobs']:
        fact = facts.get((job.get('task'), job.get('method')))
        if fact is not None:
            assert (job['pid'], job['birth']) == (fact['pid'], fact['birth'])
            job.update(fact)
    write(path, current)
write(AUDIT / 'native-Ray-sessions-budget-aliases.json', dict(collected=time.time(), jobs=list(facts.values()),
      scope='Exact current driver GCS log_dir and original budget alias only; no log parsing or training calls'))
print(json.dumps(list(facts.values())))
