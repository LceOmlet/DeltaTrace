"""One original read-only import observation for the current resumed lifetime.

Run only after its first complete native iteration and eight storage receipts.
The original metadata owner supplies the RPC; no recorder or computation is added.
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/textcraft-v10-eight-gpu-generation-aligned-20261022-v1'
RUN = ROOT / 'runs/textcraft-dtpo-v10-target-logprob-8gpu-formal-20261022-v2'
PHASE = 'after-bound-resource-DT-v1'


def main():
    source = json.loads((RUN / 'source.json').read_text())
    assert (source['pid'], source['birth']) == (3827242, 1792616307.66)
    import psutil
    driver = psutil.Process(source['pid'])
    assert driver.create_time() == source['birth'] and driver.status() != psutil.STATUS_ZOMBIE
    snapshot = json.loads((AUDIT / 'dt-latest-source-snapshot.json').read_text())
    job, = snapshot['jobs']
    assert job['pid'] == source['pid']
    assert max(row['step'] for row in job['metrics']) >= 6
    receipts = [json.loads(path.read_text()) for path in
                (RUN / 'runtime-overrides/accepted-storage-bound-resource-v2').glob('rank*.json')]
    assert len(receipts) == 8 and {row['rank'] for row in receipts} == set(range(8))
    assert all(row['birth'] >= source['birth'] for row in receipts)
    for path in AUDIT.glob('worker-imports-request-*.json'):
        existing = json.loads(path.read_text())
        existing_pid = existing.get('job_pid')
        if existing.get('receipt'):
            existing_pid = json.loads(Path(existing['receipt']).read_text()).get('job', {}).get('pid')
        assert existing_pid != source['pid'], 'Inspect the existing observation for this lifetime; never duplicate it'
    request = AUDIT / ('worker-imports-request-dt-' + PHASE + '.json')
    result = RUN / 'runtime-overrides' / ('actual-worker-imports-' + PHASE + '.json')
    assert not request.exists(), 'Retain existing queued or completed observation; never repeat its RPC'
    assert not result.exists(), 'Retain the completed observation'

    owner = AUDIT / 'source/inspect_native_workers_dp8.py'
    assert hashlib.sha256(owner.read_bytes()).hexdigest() == \
        '5e9f8bce8890f358f9d3729da836680e50a9176645b77f9a23162b457546f129'
    code = owner.read_text()
    anchor = "exec(compile(code, str(owner), 'exec'), globals())"
    assert code.count(anchor) == 1
    code = code.replace(anchor,
        "assert code.count(\"'after-first-DT-v1'\") == 1\n"
        "code = code.replace(\"'after-first-DT-v1'\", " + repr(repr(PHASE)) + ")\n" + anchor)
    sys.argv = [str(Path(__file__)), 'dt', '--after-DT'] + (['--client'] if '--client' in sys.argv else [])
    namespace = dict(__name__='__main__', __file__=str(Path(__file__)))
    exec(compile(code, str(owner), 'exec'), namespace)


if __name__ == '__main__':
    main()
