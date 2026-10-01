"""Keep pending owner requests together until the original Ray pool has room.

Local preparation only. No live source, task setting or numerical code changes.
"""
from pathlib import Path
import difflib
import hashlib
import json


def main():
    audit = Path(__file__).resolve().parent
    prior = audit / 'appworld-rank-completion-20261002' / 'loop_owner_rollout.py'
    expected = '8e93ec140303a88e35d8ff2a9a50aa1fc8012215e979bb9d8d5967da2a95b030'
    assert hashlib.sha256(prior.read_bytes()).hexdigest() == expected
    source = prior.read_text(encoding='utf8')
    replacements = {
        '        while len(results) < len(pool.processes) or engines.has_next():\n'
        '            event = pool.event() if not pending else None\n':
        '        while len(results) < len(pool.processes) or engines.has_next() or pending:\n'
        '            event = None if pending and engines.has_free() else pool.event()\n',
        '            requests, pending = pending, []\n':
        '            # ActorPool owns availability. While both actors are busy,\n'
        '            # retain exact requests here rather than freezing each new\n'
        '            # arrival into its own queued RPC. No timer or batch cap.\n'
        '            if engines.has_free():\n'
        '                requests, pending = pending, []\n'
        '            else:\n'
        '                requests = []\n',
    }
    after = source
    for before, replacement in replacements.items():
        assert after.count(before) == 1
        after = after.replace(before, replacement)
    compile(after, 'completion-batching/loop_owner_rollout.py', 'exec')
    target = audit / 'appworld-batch-coalescing-20261002'
    assert not target.exists(), 'Preserve existing candidate and diagnostic receipts'
    target.mkdir()
    path = target / prior.name
    path.write_text(after, encoding='utf8', newline='\n')
    (target / 'candidate.diff').write_text(''.join(difflib.unified_diff(
        source.splitlines(True), after.splitlines(True),
        fromfile='deployed/loop_owner_rollout.py', tofile='candidate/loop_owner_rollout.py', n=0)),
        encoding='utf8')
    (target / 'source.json').write_text(json.dumps(dict(
        status='prepared_local_not_tested_or_deployed', prior_source=str(prior),
        prior_sha256=expected, candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        owner_apis=['Ray ActorPool.has_free', 'Ray ActorPool.submit',
                    'Ray ActorPool.get_next_unordered', 'VERL DataProto.chunk'],
        observed_defect='174 of 181 nonempty native replies had batch size one while incoming requests were queued as separate busy-actor RPCs.',
        scope='Only RPC batching while the native actors are busy; original allocation, completion order, cancellation, artifacts and all task/training/numerical settings retained.'),
        indent=2) + '\n', encoding='utf8')
    print(path)


if __name__ == '__main__':
    main()
