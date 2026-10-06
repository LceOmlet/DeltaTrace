"""Check the two-clause query repair against saved native IDs, without a model.

The saved clock experiment already scored these exact queries. This checks only
encoding, original-prefix/target identity and the existing adapter unit tests;
it is not evidence of a task-gradient or training-quality repair.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import torch
from transformers import AutoTokenizer
from reward_readout import RewardAlphabet


def artifact(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size)


def main():
    started = time.time()
    root = Path(os.environ['DT_QUERY_REPAIR_ROOT'])
    native = root.parent / 'textcraft-native-minibatch-20261006-v4' / 'readout-first-response-cases.json'
    clock = root.parent / 'textcraft-response-clock-20261006-v2'
    assert artifact(native)['sha256'] == '3ca15c0ac56b0d73fb3b5948330773d95ffb1990e6fe0edca53fbf5b67575d41'
    pack = json.loads(native.read_bytes())
    tokenizer = AutoTokenizer.from_pretrained('/mnt/si0021787ci2/default/models/Qwen3.5-9B',
                                              local_files_only=True)
    alphabet = RewardAlphabet.for_task('TextCraft')
    records, sources = [], [artifact(native), artifact(inspect.getfile(RewardAlphabet)),
                            artifact(__file__)]
    for rank in range(2):
        path = clock / f'rank{rank}-readout.json'
        observed = json.loads(path.read_bytes())
        sources.append(artifact(path))
        assert observed['input_sha256'] == artifact(native)['sha256']
        assert len(pack['rank_cases'][rank]) == len(observed['cases']) == 32
        for case, saved in zip(pack['rank_cases'][rank], observed['cases']):
            assert case['traj_uid'] == saved['traj_uid']
            ids, start, end = case['selected_input_ids'], case['source_start'], case['source_end']
            args = dict(current_step=case['source_step'], max_steps=pack['max_steps'], sampling=pack['sampling'])
            query = alphabet.query_ids(tokenizer, **args)
            assert ids[end:-1] == saved['query']['original_ids']
            assert query == saved['query']['corrected_ids']
            assert tokenizer.decode(query) == saved['query']['corrected_text']
            labels = alphabet.label_ids(tokenizer)
            assert labels == case['outcome_token_ids'] == [15, 16]
            target = labels[alphabet.observed_index(case['observed_return'])]
            assert target == ids[-1] == case['target_id']
            repaired = ids[:end] + query + [target]
            reference = repaired.copy()
            reference[start:end] = [tokenizer.eos_token_id] * (end - start)
            assert repaired[:end] == ids[:end]
            assert reference[:start] == repaired[:start] and reference[end:] == repaired[end:]
            assert end - start == len(ids[start:end])
            records.append(dict(rank=rank, traj_uid=case['traj_uid'], source_step=case['source_step'],
                source_start=start, source_end=end, observed_return=case['observed_return'], target_id=target,
                original_context_tokens=len(ids), repaired_context_tokens=len(repaired),
                corrected_query_matches_native_candidate=True, original_prefix_and_target_preserved=True))
    test = root / 'test_reward_readout.py'
    with (root / 'pytest.stdout.txt').open('wb') as log:
        run = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(test)],
                             cwd=root, stdout=log, stderr=subprocess.STDOUT)
    sources.append(artifact(test))
    result = dict(scope=__doc__, observed_unix=time.time(), seconds=time.time()-started,
        pid=os.getpid(), pid_birth=psutil.Process().create_time(), python=sys.executable,
        sources=sources, cases=records, matched_queries=len(records), pytest_exit_code=run.returncode,
        resources=dict(rss_bytes=psutil.Process().memory_info().rss,
                       host_available_bytes=psutil.virtual_memory().available,
                       cuda_initialized=torch.cuda.is_initialized(),
                       distributed_initialized=torch.distributed.is_initialized()),
        operations=dict(model_initializations=0, model_forwards=0, finite_trace_calls=0,
                        backward_calls=0, optimizer_steps=0, scheduler_steps=0),
        role='Prepared-only semantic repair; frozen runtime and formal jobs untouched')
    assert not result['resources']['cuda_initialized'] and not result['resources']['distributed_initialized']
    (root / 'query-clock-repair.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('matched_queries', 'pytest_exit_code', 'seconds', 'resources', 'operations', 'role')}))
    return run.returncode


if __name__ == '__main__':
    raise SystemExit(main())
