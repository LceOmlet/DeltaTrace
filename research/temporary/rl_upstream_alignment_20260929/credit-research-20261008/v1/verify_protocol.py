"""CPU checks of the real frozen corpus and existing owner-returned scores.

These check splits/aggregation only. Reading saved original owner scores is not
new attribution-quality evidence on the development or test collection.
"""
from pathlib import Path
import copy
import hashlib
import json
import random
import time

from freeze_manifest import freeze, EXPOSED_BATCHES
from score_summary import summarize, paired_differences

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def verify():
    corpus = json.loads((HERE/'corpus.json').read_bytes())
    manifest = json.loads((HERE/'manifest.json').read_bytes())
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    check('corpus SHA binds exact frozen identities',
          hashlib.sha256((HERE/'corpus.json').read_bytes()).hexdigest() == manifest['corpus']['sha256'])
    rebuilt = freeze(corpus)
    check('manifest is reproducible without credit scores',
          all(rebuilt[key] == manifest[key] for key in rebuilt))
    shuffled = copy.deepcopy(corpus)
    for data in shuffled['tasks'].values():
        random.Random(739).shuffle(data['records'])
    check('split and stage selection do not depend on input file/row order', freeze(shuffled) == rebuilt)
    for task, data in manifest['tasks'].items():
        records = corpus['tasks'][task]['records']
        check(task+' copied UIDs receive one identity', len({r['traj_uid'] for r in records}) == len(records))
        dev = {g['initial_state_sha256'] for g in data['groups'] if g['split'] == 'development'}
        test = {g['initial_state_sha256'] for g in data['groups'] if g['split'] == 'test'}
        check(task+' no initial state crosses split', not dev & test)
        for record in records:
            if any(v['file_sha256'] in EXPOSED_BATCHES for v in record['native_occurrences']):
                check(task+' exposed UID stays development: '+record['traj_uid'], record['initial_state_sha256'] in dev)
        check(task+' first stage includes every group', all(group['first_stage_uids'] for group in data['groups']))
    check('all eight AppWorld missing captures remain in corpus',
          sum(not row['native_occurrences'] for row in corpus['tasks']['appworld']['records']) == 8)
    check('only actual first-DT nonzero-reward bank claimed',
          all(row['reward'] > 0 for data in corpus['tasks'].values() for row in data['records']))

    score_files = [
        ('textcraft', '6e76f70e-bdeb-4726-8b35-d9e21eff2f68',
         REPO/'experiments/rl/results_textcraft_actual_author_curves_20261008.json'),
    ]
    rows = []
    for task, uid, path in score_files:
        original = json.loads(path.read_bytes())
        views = original['views']
        rows.append({'task': task, 'traj_uid': uid, 'metrics': {
            'rise': views['signed_RISE']['author_return']['rise'],
            'mas': views['positive_MAS']['author_return']['mas']}})
    summary = summarize(manifest, rows, 'development')
    check('one old selected curve is explicitly incomplete, not collection validation',
          summary['textcraft']['rise']['scored_finite'] == 1 and
          not summary['textcraft']['rise']['complete'] and
          len(summary['textcraft']['rise']['missing_uids']) == 84)
    check('missing AppWorld scores stay absent, never zero',
          summary['appworld']['rise']['available_trajectory_mean'] is None and
          summary['appworld']['mas']['available_equal_state_mean'] is None)
    try:
        summarize(manifest, rows+rows, 'development')
    except ValueError:
        checks.append('duplicate rank/copy score rejected')
    else:
        raise AssertionError('Duplicate score accepted')
    try:
        summarize(manifest, rows, 'test')
    except ValueError:
        checks.append('development score cannot enter frozen test summary')
    else:
        raise AssertionError('Split leakage accepted')
    paired = paired_differences(manifest, rows, rows, 'development')
    check('paired comparison uses identical actual UID and metric values',
          paired['textcraft']['rise']['available_trajectory_mean'] == 0 and
          not paired['textcraft']['rise']['complete'])
    return {'scope': __doc__, 'unix': time.time(), 'checks': checks,
            'passed': len(checks), 'failed': 0, 'model_calls': 0, 'DT_calls': 0,
            'optimizer_steps': 0, 'collection_quality_improvement_claimed': False}


if __name__ == '__main__':
    result = verify()
    (HERE/'cpu-verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result, indent=2))
