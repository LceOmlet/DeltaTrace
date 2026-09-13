"""Import the frozen 72-case symmetric-GDN comparison, preserving old fixtures."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'research/temporary/qwen35_niah_causal_20260913'
DATA = HERE / 'data'
SNAPSHOT = DATA / 'qwen35_official_sources'
SNAPSHOT.mkdir(exist_ok=True)
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: json.loads(p.read_bytes())
manifest = load(SOURCE / 'artifact_manifest.json')
frozen = {r['path']: r['sha256'] for r in manifest['files']}
ledger = []


def read_source(relative, *, snapshot=True):
    p = SOURCE / relative
    name = p.relative_to(ROOT).as_posix()
    assert digest(p) == frozen[name], name
    item = dict(repository_source=name, sha256=digest(p))
    if snapshot:
        fixture = SNAPSHOT / relative.replace('/', '__')
        fixture.write_bytes(p.read_bytes())
        item['fixture'] = fixture.relative_to(HERE).as_posix()
    ledger.append(item)
    return load(p)


validation = read_source('raw/validation_v1/validation_analysis.json')
generalization = read_source('raw/generalization_fresh_v1/generalization_analysis.json')
assert validation['status'] == generalization['status'] == 'complete'
assert validation['actual_cases'] == 60 and generalization['cases'] == 12
assert generalization['deployment_bitwise']
plans = [read_source('validation_protocol.json'), read_source('generalization_fresh_protocol.json')]
for relative in ('candidate_receipt.json', 'raw/validation_v1/independent_verification.json',
                 'raw/validation_v1/input_verification.json',
                 'raw/generalization_fresh_v1/independent_verification.json'):
    read_source(relative)
profile_path = ROOT / 'deltatrace/profiles/qwen35_gdn_symmetric.py'
assert digest(profile_path) == frozen[profile_path.relative_to(ROOT).as_posix()]

case_rows = list(validation['rows'])
status = read_source('raw/generalization_fresh_v1/status.json')
for item in status['cases']:
    path = f"research/temporary/qwen35_niah_causal_20260913/raw/generalization_fresh_v1/{item['path']}/results.json"
    frozen[path] = item['results_sha256']
for task, indices in plans[1]['selection'].items():
    for index in indices:
        case = read_source(f'raw/generalization_fresh_v1/{task}_{index}/results.json', snapshot=False)
        assert case['status'] == 'complete'
        for method in ('DT_original', 'DT_gdn_symmetric', 'FT_K1'):
            values = case['FT_K1'] if method == 'FT_K1' else case['methods'][method]
            case_rows.append(dict(dataset=task, index=index, method=method,
                                  **{m: values.get(m) for m in ('rise', 'mas', 'recall')}))
assert len(case_rows) == 276
means = [r for a in (validation, generalization) for r in a['means'] if r['dataset'] != 'macro']
selected = []
for r in means:
    rows = [c for c in case_rows if (c['dataset'], c['method']) == (r['dataset'], r['method'])]
    assert len(rows) == r['n']
    expected = next(p['selection'][r['dataset']] for p in plans if r['dataset'] in p['selection'])
    assert sorted(c['index'] for c in rows) == expected
    for metric in ('rise', 'mas', 'recall'):
        values = [c.get(metric) for c in rows]
        if r.get(metric) is None:
            assert all(v is None for v in values)
        else:
            assert all(v is not None and math.isfinite(v) for v in values)
            assert abs(statistics.mean(values) - r[metric]) < 1e-14
    if r['method'] == 'DT_original':
        continue
    selected.append(dict(dataset=r['dataset'], method='DT' if r['method']=='DT_gdn_symmetric' else r['method'],
                         count=r['n'], **{m:r.get(m) for m in ('rise','mas','recall')}))
assert len(selected) == 22


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


output = DATA / 'qwen35_official_quality.csv'
per_case = DATA / 'qwen35_official_cases.csv'
write_csv(output, selected)
write_csv(per_case, case_rows)
receipt = dict(version='qwen35-gdn-symmetric-v1-quality-72', status='verified',
               attribution_profile='gdn-symmetric-v1', attribution_source_sha256=digest(profile_path),
               promoted_by_user='2026-09-13', tasks=8, cases=72,
               scope='60 novel key/value assignments on fixed NIAH templates; 6 fixed original MATH and 6 MoreHopQA cases. Descriptive validation, not a full benchmark.',
               selection={k:v for p in plans for k,v in p['selection'].items()},
               metrics={'rise':'signed DT ranking; FT K1', 'mas':'positive DT ranking; FT K1',
                        'recall':'NIAH positive token ranking, 10% eligible-token budget; FT K3'},
               output_csv=output.relative_to(HERE).as_posix(), output_sha256=digest(output),
               case_csv=per_case.relative_to(HERE).as_posix(), case_csv_sha256=digest(per_case),
               historical_full_benchmark='data/qwen35_full_quality.csv (clean-v1; excluded from current table)',
               files=ledger)
(DATA/'qwen35_official_quality_sources.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:receipt[k] for k in ('status','tasks','cases','output_sha256')}))
