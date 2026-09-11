"""Import the verified VT/HotpotQA recovery comparison into manuscript fixtures."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'research/temporary/vt_budget_extension_20260911/selected_budget'
DATA = HERE / 'data'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    verified = json.loads((SOURCE / 'verification.json').read_bytes())
    assert verified['status'] == 'passed' and verified['score_rows'] == 3584
    for name, digest in verified['output_sha256'].items():
        assert sha(SOURCE / name) == digest, name
    for name, digest in verified['source_sha256'].items():
        assert sha(SOURCE / name) == digest, name
    policy = json.loads((SOURCE / 'protocol.json').read_bytes())
    assert policy['budgets'] == dict(vt_h2_c3=.1, vt_h4_c1=.1, vt_h6_c1=.2,
                                    vt_h10_c1=.3, hotpotqa_long=.1)
    with (SOURCE / 'per_case.csv').open(newline='', encoding='utf-8') as stream:
        cases = list(csv.DictReader(stream))
    with (SOURCE / 'primary.csv').open(newline='', encoding='utf-8') as stream:
        primary = {r['method']: r for r in csv.DictReader(stream)}
    rows = []
    for task, fraction in policy['budgets'].items():
        for method in policy['main_methods']:
            data = [r for r in cases if r['dataset'] == task and r['method'] == method]
            assert len(data) == (48 if task == 'hotpotqa_long' else 100)
            assert all(float(r['fraction']) == fraction for r in data)
            mean = statistics.fmean(float(r['recall']) for r in data)
            key = f'{task}_recall_at_{round(fraction * 100)}pct'
            assert mean == float(primary[method][key])
            rows.append(dict(dataset=task, method='FT' if method == 'FT_K3' else method,
                             attribution_variant=method, count=len(data), Recovery=mean,
                             RecoveryBudgetFraction=fraction,
                             RecoveryMetric='supporting_fact_recall' if task == 'hotpotqa_long' else 'body_token_recall',
                             RecoveryBudgetUnit=data[0]['budget_unit'],
                             RecoveryTargetMode=data[0]['target_mode'],
                             RecoveryProtocol='user-selected-vt-budgets-10-10-20-30-v1'))
    assert len(rows) == 35
    with (DATA / 'current_recovery.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fixtures = []
    for name in ('protocol.json', 'primary.csv', 'per_case.csv', 'verification.json'):
        target = DATA / 'current_recovery' / name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes((SOURCE / name).read_bytes())
        fixtures.append(dict(source=(SOURCE / name).relative_to(ROOT).as_posix(),
                             fixture=target.relative_to(HERE).as_posix(), sha256=sha(target)))
    manifest = dict(status='verified_current_recovery_import', selection_history=policy['selection_history'],
                    budgets=policy['budgets'], selected_inputs=448, source_score_rows=3584,
                    method_task_cells=35, new_model_calls=0, files=fixtures,
                    output_csv='data/current_recovery.csv', output_sha256=sha(DATA / 'current_recovery.csv'),
                    importer_sha256=sha(Path(__file__)))
    (DATA / 'current_recovery_sources.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    sources = json.loads((HERE / 'sources.json').read_bytes())
    sources['selection'] = ('All 13 task RISE/MAS results; released recovery for six NIAH tasks; '
                            'new verified VT body-token recovery at 10/10/20/30 percent and HotpotQA '
                            'supporting-fact recovery at 10 percent. Budgets selected retrospectively. '
                            'MATH/MoreHopQA recovery remains unavailable.')
    sources['sources']['current_recovery'] = dict(
        source=(SOURCE / 'primary.csv').relative_to(ROOT).as_posix(), source_sha256=sha(SOURCE / 'primary.csv'),
        fixture='data/current_recovery.csv', fixture_sha256=sha(DATA / 'current_recovery.csv'),
        provenance='data/current_recovery_sources.json', provenance_sha256=sha(DATA / 'current_recovery_sources.json'))
    (HERE / 'sources.json').write_text(json.dumps(sources, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=manifest['status'], added_recovery_cells=35, budgets=policy['budgets'])))


if __name__ == '__main__':
    main()
