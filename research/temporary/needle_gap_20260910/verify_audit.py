"""Verify source receipts, paired aggregates, and invariants of the saved audit."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_csv(name):
    with (HERE / name).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def verify_aggregate(cases, tasks, fields):
    for r in tasks:
        group = [c for c in cases if c['dataset'] == r['dataset'] and (
            r.get('subset', 'all') == 'all' or int(c['index']) % 2 == (r['subset'] == 'odd'))]
        assert len(group) == int(r['count'])
        for field in fields:
            assert abs(np.mean([float(c[field]) for c in group]) - float(r[field])) < 1e-12, (r['dataset'], field)


def main():
    receipt = json.loads((HERE / 'source_receipt.json').read_text())
    assert receipt['script_sha256'] == sha(HERE / 'audit_saved.py')
    assert receipt['diagnostic_module_sha256'] == sha(ROOT / 'experiments/official/recovery_diagnostics.py')
    assert receipt['cases'] == 1048 and receipt['tasks'] == 11 and receipt['model_calls'] == 0
    structure_receipt = json.loads((HERE / 'structure_receipt.json').read_text())
    assert structure_receipt['cases'] == structure_receipt['gold_offset_checks'] == 1048
    for name, digest in structure_receipt['sha256'].items():
        assert sha(ROOT / name) == digest, name
    ni_receipt = json.loads((HERE / 'niah_source_receipt.json').read_text())
    assert ni_receipt['script_sha256'] == sha(HERE / 'probe_niah_source.py')
    assert ni_receipt['cases'] == 600 and ni_receipt['model_calls'] == 0
    assert ni_receipt['tokenizer_sha256'] == structure_receipt['tokenizer_sha256']
    cases, tasks = read_csv('case_diagnostics.csv'), read_csv('task_diagnostics.csv')
    structure_cases, structure_tasks = read_csv('structure_cases.csv'), read_csv('structure_tasks.csv')
    ni_cases, ni_tasks = read_csv('niah_source_cases.csv'), read_csv('niah_source_tasks.csv')
    assert len(cases) == len(structure_cases) == 1048 and len(ni_cases) == 600
    assert len({(r['dataset'], r['index']) for r in cases}) == 1048
    verify_aggregate(cases, tasks, ['DT', 'FT', 'ceiling', 'DT_adjusted', 'absolute_recall'])
    verify_aggregate(structure_cases, structure_tasks, ['DT', 'FT', 'DT_scope', 'FT_scope', 'DT_scope_sentence', 'FT_scope_sentence', 'DT_sentence_positive'])
    verify_aggregate(ni_cases, ni_tasks, ['DT', 'FT', 'DT_source', 'FT_source'])
    index = {(r['dataset'], r['index']): r for r in cases}
    for row in structure_cases + ni_cases:
        original = index[row['dataset'], row['index']]
        for field in ['DT', 'FT', 'ceiling']:
            assert abs(float(row[field]) - float(original[field])) < 1e-12
        for field in ['DT_scope', 'FT_scope', 'DT_scope_sentence', 'FT_scope_sentence', 'DT_sentence_positive', 'DT_source', 'FT_source']:
            if field in row:
                assert 0 <= float(row[field]) <= float(row['ceiling']) + 1e-12
    assert all(float(r['DT_scope']) > float(r['DT']) for r in structure_cases if r['dataset'].startswith('vt_'))
    assert all(float(r['DT_source']) >= float(r['DT']) for r in ni_cases)
    sources = json.loads((ROOT / 'deltatrace/clean/sources.json').read_text())
    preserved = 0
    for family, model in sources['models'].items():
        for name, source in model['files'].items():
            assert sha(ROOT / name) == source['sha256'], name
            preserved += 1
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'experiments/official', '-p', 'test_recovery_diagnostics.py', '-v'],
                          cwd=ROOT, capture_output=True, text=True)
    assert test.returncode == 0, test.stdout + test.stderr
    (HERE / 'test_results.txt').write_text(test.stdout + test.stderr, encoding='utf-8')
    result = {'status': 'passed', 'saved_cases': len(cases), 'recovery_tasks': len(tasks),
              'source_restricted_NIAH_cases': len(ni_cases), 'clean_source_files_unchanged': preserved,
              'tests': 9, 'model_calls': 0, 'new_GPU_run_performed': False,
              'python': sys.version,
              'artifact_sha256': {p.name: sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ['.csv', '.json', '.png', '.svg', '.md', '.py'] and p.name != 'validation.json'},
              'implementation_sha256': {n: sha(ROOT / 'experiments/official' / n) for n in ['evaluate.py', 'recovery_diagnostics.py', 'retrieval_views.py', 'test_recovery_diagnostics.py']}}
    (HERE / 'validation.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ['artifact_sha256', 'implementation_sha256']}, indent=2))


if __name__ == '__main__':
    main()
