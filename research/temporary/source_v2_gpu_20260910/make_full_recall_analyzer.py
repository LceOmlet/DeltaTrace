"""Keep the verified per-case checks, extending only complete-data coverage."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
source=(HERE/'analyze_target_scope.py').read_text(encoding='utf-8')
imports=source[:source.index('def main():')]
start=source.index('    rows, diagnostics, residuals = [], [], []')
end=source.index("    assert stage != 'development'",start)
body=source[start:end].replace("    for row in r['cases']:","    for row in records:")
code=imports+'''def verify_records(records, vectors, caches, originals, tok):
'''+body+'''    return rows, diagnostics, residuals, identity_checks
'''
compile(code,'verify_full_recall_records.py','exec')
(HERE/'verify_full_recall_records.py').write_text(code,encoding='utf-8')
stats=(HERE/'scope_statistics.py').read_text(encoding='utf-8')
stats=stats.replace('def summarize_scope(rows, split):','def summarize_full_recall(rows, task_indices):')
stats=stats.replace("indices=split['tasks'][t]['validation']","indices=task_indices[t]")
stats=stats.replace('assert len(indices)==16',"assert len(indices)==(48 if t=='hotpotqa_long' else 100)")
stats=stats.replace('confirmed_scoped_advantages=','positive_adjusted_intervals=').replace('confirmed_scoped_disadvantages=','negative_adjusted_intervals=').replace('confirmed_same_view_shared_advantage=','same_view_positive_intervals=')
stats=stats.replace('Prespecified paired Recall comparisons; never selects a new candidate.',
    'Descriptive paired intervals for the complete benchmark, including development cases.')
compile(stats,'full_recall_statistics.py','exec')
(HERE/'full_recall_statistics.py').write_text(stats,encoding='utf-8')
print('Generated full-data case verifier and descriptive statistics')
