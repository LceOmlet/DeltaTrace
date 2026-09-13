"""Read-only audit of the live Qwen3.5 experiment against the fixed manuscript."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from collections import Counter

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_bytes())

def audit(repo, snapshot):
    live = snapshot / 'repo_dynamic'
    source = live / 'experiments/qwen35_comparison'
    protocol = read(source / 'paper_recovery/protocol.json')
    fixture = read(source / 'paper_recovery/paper_inputs.json')
    budgets = read(repo / 'research/temporary/vt_budget_extension_20260911/selected_budget/protocol.json')
    expected = {t:r['count'] for t,r in read(repo / 'experiments/official/protocol.json')['tasks'].items()}
    current = read(source / 'protocol.json')
    assert {t:r['count'] for t,r in current['tasks'].items()} == expected
    assert sum(expected.values()) == 1243
    assert protocol['paper_selected_protocol_sha256'] == sha(repo / 'research/temporary/vt_budget_extension_20260911/selected_budget/protocol.json')
    assert {t:r['fraction'] for t,r in protocol['tasks'].items()} == budgets['budgets']
    assert sha(source / 'paper_recovery/paper_inputs.json') == protocol['paper_inputs_sha256']
    for relative, digest in fixture['source_files'].items():
        assert sha(repo / relative) == digest, relative
    for relative, digest in protocol['scoring_owner_sha256'].items():
        assert sha(repo / relative) == digest == sha(live / relative), relative
    original = {(r['dataset'], r['index']):r for r in read(repo / 'research/temporary/all_baselines_20260910/inputs.json')['cases']}
    native = {(r['dataset'], r['index']):r for r in read(repo / 'research/temporary/all_baselines_20260910/candidates.json')['cases']}
    labels = {(r['dataset'], r['index']):r for r in read(repo / 'research/temporary/all_baselines_20260910/labels.json')['cases']}
    keys = set()
    for row in fixture['cases']:
        key = row['dataset'], row['index']
        assert key not in keys
        keys.add(key)
        old = original[key]
        for field in ('prompt','target','target_mode','original_target_sha256'):
            assert row[field] == old[field], (key, field)
        assert row['target_mode'] == ('full' if key[0] == 'hotpotqa_long' else 'answer_only')
        if key[0] == 'hotpotqa_long':
            assert row['units'] == native[key]['units']
            assert row['official_keys'] == labels[key]['official_keys']
    assert keys == original.keys() and len(keys) == 448
    # Identify runtime-only changes, preserving the exact arithmetic functions.
    before = repo / 'deltatrace/clean/qwen35'
    after = live / 'deltatrace/clean/qwen35'
    differences = []
    allowed = {
      'finite_fla_gpu.py': {'make_compiled_finite_pullback', 'pullback'},
      'qwen35_answer_finite.py': {'__init__','__call__'},
      'qwen35_clean_runner.py': {'make_qwen35_clean_runner'},
      'qwen35_decoder_finite.py': {'__init__'},
      'qwen35_dense_finite_runner.py': {'__init__'},
    }
    for p in sorted(before.glob('*.py')):
        q = after / p.name
        assert q.is_file()
        a, b = p.read_text(encoding='utf-8'), q.read_text(encoding='utf-8')
        if a == b:
            continue
        functions = lambda s: {n.name: ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(s)) if isinstance(n, (ast.FunctionDef,ast.AsyncFunctionDef))}
        left, right = functions(a), functions(b)
        changed = {k for k in left if left[k] != right.get(k)}
        assert changed <= allowed.get(p.name,set()), (p.name, changed)
        differences.append(dict(file=p.name, changed_existing_functions=sorted(changed), paper_sha256=sha(p), current_sha256=sha(q)))
    return dict(status='passed_static_protocol_and_fixture_audit', full_tasks=13, full_cases=1243,
        paper_recovery_cases=448, vt_cases=400, hotpot_cases=48,
        paper_source_files=fixture['source_files'], scoring_owners=protocol['scoring_owner_sha256'],
        fixture_sha256=protocol['paper_inputs_sha256'], budgets=budgets['budgets'],
        snapshot_protocol_sha256=sha(source/'protocol.json'),
        recovery_protocol_sha256=sha(source/'paper_recovery/protocol.json'),
        method_runtime_differences=differences,
        limitations=[
            'This static audit does not certify recovery model execution; inspect saved actual inputs and target weights after all 448 calls complete.',
            'Dynamic clean-v2 runtime differs from the paper clean-v1 snapshot; core propagation arithmetic is unchanged in the inspected Python sources, but bitwise equivalence is not asserted.',
            'Full-benchmark one-pass timings including compilation are not the paper repeated warm development efficiency experiment.',
            'Legacy full-response VT and HotpotQA token Recall must be replaced by the separate paper-matched recovery outputs.',
        ])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report=audit(a.repo,a.snapshot)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('paper_source_files','scoring_owners','method_runtime_differences')},ensure_ascii=False,indent=2))
if __name__=='__main__':
    main()

