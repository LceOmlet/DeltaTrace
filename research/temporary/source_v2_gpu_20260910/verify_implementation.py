"""Verify the implementation and legacy compatibility; does not run a model."""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OFFICIAL = ROOT / 'experiments/official'
sys.path.insert(0, str(OFFICIAL))
from summarize import summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication', type=Path, required=True)
    args = parser.parse_args()
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        'experiments/official', '-p', 'test_*.py', '-v'], cwd=ROOT, capture_output=True, text=True)
    (HERE / 'unit_tests.txt').write_text(tests.stdout + tests.stderr, encoding='utf-8')
    assert tests.returncode == 0, tests.stderr
    help_run = subprocess.run([sys.executable, 'experiments/official/evaluate.py', '--help'],
                              cwd=ROOT, capture_output=True, text=True)
    assert help_run.returncode == 0 and 'source-v2' in help_run.stdout
    code_files = [OFFICIAL / name for name in ['evaluate.py', 'summarize.py', 'evidence_protocol.py',
        'validate_source_protocol.py', 'test_evidence_protocol.py', 'recovery_diagnostics.py']]
    for path in code_files + [Path(__file__)]:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    manifest = json.loads((ROOT / 'deltatrace/clean/sources.json').read_bytes())
    method_files = {path: record for model in manifest['models'].values() for path, record in model['files'].items()}
    for path, record in method_files.items():
        assert digest(ROOT / path) == record['sha256'], path
    # Compare actual saved runs with the summary implementation before this fix.
    old_revision = subprocess.check_output(['git', 'rev-parse', '359677c'], cwd=ROOT, text=True).strip()
    old_source = subprocess.check_output(['git', 'show', f'{old_revision}:experiments/official/summarize.py'],
                                         cwd=ROOT, text=True, encoding='utf-8')
    old_module = {'__name__': 'legacy_summary', '__file__': str(OFFICIAL / 'summarize.py')}
    exec(compile(old_source, '<pinned legacy summary>', 'exec'), old_module)
    protocol = json.loads((OFFICIAL / 'protocol.json').read_bytes())
    legacy = []
    for task in protocol['tasks']:
        path = args.publication / 'raw' / (task + '.results.json.gz')
        report = json.loads(gzip.decompress(path.read_bytes()))
        assert 'evaluation_protocol' not in report
        before = old_module['summarize'](report, protocol)
        after = summarize(report, protocol)
        for key in ('evaluation_protocol', 'evaluation_protocol_sha256', 'sentence_recovery_enabled'):
            after.pop(key)
        assert after == before, task
        legacy.append({'dataset': task, 'cases': len(report['cases']), 'results_sha256': digest(path)})
    receipt = {'status': 'passed', 'python': platform.python_version(), 'model_calls': 0,
        'GPU_execution_completed': False, 'unit_test_log_sha256': digest(HERE / 'unit_tests.txt'),
        'CLI_help_passed': True, 'AST_checks_passed': True, 'frozen_method_files_verified': len(method_files),
        'legacy_summary_revision': old_revision, 'legacy_summaries_unchanged': legacy,
        'code_sha256': {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path)
            for path in code_files + [Path(__file__), OFFICIAL / 'source_protocol.json']}}
    (HERE / 'implementation_validation.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'passed', 'legacy_tasks': len(legacy),
        'legacy_cases': sum(row['cases'] for row in legacy), 'frozen_method_files': len(method_files)}, indent=2))


if __name__ == '__main__':
    main()
