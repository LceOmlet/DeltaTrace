"""Compact status or a complete export for this experiment's own remote directory."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

BASE = Path('/tmp/codex_source_v2_gpu_20260910_v1')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('stage', choices=['smoke_v1', 'full_v1', 'full_v2', 'focus_v1'])
parser.add_argument('--export', action='store_true')
parser.add_argument('--export-completed', action='store_true')
args = parser.parse_args()
folder = BASE / args.stage
state = json.loads((folder / 'suite_status.json').read_bytes())
print(json.dumps({'status': state['status'], 'completed_tasks': len(state['completed_tasks']),
                  'current': state['current_task']}))
if state['current_task']:
    report_path = folder / state['current_task'] / 'results.json'
    if report_path.exists():
        report = json.loads(report_path.read_bytes())
        print(json.dumps({'case_status': report['status'],
            'completed_cases_in_task': sum(r['status'] == 'complete' for r in report['cases']),
            'entered_cases': len(report['cases']), 'error': report.get('error')}))
if args.export or args.export_completed:
    assert state['status'] == 'complete' or args.export_completed
    assert state['completed_tasks']
    suffix = '_export.zip' if args.export else f'_completed_{len(state["completed_tasks"])}_export.zip'
    archive = BASE / (args.stage + suffix)
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        output.writestr('suite_status.json', json.dumps(state, indent=2) + '\n')
        output.write(folder / 'plan_identity.json', 'plan_identity.json')
        paths = [path for row in state['completed_tasks'] for path in (folder / row['dataset']).rglob('*')]
        for path in sorted(paths):
            if path.is_file() and path.suffix in ('.json', '.npz', '.log'):
                output.write(path, path.relative_to(folder).as_posix())
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                      'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}))
