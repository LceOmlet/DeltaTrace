"""Finish the recorded switch after the original cleanup wait has cleared.

Only read existing sessions and continue the already-preserved completion body.
No cleanup signal, checkpoint computation or training launch is repeated here.
"""
import ast
import json
import os
import psutil
import runpy
import time
from pathlib import Path
from prepare_campaign import ROOT, AUDIT, PARENT_AUDIT, VERSION, sha, put


def main():
    assert not (AUDIT / 'controller-job.json').exists()
    assert not (AUDIT / 'previous-v12-stop.json').exists()
    script = AUDIT / 'source/preserve_and_release.py'
    ns = runpy.run_path(str(script), run_name='preserved_user_switch_definitions_only')
    controller = json.loads((PARENT_AUDIT / 'controller-job.json').read_text())
    state = json.loads((PARENT_AUDIT / 'campaign-state.json').read_text())
    row, = state['jobs']
    preserved = json.loads((AUDIT / 'preserved-v12-checkpoint.json').read_text())
    cleanup = json.loads((AUDIT / 'original-session-cleanup.json').read_text())
    assert [(v['pid'], v['birth']) for v in cleanup] == [(controller['pid'], controller['birth']), (row['pid'], row['birth'])]
    members = []
    for p in psutil.process_iter(['pid', 'create_time', 'status', 'name']):
        try:
            if os.getsid(p.pid) in (controller['pid'], row['pid']) and p.status() != psutil.STATUS_ZOMBIE:
                members.append(p.info)
        except (psutil.Error, OSError):
            pass
    assert not members, 'Original sessions still occupied; do not repeat stop or launch'
    checkpoint = Path(preserved['checkpoint'])
    for name, size in preserved['required_files'].items():
        p = checkpoint / name
        assert p.is_file() and p.stat().st_size == size
    assert sha(checkpoint / 'data.pt') == preserved['data_SHA256']
    owner = ROOT / 'receipts/two-task-eight-gpu-sequential-20261017-v2/source/run_campaign.py'
    assert sha(owner) == 'f71ff4d85087c18d37bdb9fd2e40dbf7f36763e64b528e772612f24004e85441'
    definitions = [n for n in ast.parse(owner.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'verify_GPU_empty']
    namespace = dict(ns)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(owner), 'exec'), namespace)
    physical = namespace['verify_GPU_empty']()
    put(AUDIT / 'original-cleanup-timeout-resolved-v1.json', dict(status='original_cleanup_wait_cleared_on_read_only_followup', observed_unix=time.time(), original_results=cleanup, remaining_non_zombie=members, physical_GPU=physical, cleanup_signals_repeated=False, training_launched=False))
    for value in cleanup:
        value['original_result_retained'] = value.pop('result')
        value['result'] = dict(remaining_non_zombie=[], observed_after_original_cleanup_wait=True)
    function = next(n for n in ast.parse(script.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    begin = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'physical' for t in n.targets))
    ns.update(namespace=namespace, controller=controller, state=state, row=row, preserved=preserved, checkpoint=checkpoint, step=preserved['preserved_step'], completed=preserved['completed_iterations'], cleanup=cleanup, files=[checkpoint / name for name in preserved['required_files']])
    exec(compile(ast.Module(body=function.body[begin:], type_ignores=[]), str(script), 'exec'), ns)


if __name__ == '__main__':
    main()
