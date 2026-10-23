"""Preserve a completed native checkpoint, then call the original session owner.

This one-shot user switch contains no training, checkpoint-save or retry loop.
"""
import ast
import json
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import time
import psutil
from prepare_campaign import ROOT, AUDIT, PARENT_AUDIT, VERSION, sha, put


def main():
    assert not (AUDIT / 'controller-job.json').exists()
    assert not (AUDIT / 'previous-v5-stop.json').exists()
    controller = json.loads((PARENT_AUDIT / 'controller-job.json').read_text())
    state = json.loads((PARENT_AUDIT / 'campaign-state.json').read_text())
    row, = state['jobs']
    assert row['version'] == 'textcraft-dtpo-v5-ssm-fp32-8gpu-formal-20261022-v1'
    for item in (controller, row):
        assert psutil.Process(item['pid']).create_time() == item['birth']
    marker = Path(row['checkpoints']) / 'latest_checkpointed_iteration.txt'
    step = int(marker.read_text().strip())
    assert step >= 200, 'Wait for the existing native checkpoint; do not call save or stop early'
    reader = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'
    assert sha(reader) == 'c20559e38e651284859837735b8e7d15fe7ca59b090c62b7bc651228f6ba3607'
    log_owner = runpy.run_path(str(reader), run_name='original_native_user_switch_reader')
    snapshot = log_owner['collect'](str(PARENT_AUDIT), [row['output']])
    current, = snapshot['jobs']
    completed = log_owner['completed'](current)
    assert completed >= step and {m['step'] for m in current['metrics']} >= set(range(1, step + 1))
    checkpoint = marker.parent / f'global_step_{step}'
    files = [checkpoint / 'data.pt'] + [checkpoint / 'actor' / f'{kind}_world_size_8_rank_{rank}.pt' for rank in range(8) for kind in ('model', 'optim', 'extra_state')]
    assert all(p.is_file() and p.stat().st_size for p in files)
    put(AUDIT / 'preserved-v5-source-snapshot.json', snapshot)
    preserved = dict(status='complete_native_eight_rank_checkpoint_preserved', observed_unix=time.time(), run=row['run'], completed_iterations=completed, preserved_step=step, checkpoint=str(checkpoint), required_files={str(p.relative_to(checkpoint)):p.stat().st_size for p in files}, marker_SHA256=sha(marker), data_SHA256=sha(checkpoint / 'data.pt'), source_snapshot=str(AUDIT / 'preserved-v5-source-snapshot.json'), natural_exit0=False, reason='latest_human_switch_to_v11_empty_target_GRPO_only')
    put(AUDIT / 'preserved-v5-checkpoint.json', preserved)
    owner = ROOT / 'receipts/two-task-eight-gpu-sequential-20261017-v2/source/run_campaign.py'
    assert sha(owner) == 'f71ff4d85087c18d37bdb9fd2e40dbf7f36763e64b528e772612f24004e85441'
    definitions = [n for n in ast.parse(owner.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in ('release_finished_session', 'verify_GPU_empty')]
    namespace = dict(AUDIT=AUDIT, ROOT=ROOT, sha=sha, ast=ast, psutil=psutil, signal=signal, time=time, os=os, subprocess=subprocess, re=re)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(owner), 'exec'), namespace)
    cleanup = []
    for item in (controller, row):
        result = namespace['release_finished_session'](item['pid'], item['birth'])
        cleanup.append(dict(pid=item['pid'], birth=item['birth'], result=result))
        put(AUDIT / 'original-session-cleanup.json', cleanup)
        assert not result['remaining_non_zombie'], 'Inspect original cleanup result; do not launch while occupied'
    physical = namespace['verify_GPU_empty']()
    (AUDIT / 'physical-after-v5-release.txt').write_text(physical)
    stopped = dict(status='stopped_by_latest_human_switch_to_v11_empty_target_GRPO_only', stopped_unix=time.time(), preserved=preserved, cleanup=cleanup, natural_exit0=False, training_numerical_failure=False)
    put(AUDIT / 'previous-v5-stop.json', stopped)
    state.update(status=stopped['status'], finished_unix=time.time(), requested_next_run=VERSION)
    row.update(status=stopped['status'], active=False, alive=False, final_native_checkpoint=str(checkpoint), checkpoint_preserved_step=step, completed_iterations=completed, phase='stopped_after_native_checkpoint_for_user_switch', stopped_unix=time.time())
    put(PARENT_AUDIT / 'campaign-state.json', state)
    put(Path(row['run']) / 'source.json', row)
    for base in (PARENT_AUDIT, ROOT):
        for name in ('formal-training.json', 'active-training.json', 'active-source.json'):
            path = base / name
            if not path.exists():
                continue
            manifest = json.loads(path.read_text())
            matching = [r for r in manifest.get('jobs', []) if r.get('version') == row['version'] and r.get('pid') == row['pid']]
            for value in matching:
                value.update(row)
            if matching:
                put(path, manifest)
    prepared = json.loads((AUDIT / 'dt-prepared.json').read_text())
    prepared['prior_execution_preserved'] = preserved
    put(AUDIT / 'dt-prepared.json', prepared)
    campaign = json.loads((AUDIT / 'campaign-prepared.json').read_text())
    campaign['jobs'] = [prepared]
    put(AUDIT / 'campaign-prepared.json', campaign)
    print(json.dumps(dict(status=stopped['status'], preserved_step=step, required_files=len(files), all_eight_physical_GPUs_empty=True)))


if __name__ == '__main__':
    main()
