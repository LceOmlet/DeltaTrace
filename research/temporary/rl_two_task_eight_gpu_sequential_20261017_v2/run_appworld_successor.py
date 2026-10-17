"""Wait for the existing TextCraft segment; reuse its native process owner.

This adds no training, sampling, optimization, retry or checkpoint algorithm.
It does not alter or restart the predecessor, or skip a failed combination.
"""
import json
import os
from pathlib import Path
import time

import psutil
import run_campaign as owner

ROOT = owner.ROOT
AUDIT = ROOT / 'receipts/two-task-eight-gpu-sequential-20261017-v2'
APP = ROOT / 'receipts/appworld-eight-gpu-sequential-20261017-v1'


def confirm_predecessor():
    plan = json.loads((AUDIT / 'six-combination-manifest.json').read_bytes())
    predecessor = plan['segments'][0]
    state_path = Path(predecessor['state'])
    while True:
        state = json.loads(state_path.read_bytes())
        assert (state['controller_pid'], state['controller_birth']) == (
            predecessor['controller_pid'], predecessor['controller_birth'])
        handoff = dict(collected=time.time(), controller_pid=os.getpid(),
            controller_birth=psutil.Process().create_time(),
            predecessor_state=state['status'], predecessor_state_path=str(state_path),
            predecessor_current_method=state.get('current_method'),
            predecessor_completed=[dict(method=j['method'], completed=j.get('completed_iterations'))
                                   for j in state['jobs']],
            status='waiting_for_TextCraft_three_native_completions',
            model_calls=0, training_processes_started=0)
        owner.write(AUDIT / 'handoff-state.json', handoff)
        if state['status'] == 'all_three_native_200_iteration_runs_completed':
            assert state['order'] == ['dt', 'ppo', 'grpo'] and len(state['jobs']) == 3
            for row in state['jobs']:
                assert row['status'] == 'completed_200_native_iterations'
                assert row['completed_iterations'] == 200
                assert not row['native_finished_session_cleanup']['remaining_non_zombie']
            owner.verify_GPU_empty()
            handoff.update(status='predecessor_complete_handoff_to_AppWorld', collected=time.time())
            owner.write(AUDIT / 'handoff-state.json', handoff)
            return
        if state['status'] == 'needs_attention_no_failed_combination_skipped':
            raise RuntimeError('Predecessor native combination failed; do not skip it or start AppWorld')
        process = psutil.Process(predecessor['controller_pid'])
        assert process.create_time() == predecessor['controller_birth']
        assert process.status() != psutil.STATUS_ZOMBIE
        time.sleep(60)


if __name__ == '__main__':
    try:
        owner.main(audit=APP, version='appworld-eight-gpu-sequential-20261017-v1',
                   readiness=confirm_predecessor)
    except BaseException as exc:
        owner.write(AUDIT / 'handoff-error.json', dict(observed=time.time(),
            error_type=type(exc).__name__, error=str(exc),
            status='needs_attention_no_failed_combination_skipped'))
        raise
