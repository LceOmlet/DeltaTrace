"""Compose the same pinned process owner twice; no training loop is replaced."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT / 'receipts/six-run-fresh-after-efficiency-20261017-v1'


def main():
    old_source = ROOT / 'receipts/textcraft-eight-gpu-sequential-20261017-v1/source'
    owner = old_source / 'run_campaign.py'
    assert hashlib.sha256(owner.read_bytes()).hexdigest() == '86d65a671e1c7167b025388ffcd07e1dbdbbf0291eba62c68ed085cb9313ff6f'
    sys.path.insert(0, str(old_source))
    native = runpy.run_path(str(owner), run_name='pinned_six_run_process_owner')
    plan = json.loads((AUDIT / 'six-combination-manifest.json').read_bytes())
    assert plan['all_six_start_from_base'] and len(plan['combinations']) == 6
    deployed = json.loads((AUDIT / 'native-resource-owner-deployed.json').read_bytes())
    assert deployed['status'] == 'deployed_after_pilot_stopped'
    def ready():
        stopped = json.loads((AUDIT / 'user-replaced-tuning-jobs-stop.json').read_bytes())
        assert stopped['status'] == 'stopped_by_user_replacement' and not stopped['remaining_non_zombie']
    for segment in plan['segments']:
        audit = Path(segment['state']).parent
        native['main'](audit=audit, version=audit.name, readiness=ready)
    plan.update(status='all_six_native_200_iteration_runs_completed', finished=time.time())
    native['write'](AUDIT / 'six-combination-manifest.json', plan)


if __name__ == '__main__':
    main()
