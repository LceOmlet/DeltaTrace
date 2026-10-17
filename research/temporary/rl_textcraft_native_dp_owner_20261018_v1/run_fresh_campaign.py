"""Compose the pinned six-run process owner; only accepted native placement starts."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT = ROOT/'receipts/six-run-fresh-native-dp-owner-20261018-v1'
DP_AUDIT = ROOT/'receipts/textcraft-native-dp-owner-20261018-v1'


def main():
    original = ROOT/'receipts/two-task-eight-gpu-sequential-20261017-v2/source'
    owner = original/'run_campaign.py'
    assert hashlib.sha256(owner.read_bytes()).hexdigest() == 'f71ff4d85087c18d37bdb9fd2e40dbf7f36763e64b528e772612f24004e85441'
    sys.path.insert(0, str(original))
    native = runpy.run_path(str(owner), run_name='pinned_six_run_process_owner')
    plan = json.loads((AUDIT/'six-combination-manifest.json').read_bytes())
    assert plan['all_six_start_from_base'] and len(plan['combinations']) == 6

    def ready():
        stopped = json.loads((DP_AUDIT/'tuning-stopped-for-native-benchmark.json').read_bytes())
        assert stopped['status'] == 'stopped_by_user_replacement' and not stopped['remaining_non_zombie']
        accepted = json.loads(Path(plan['native_placement_acceptance']).read_bytes())
        assert accepted['status'] == 'accepted_native_DP_episode_placement'
        for path, expected in accepted['candidate_source_SHA256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected

    for segment in plan['segments']:
        audit = Path(segment['state']).parent
        native['main'](audit=audit, version=audit.name, readiness=ready)
    plan.update(status='all_six_native_200_iteration_runs_completed', finished=time.time())
    native['write'](AUDIT/'six-combination-manifest.json', plan)


if __name__ == '__main__':
    main()
