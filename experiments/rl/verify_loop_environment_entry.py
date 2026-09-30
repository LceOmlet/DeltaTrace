"""Run the current CPU checks of original LOOP environment and worker methods."""
import os
from pathlib import Path
import site
import pytest

if __name__ == '__main__':
    if os.environ.get('LOOP_EXTRAS'):
        site.addsitedir(os.environ['LOOP_EXTRAS'])
    root = Path(__file__).resolve().parent
    os.environ.setdefault('LOOP_ROLLOUT_BASELINE', str(root/'loop-original-rollout-worker.py'))
    raise SystemExit(pytest.main(['-q', str(root/'test_loop_owner_worker.py'),
        '--junitxml='+str(root/'loop-native-worker-tests.xml')]))
