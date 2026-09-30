"""CPU verification entry for the current original AgentGym rollout integration.

The retired one-step manager fixture is replaced by the actual native-loop
checks, including the author's reset/step failure behavior.
"""
from pathlib import Path
import os
import pytest

if __name__ == '__main__':
    entry = Path(__file__).resolve().parent
    os.environ.setdefault('TEXTCRAFT_ORIGINAL_ROLLOUT', str(entry/'textcraft-original-rollout.py'))
    os.environ.setdefault('TEXTCRAFT_BASELINE_SCHEMA', str(entry/'textcraft-schema-before-artifacts.py'))
    raise SystemExit(pytest.main(['-q', '--tb=short', str(entry/'test_textcraft_trajectory_entry.py'),
        '--junitxml='+str(entry/'textcraft-original-loop-tests.xml')]))
