"""Refresh the six-run campaign figures, then the requested historical controls overlay."""
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
for entry in (HERE.parent/'rl_dtpo_symmetric_probability_ratio_postgroup_variance_20261018_v8/plot_both_tasks.py',
              HERE/'plot_comparison.py'):
    subprocess.run([sys.executable,'-X','utf8',str(entry)],env=env,check=True)
