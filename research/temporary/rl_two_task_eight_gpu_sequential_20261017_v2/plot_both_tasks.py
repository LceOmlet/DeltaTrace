"""Produce separate three-algorithm figures using the existing plot entry."""
import os
from pathlib import Path
import subprocess
import sys

entry = Path(__file__).resolve().parent.parent / 'rl_textcraft_eight_gpu_sequential_20261017_v1/plot_campaign.py'

if __name__ == '__main__':
    for task in ('TextCraft', 'AppWorld'):
        env = dict(os.environ, RL_CAMPAIGN_PLOT_TASK=task, CUDA_VISIBLE_DEVICES='')
        subprocess.run([sys.executable, '-X', 'utf8', str(entry)], env=env, check=True)
