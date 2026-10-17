"""Switch only fresh campaign identities around the same accepted plot entry."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys


if __name__ == '__main__':
    if os.environ.get('RL_FRESH_PLOT_CHILD') != '1':
        for task in ('TextCraft', 'AppWorld'):
            env = dict(os.environ, RL_CAMPAIGN_PLOT_TASK=task, RL_FRESH_PLOT_CHILD='1', CUDA_VISIBLE_DEVICES='')
            subprocess.run([sys.executable, '-X', 'utf8', __file__], env=env, check=True)
    else:
        source = Path(__file__).resolve().parent.parent / 'rl_textcraft_eight_gpu_sequential_20261017_v1/plot_campaign.py'
        assert hashlib.sha256(source.read_bytes()).hexdigest() == 'aac522f9a74973bd1842078d182a1951e509b7e590f1ee99e27ee72e9b8cf5e1'
        code = source.read_text()
        for task in ('textcraft', 'appworld'):
            code = code.replace(task+'-eight-gpu-sequential-20261017-v1', task+'-eight-gpu-sequential-20261017-v2')
        exec(compile(code, str(source), 'exec'), globals())
