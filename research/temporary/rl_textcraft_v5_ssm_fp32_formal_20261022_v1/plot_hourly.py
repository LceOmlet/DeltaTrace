"""Default plot: current v5 and frozen original PPO/GRPO only."""
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
runpy.run_path(str(HERE/'plot_algorithms.py'), run_name='__main__')
(HERE/'latest-output.txt').write_text((HERE/'latest-algorithm-output.txt').read_text(encoding='utf-8'), encoding='utf-8')
