"""One native source collection; common, full-prefix and fixed-200 figures."""
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
runpy.run_path(str(HERE/'plot_algorithms.py'), run_name='__main__')
(HERE/'latest-output.txt').write_text((HERE/'latest-algorithm-output.txt').read_text(encoding='utf-8'), encoding='utf-8')
runpy.run_path(str(HERE/'plot_window_200.py'), run_name='__main__')
