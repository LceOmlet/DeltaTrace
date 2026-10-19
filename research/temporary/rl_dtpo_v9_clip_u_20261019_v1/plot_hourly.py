"""Save both formal task plots, then the requested TextCraft archived controls."""
from pathlib import Path
import runpy

HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    runpy.run_path(str(HERE/'plot_both_tasks.py'),run_name='__main__')
    runpy.run_path(str(HERE/'plot_historical_controls.py'),run_name='__main__')
