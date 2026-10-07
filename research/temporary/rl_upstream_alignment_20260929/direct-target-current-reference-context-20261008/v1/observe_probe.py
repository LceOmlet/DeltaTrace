"""Reuse the existing read-only PID/birth/physical-memory curve observer."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
original=HERE.parents[1]/'direct-target-action-author-curve-20261007/v1/poll_curve.py'
code=original.read_text()
old='receipts/direct-target-action-author-curve-20261007-v1'
assert code.count(old)==1
code=code.replace(old,'receipts/direct-target-current-reference-context-20261008-v1')
code=code.replace("b['views'].items()","b.get('views',{}).items()")
exec(compile(code,str(original),'exec'),dict(__file__=str(HERE/'observe_probe.py'),__name__='__main__'))
