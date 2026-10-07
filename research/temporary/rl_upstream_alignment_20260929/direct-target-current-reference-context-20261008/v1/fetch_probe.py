"""Reuse original SHA-verified diagnostic artifact transport for current case."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
original=HERE.parents[1]/'direct-target-action-author-curve-20261007/v1/fetch_completed.py'
code=original.read_text()
old='receipts/direct-target-action-author-curve-20261007-v1'
assert code.count(old)==1
code=code.replace(old,'receipts/direct-target-current-reference-context-20261008-v1')
exec(compile(code,str(original),'exec'),dict(__file__=str(HERE/'fetch_probe.py'),__name__='__main__'))
