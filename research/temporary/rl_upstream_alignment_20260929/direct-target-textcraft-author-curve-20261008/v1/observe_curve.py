"""Call the existing read-only curve observer for this exact task receipt."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
original=HERE.parents[1]/'direct-target-action-author-curve-20261007/v1/poll_curve.py'
code=original.read_text()
old='receipts/direct-target-action-author-curve-20261007-v1'
assert code.count(old)==1
code=code.replace(old,'receipts/direct-target-textcraft-author-curve-20261008-v1')
exec(compile(code,str(original),'exec'),dict(__file__=str(HERE/'observe_curve.py'),__name__='__main__'))
