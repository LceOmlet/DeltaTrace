"""Reuse original SHA-verified completed-artifact transport for the block test."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
original = HERE.parents[1]/'direct-target-action-author-curve-20261007/v1/fetch_completed.py'
code = original.read_text()
old = 'receipts/direct-target-action-author-curve-20261007-v1'
assert code.count(old)==1
code = code.replace(old,'receipts/current-extreme-official-block-dtype-20261008-v1')
code = code.replace("destination = HERE / 'actual-results'", "destination = HERE / 'block-dtype-results'")
code = code.replace("assert (root/'results/completed.json').is_file()",
                    "assert json.loads((root/'phase.json').read_bytes())['phase']=='complete'")
exec(compile(code,str(original),'exec'),dict(__file__=str(HERE/'fetch_current_block_dtype.py'),__name__='__main__'))
