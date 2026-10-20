"""Reuse the original one-run v5 process owner after native eight-rank checks."""
from pathlib import Path
import json
from prepare_campaign import ROOT, AUDIT

owner = ROOT/'receipts/textcraft-v5-eight-gpu-20261019-v1/source/run_campaign.py'
code = owner.read_text()
code = code.replace("    for path,digest in row['source_bindings'].items():",
    "    native=json.loads((AUDIT/'native-behavior-status.json').read_bytes())\n"
    "    assert native['status']=='completed'\n"
    "    assert native['physical_gpus']==list(range(8))\n"
    "    for rank in range(8):\n"
    "        checked=json.loads((AUDIT/f'native-multi-target-rank{rank}.json').read_bytes())\n"
    "        assert checked['status']=='completed' and checked['rank']==rank\n"
    "        assert checked['credit_calls'] and all(x['exact_expression_equality'] for x in checked['credit_calls'])\n"
    "    for path,digest in row['source_bindings'].items():")
exec(compile(code, str(owner), 'exec'), globals())
