"""Reuse the original all-rank actual-import and effective-mode metadata RPC."""
from prepare_campaign import ROOT

owner = ROOT/'receipts/textcraft-v5-eight-gpu-20261019-v1/source/inspect_native_workers.py'
code = owner.read_text().replace('textcraft-v5-eight-gpu-20261019-v1',
    'textcraft-v5-credit-integration-20261020-v1')
exec(compile(code, str(owner), 'exec'), globals())
