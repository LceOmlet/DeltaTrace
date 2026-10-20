"""Reuse the provisioned author service; retain separate new-run receipts."""
from prepare_campaign import ROOT

owner = ROOT/'receipts/textcraft-v5-eight-gpu-20261019-v1/source/ensure_service.py'
code = owner.read_text().replace('receipts/textcraft-v5-eight-gpu-20261019-v1',
    'receipts/textcraft-v5-credit-integration-20261020-v1')
exec(compile(code, str(owner), 'exec'), globals())
