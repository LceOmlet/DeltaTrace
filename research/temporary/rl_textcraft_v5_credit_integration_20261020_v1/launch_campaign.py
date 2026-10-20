"""Reuse the occupancy-aware original v5 controller launcher."""
from prepare_campaign import ROOT

owner = ROOT/'receipts/textcraft-v5-eight-gpu-20261019-v1/source/launch_campaign.py'
code = owner.read_text().replace('receipts/textcraft-v5-eight-gpu-20261019-v1',
    'receipts/textcraft-v5-credit-integration-20261020-v1')
exec(compile(code, str(owner), 'exec'), globals())
