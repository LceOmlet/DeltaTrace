"""Run the existing v5 method and native transport checks unchanged."""
from prepare_campaign import ROOT

owner = ROOT/'receipts/textcraft-v5-eight-gpu-20261019-v1/source/run_cpu_checks.py'
exec(compile(owner.read_text(), str(owner), 'exec'), globals())
