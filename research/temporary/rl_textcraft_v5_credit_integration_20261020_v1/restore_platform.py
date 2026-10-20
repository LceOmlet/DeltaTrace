"""Reuse the original exact-hash restoration after the observed host reset."""
from prepare_campaign import ROOT

owner = ROOT/'receipts/textcraft-v5-eight-gpu-20261019-v1/source/restore_platform.py'
exec(compile(owner.read_text(), str(owner), 'exec'), globals())
