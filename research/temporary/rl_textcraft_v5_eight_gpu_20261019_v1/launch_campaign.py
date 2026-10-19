"""Reuse the accepted physical-occupancy and no-duplicate controller launcher."""
import hashlib
from pathlib import Path
from prepare_campaign import ROOT,AUDIT

owner=ROOT/'receipts/six-run-fresh-validated-efficiency-20261018-v1/source/launch_campaign.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest()=='fb2013fc2d8c18b1eb58d18a05f9c2dd637b76dd140f6236451d9e5bc0310cff'
code=owner.read_text().replace("'receipts/six-run-fresh-validated-efficiency-20261018-v1'",
    "'receipts/textcraft-v5-eight-gpu-20261019-v1'")
code=code.replace("plan['all_six_start_from_base']", "plan['all_requested_runs_start_from_base']")
code=code.replace('submitted_fresh_six_run_native_queue','submitted_native_v5_eight_gpu_run')
exec(compile(code,str(owner),'exec'),globals())
