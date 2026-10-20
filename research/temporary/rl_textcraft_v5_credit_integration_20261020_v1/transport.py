"""Reuse the recorded strict transport and provisioned interpreter."""
import importlib.util
from pathlib import Path

_owner = Path(__file__).resolve().parents[1] / 'rl_textcraft_v5_eight_gpu_20261019_v1/transport.py'
_spec = importlib.util.spec_from_file_location('v5_20261020_original_transport', _owner)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
run_python = _module.run_python
upload = _module.upload
entry = _module.entry
