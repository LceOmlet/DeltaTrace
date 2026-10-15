"""Reuse the native partial-state observer for saved whole-direction feedback."""
import hashlib
from pathlib import Path

source = Path(__file__).with_name('launch_upper_adapter_partition_v81.py')
code = source.read_text().replace('native-upper-adapter-partition-v81', 'native-full-policy-saved-update-feedback-v90')
code = code.replace('checkpoints=[40,80]', 'checkpoints=[80,160]')
code = code.replace('diagnostic_parameter_partition_layers=list(range(24,32))', 'diagnostic_parameter_partition_layers=list(range(32))')
exec(compile(code, str(source), 'exec'), globals())
