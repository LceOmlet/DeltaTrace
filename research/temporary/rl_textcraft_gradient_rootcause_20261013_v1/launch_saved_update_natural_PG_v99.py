"""Reuse the protected native partial-state launcher for actual PG readouts."""
from pathlib import Path

source = Path(__file__).with_name('launch_upper_adapter_partition_v81.py')
code = source.read_text()
code = code.replace('native-upper-adapter-partition-v81', 'native-saved-update-natural-PG-v99')
code = code.replace("sources=[HERE/'observe_upper_adapter_partition_v81.py',HERE/'observe_native_gradient_chain_v4.py',", "sources=[HERE/'observe_saved_update_natural_PG_v99.py',HERE/'observe_upper_adapter_partition_v81.py',HERE/'observe_native_gradient_chain_v4.py',")
code = code.replace("str(source_dir/'observe_upper_adapter_partition_v81.py')", "str(source_dir/'observe_saved_update_natural_PG_v99.py')")
code = code.replace("(source_dir/'observe_upper_adapter_partition_v81.py').read_bytes()", "(source_dir/'observe_saved_update_natural_PG_v99.py').read_bytes()")
code = code.replace('checkpoints=[40,80]', 'checkpoints=[80]')
code = code.replace('diagnostic_parameter_partition_layers=list(range(24,32))', 'diagnostic_parameter_partition_layers=list(range(32))')
exec(compile(code, str(source), 'exec'), globals())
