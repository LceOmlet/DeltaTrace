"""Launch the existing native observer with its partial-state diagnostic seam."""
import hashlib
from pathlib import Path

source = Path(__file__).with_name('launch_native_gradient_chain_v4.py')
expected = '0897442bbd2425cf2f9c2911c2ad49e6fc1011dc781b3fcb7897cc85b9bb1021'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
code = source.read_text()
changes = {
    "'native-PG-driver-components-v38')": "'native-PG-driver-components-v38','native-upper-adapter-partition-v81')",
    r"r'\|\s*[01]\s+\d+\s+'": r"r'\|\s*[23]\s+\d+\s+'",
    'GPU0/1 occupied': 'GPU2/3 occupied',
    "env['CUDA_VISIBLE_DEVICES']='0,1'": "env['CUDA_VISIBLE_DEVICES']='2,3'",
    "'physical_gpus':[0,1]": "'physical_gpus':[2,3]",
    "grpo=psutil.Process(3174448);assert grpo.create_time()==1791793629.73":
        "grpo=psutil.Process(3174448);assert grpo.create_time()==1791793629.73\ndt=psutil.Process(3140624);assert dt.create_time()==1791993570.92",
    "sources=[HERE/'observe_native_gradient_chain_v4.py',":
        "sources=[HERE/'observe_upper_adapter_partition_v81.py',HERE/'observe_native_gradient_chain_v4.py',",
    "str(source_dir/'observe_native_gradient_chain_v4.py')": "str(source_dir/'observe_upper_adapter_partition_v81.py')",
    "(dest/'job.json').write_text": """if dest.name=='native-upper-adapter-partition-v81':
 record.update(discriminating_quantity=case_description['discriminating_quantity'],
  decision=case_description['decision'],counterfactual_scope=case_description['counterfactual_scope'],
  checkpoints=[40,80],native_PG_passes=0,diagnostic_fixed_seed_VJPs=2,
  diagnostic_parameter_partition_layers=list(range(24,32)),
  wrapper_source=str(source_dir/'observe_upper_adapter_partition_v81.py'),
  wrapper_SHA256=hashlib.sha256((source_dir/'observe_upper_adapter_partition_v81.py').read_bytes()).hexdigest(),
  protected_DTPO_v3=dict(pid=dt.pid,birth=dt.create_time()))
(dest/'job.json').write_text""",
}
for old, new in changes.items():
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(source), 'exec'), globals())
