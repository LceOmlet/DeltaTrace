"""Compare the existing native GPU-storage flag on one bounded long B4.

Compose the completed A-B-A observer; the finite runner and its arithmetic
are unchanged. The original result is returned to training. Optional control
calls are validation work and must be excluded from production efficiency.
"""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('check_existing_FA_capture_resource_pair.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '08ae1c9c2ac431e15c142fffa021970741378d697626060fe434c1ef0aa095ac'
code = owner.read_text()
start = code.index('        import ast,hashlib,__future__,torch\n')
end = code.index("''')\nanchor = '            count=[0]'", start)
code = code[:start] + '''        import hashlib,torch
        from pathlib import Path
        original_native=runner._bounded_native_mixer_storage['original_attribute']
        native_file=Path(original_native.__func__.__code__.co_filename)
        assert hashlib.sha256(native_file.read_bytes()).hexdigest()=='6a112224b3d4e63d047a576d7df68ff1ee65e4bbacfcb0014f84c17bd82b8520'
''' + code[end:]
for old, new in (
    ('_FA_capture_resource_pair_scope_installed_v1', '_long_GPU_resident_resource_pair_scope_installed_v1'),
    ('native_DT_FA_capture_resource_pair', 'native_DT_long_GPU_resident_resource_pair'),
    ('FA_capture_resource_pair_observer_', 'long_GPU_resident_resource_pair_observer_'),
    ('FA_capture_memory_guard', 'long_GPU_resident_memory_guard'),
    ('optional_same_input_FA_capture_pair', 'optional_same_input_long_GPU_resident_pair'),
    ('FA-capture-original-gate-host-pair-v1', 'long-GPU-resident-native-owner-pair-v1'),
    ('2048<paired.shape[1]<=14000', '2048<paired.shape[1]<=12288'),
    ('extra_bound=12*', 'extra_bound=48*'),
    ('extra_FA_capture_bound', 'extra_resident_capture_bound'),
    ('baseline_flags=list(flags),candidate_flags=list(flags)',
     'baseline_flags=list(flags),candidate_flags=[False,True,False]'),
    ('candidate_FA_capture_cuda=True,MLP_gate_and_GDN_original_storage=True',
     'candidate_offload_replay_mixer=False,all_original_capture_owners_unchanged=True'),
    ('candidate_source=str(candidate_path),candidate_source_SHA256=hashlib.sha256(candidate_path.read_bytes()).hexdigest()',
     'candidate_source=str(native_file),candidate_source_SHA256=hashlib.sha256(native_file.read_bytes()).hexdigest()'),
    ('runner.fa_replay_capture_cuda=True', 'runner.offload_replay_mixer=False'),
    ('candidate=candidate_attribute(*native_args,**native_kwargs)',
     'candidate=original_native(*native_args,**native_kwargs)'),
    ('                            runner.fa_replay_capture_cuda=old_fa\n',
     '                            runner.offload_replay_mixer=flags[2]\n'),
):
    assert old in code, old
    code = code.replace(old, new)
exec(compile(code, str(owner), 'exec'), globals())
