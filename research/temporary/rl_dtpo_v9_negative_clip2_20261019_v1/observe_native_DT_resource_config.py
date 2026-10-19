"""Read the loaded runner's existing resource options via the original RPC."""
import hashlib
from pathlib import Path

owner = Path(__file__).with_name('inspect_native_workers.py')
assert hashlib.sha256(owner.read_bytes()).hexdigest() == '986610703d7d1ad9616329c4c6e52534a703272a2bb2899078a807adca1d46c0'
code = owner.read_text().replace('v9-negative2-eight-GPU-initial-v1',
                                'native-resource-config-readonly-v1')
anchor = "exec(compile(code,str(owner),'exec'),globals())"
assert code.count(anchor) == 1
observation = '''    runner=actor._deltatrace_producer.runner
    result['native_DT_resource_config']={name:getattr(runner,name,None) for name in (
        'gdn_head_batch_size','gdn_gpu_capture_names','copy_replay_captures',
        'offload_replay_mixer','pin_root_host','pin_replay_host','defer_diagnostics',
        'fa_coefficient_suffix','gdn_coefficient_suffix','compact_gdn_captures')}
    result['native_DT_resource_config_read_only']=True
    return result'''
code = code.replace(anchor,
    "assert code.count('    return result')==1\ncode=code.replace('    return result',"
    + repr(observation) + ")\n" + anchor)
exec(compile(code, str(owner), 'exec'), globals())
