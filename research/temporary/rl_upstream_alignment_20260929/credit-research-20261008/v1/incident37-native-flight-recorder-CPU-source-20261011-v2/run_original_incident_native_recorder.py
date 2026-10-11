"""Use Ray's setup hook solely to export the owner's recorder after reporting."""
import functools
import hashlib
import os
from pathlib import Path
import ray
import capture_native_incident37_GEMM_rows_20261011 as gemm

def main():
    assert hashlib.sha256(Path(gemm.__file__).read_bytes()).hexdigest() == '35262c5946fe86989425a68bbdba4414fd22ee13e000f404ef6c45a89889d3f8'
    original = ray.init
    @functools.wraps(original)
    def init(*args, **kwargs):
        runtime_env = dict(kwargs.get('runtime_env') or {})
        assert not runtime_env.get('worker_process_setup_hook')
        runtime_env['worker_process_setup_hook'] = 'dump_native_collectives_after_report.install'
        values = dict(runtime_env.get('env_vars') or {})
        for key in ('DT_INCIDENT_AFTER_REPORT_RESULT','DT_INCIDENT_AFTER_REPORT_HOOK_RECORD',
                    'TORCH_NCCL_TRACE_BUFFER_SIZE','TORCH_NCCL_TRACE_CPP_STACK','TORCH_NCCL_ENABLE_TIMING'):
            values[key] = os.environ[key]
        runtime_env['env_vars'] = values
        kwargs['runtime_env'] = runtime_env
        return original(*args, **kwargs)
    ray.init = init
    try:
        gemm.main()
    finally:
        ray.init = original

if __name__ == '__main__':
    main()
