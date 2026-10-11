"""Verify the production hook against the installed owner and native Ray startup."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import ray
import torch
from torch.distributed.fsdp._fully_shard import _fsdp_param_group as owner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    function = owner.FSDPParamGroup.finalize_backward
    original_code = function.__code__
    import metax_fsdp_event_wait as fix
    assert function.__code__ is original_code  # import is default-inert
    installed = fix.install()
    assert owner.FSDPParamGroup.finalize_backward is function
    assert installed['original_code_SHA256'] == 'f29d8303a7d522339cb829f5487f7b4322fed1d2dbf0ef1f7bfafa71726b87ff'
    assert installed['patched_code_SHA256'] == '1609439cddb3ffa0b30e18fa55ad6119c1527e009515335eafca5b24dcb3abfc'
    assert fix.install() == installed
    assert not torch.cuda.is_initialized()
    ray.init(num_cpus=1, include_dashboard=False, runtime_env={
        'worker_process_setup_hook': 'metax_fsdp_event_wait.worker_setup',
        'env_vars': {'PYTHONPATH': os.environ['PYTHONPATH'],
                     'DT_FSDP_EVENT_WAIT_RECEIPTS': str(args.output/'worker-receipts')},
    })
    try:
        @ray.remote
        def inspect_worker():
            import metax_fsdp_event_wait as worker_fix
            import torch
            return dict(receipt=worker_fix.install(), CUDA_initialized=torch.cuda.is_initialized())
        result = ray.get(inspect_worker.remote())
        assert result['receipt'] == installed
        assert not result['CUDA_initialized']
        out = dict(default_import_inert=True, original_function_identity_retained=True,
                   candidate_and_production_original_and_patched_code_SHA256_identical=True,
                   repeated_install_idempotent=True, native_Ray_hook_result=result,
                   installed_files_modified=0, GPU_calls=0)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'verification.json').write_text(json.dumps(out, indent=2)+'\n')
        print(json.dumps(out), flush=True)
    finally:
        ray.shutdown()


if __name__ == '__main__':
    main()
