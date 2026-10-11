"""Delegate to the verified TextCraft launcher with Ray's native setup hook.

All CLI arguments, workload, checkpoint restoration and training remain owned
by that launcher and VERL. Only the two worker-setup configuration keys are
added. This entry is opt-in and does not modify the historical entry or packages.
"""
import hashlib
import os
from pathlib import Path
import runpy
import sys

OWNER_LAUNCHER_SHA256 = "4b6bc6c4009f2ee68b32eaff0139025709df0e4d69963f969def55b0c1682c56"


def configure_worker_setup():
    import owner_runtime_options

    original = owner_runtime_options.runtime_options

    def runtime_options():
        options = original()
        options['+ray_init.runtime_env.worker_process_setup_hook'] = 'metax_fsdp_event_wait.worker_setup'
        options['+ray_init.runtime_env.env_vars.DT_FSDP_EVENT_WAIT_RECEIPTS'] = os.environ['DT_FSDP_EVENT_WAIT_RECEIPTS']
        return options

    owner_runtime_options.runtime_options = runtime_options


def owner_launcher():
    path = Path(os.environ['DT_FSDP_OWNER_LAUNCHER'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != OWNER_LAUNCHER_SHA256:
        raise RuntimeError('Verified TextCraft launcher differs from the recorded owner')
    return path


if __name__ == '__main__':
    path = owner_launcher()
    configure_worker_setup()
    sys.argv[0] = str(path)
    runpy.run_path(str(path), run_name='__main__')
