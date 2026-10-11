"""Compose the existing failing observer with the existing post-return RPC.

No new pre-failure tensor reads, retained tensors, synchronization or model code.
The sole new observation is CPU source data after the original method exits.
"""
from pathlib import Path
import hashlib

import capture_native_two_group_lifecycle as original
import run_original_incident_after_replay_rpc as after_rpc


def main():
    assert hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() == 'f5ced5691c7c593637d2d34b2fb0f863fe2c70658f238e0ab70fcac511869358'
    assert hashlib.sha256(Path(after_rpc.__file__).read_bytes()).hexdigest() == 'e8932f725ee2e6094b8789869fa5f243c0aed56ae3b628d3e7894cb0aa14ff35'
    after_rpc.driver = original
    after_rpc.DRIVER_SHA = 'f5ced5691c7c593637d2d34b2fb0f863fe2c70658f238e0ab70fcac511869358'
    after_rpc.main()


if __name__ == '__main__':
    main()
