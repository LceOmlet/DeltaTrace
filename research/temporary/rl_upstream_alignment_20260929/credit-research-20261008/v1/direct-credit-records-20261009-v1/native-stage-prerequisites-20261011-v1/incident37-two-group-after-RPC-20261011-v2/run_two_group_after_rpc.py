"""Compose the existing failing observer with the existing post-return RPC.

No new pre-failure tensor reads, retained tensors, synchronization or model code.
The sole new observation is CPU source data after the original method exits.
"""
from pathlib import Path
import argparse
import hashlib

import capture_native_two_group_lifecycle as original
import run_original_incident_after_replay_rpc as after_rpc


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    args, _ = parser.parse_known_args()
    selection = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-incident37-competing-weight-capture-20261011-v1/selection.json').read_bytes()
    assert hashlib.sha256(selection).hexdigest() == '9a82a6912a18e44cf11cca67bcc217ffbcdf6d16dbc8adc71bd487f461c5cd1c'
    target = args.output.parent / 'selection.json'
    if target.exists():
        assert target.read_bytes() == selection
    else:
        target.write_bytes(selection)
    assert hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() == 'f5ced5691c7c593637d2d34b2fb0f863fe2c70658f238e0ab70fcac511869358'
    assert hashlib.sha256(Path(after_rpc.__file__).read_bytes()).hexdigest() == 'e8932f725ee2e6094b8789869fa5f243c0aed56ae3b628d3e7894cb0aa14ff35'
    after_rpc.driver = original
    after_rpc.DRIVER_SHA = 'f5ced5691c7c593637d2d34b2fb0f863fe2c70658f238e0ab70fcac511869358'
    after_rpc.main()


if __name__ == '__main__':
    main()
