"""Reuse the guarded native-reader launcher for an isolated label diagnostic."""
import argparse
import hashlib
import subprocess

from stage_environment_entry import AUDIT, ROOT, SSH, SCP
from stage_textcraft_native_readout import BASE, SCRIPT as ORIGINAL_SCRIPT

OUT = ROOT + '/receipts/textcraft-label-encoding-20261006-v2'
LOCAL = AUDIT / 'textcraft-degradation-20261005/label-encoding-20261006/v2'
NAME = 'verify_textcraft_label_encoding.py'

SCRIPT = ORIGINAL_SCRIPT.replace(
    "DT_TEXTCRAFT_READOUT_ROOT=str(out),DT_TEXTCRAFT_READOUT_BASE=str(base),",
    "DT_TEXTCRAFT_LABEL_ROOT=str(out),DT_TEXTCRAFT_READOUT_ROOT=str(out),DT_TEXTCRAFT_READOUT_BASE=str(base),"
).replace(
    "PYTHONPATH=str(out)+':'+tail",
    "PYTHONPATH=str(out)+':'+str(base.parent/'textcraft-native-readout-20261006-v2')+':'+tail"
).replace(
    'Isolated 64 real first-response prefixes; original native outcome reader; no rollout/DT/backward/update',
    'Isolated original64 first-response paired label encoding; unchanged outcome semantics and original native reader; no rollout/DT/backward/update'
).replace('native_reader_submitted', 'native_label_encoding_submitted')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch'))
    args = parser.parse_args()
    sha = hashlib.sha256((AUDIT / NAME).read_bytes()).hexdigest()
    if args.mode == 'prepare':
        subprocess.run(SSH + ['mkdir', '-p', OUT], check=True)
        subprocess.run(SCP + [str(AUDIT / NAME), f'{SSH[-1]}:{OUT}/{NAME}'], check=True)
    script = (SCRIPT.replace('__OUT__', OUT).replace('__BASE__', BASE).replace('__MODE__', args.mode)
              .replace('__NAME__', NAME).replace('__SHA__', sha))
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / f'{args.mode}.stdout.txt').write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
