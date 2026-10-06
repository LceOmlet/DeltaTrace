"""Reuse the prepared source/PID/resource guards for a query-only diagnostic."""
import argparse
import hashlib
import subprocess
from stage_environment_entry import AUDIT, ROOT, SSH, SCP
from stage_textcraft_native_readout import BASE
from stage_textcraft_label_encoding import SCRIPT as OWNER_SCRIPT

OUT = ROOT + '/receipts/textcraft-response-clock-20261006-v1'
LOCAL = AUDIT / 'textcraft-degradation-20261005/response-clock-20261006/v1'
NAME = 'verify_textcraft_response_clock.py'
SCRIPT = OWNER_SCRIPT.replace('Isolated original64 first-response paired label encoding; unchanged outcome semantics and original native reader; no rollout/DT/backward/update',
    'Isolated original64 first-response current-action-clock wording; unchanged labels, return, original native reader; no rollout/DT/backward/update')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch'))
    args = parser.parse_args()
    sha = hashlib.sha256((AUDIT / NAME).read_bytes()).hexdigest()
    if args.mode == 'prepare':
        subprocess.run(SSH + ['mkdir', '-p', OUT], check=True)
        subprocess.run(SCP + [str(AUDIT / NAME), f'{SSH[-1]}:{OUT}/{NAME}'], check=True)
    script = SCRIPT.replace('__OUT__', OUT).replace('__BASE__', BASE).replace('__MODE__', args.mode).replace('__NAME__', NAME).replace('__SHA__', sha)
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / f'{args.mode}.stdout.txt').write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
