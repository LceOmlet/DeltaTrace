"""Reuse the guarded original stager for a same-pair passive observation."""
import argparse
import ast
import hashlib
import subprocess

import stage_textcraft_conditional_fla as prior
from stage_environment_entry import ROOT, AUDIT, SSH, SCP


OUT = ROOT + '/receipts/textcraft-joint-fla-20261006-v1'
NAME = 'verify_textcraft_joint_fla.py'
LOCAL = AUDIT / 'textcraft-degradation-20261005/joint-fla-20261006/v1'
DEPS = ('observe_textcraft_joint_fla.py', prior.NAME, *prior.DEPS)


def script(mode, sha):
    saved = prior.DEPS, prior.OUT, prior.NAME
    try:
        prior.DEPS, prior.OUT, prior.NAME = DEPS, OUT, NAME
        return prior.script(mode, sha)
    finally:
        prior.DEPS, prior.OUT, prior.NAME = saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch'))
    args = parser.parse_args()
    source = AUDIT / NAME
    ast.parse(source.read_text(encoding='utf-8'))
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    LOCAL.mkdir(parents=True, exist_ok=True)
    if args.mode == 'prepare':
        subprocess.run(SSH + ['mkdir', '-p', OUT], check=True)
        subprocess.run(SCP + [str(AUDIT / name) for name in (NAME, *DEPS)]
            + [f'{SSH[-1]}:{OUT}/'], check=True)
    code = script(args.mode, sha)
    ast.parse(code.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0])
    result = subprocess.run(SSH + ['bash', '-s'], input=code.encode(), capture_output=True)
    (LOCAL / f'{args.mode}.stdout.txt').write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
    if args.mode == 'prepare':
        for name in ('prepared.json', 'native-owner-inspection.json', 'effective-config.yaml'):
            subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/{name}', str(LOCAL / name)], check=True)
