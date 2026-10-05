"""Reuse the guarded GDN stager for the next passive owner boundary."""
import argparse
import ast
import hashlib
import subprocess

import stage_textcraft_conditional_gdn as prior


OUT = prior.prior.prior.prior.existing.ROOT + '/receipts/textcraft-conditional-fla-20261006-v1'
NAME = 'verify_textcraft_conditional_fla.py'
LOCAL = prior.prior.prior.prior.existing.AUDIT / 'textcraft-degradation-20261005/conditional-fla-20261006/v1'
DEPS = ('observe_textcraft_conditional_fla.py', 'verify_textcraft_conditional_gdn.py',
        'observe_textcraft_conditional_gdn.py', 'verify_textcraft_conditional_primitives.py',
        'observe_textcraft_conditional_primitives.py', 'verify_textcraft_conditional_boundaries.py',
        'observe_textcraft_conditional_boundaries.py', 'observe_textcraft_finite_effects.py',
        'observe_textcraft_norm_operands.py')


def script(mode, sha):
    previous_deps, previous_out, previous_name = prior.DEPS, prior.OUT, prior.NAME
    try:
        prior.DEPS, prior.OUT, prior.NAME = DEPS, OUT, NAME
        return prior.script(mode, sha)
    finally:
        prior.DEPS, prior.OUT, prior.NAME = previous_deps, previous_out, previous_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch'))
    args = parser.parse_args()
    audit = prior.prior.prior.prior.existing.AUDIT
    source = audit / NAME
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    LOCAL.mkdir(parents=True, exist_ok=True)
    if args.mode == 'prepare':
        subprocess.run(prior.prior.prior.prior.existing.SSH + ['mkdir', '-p', OUT], check=True)
        for name in (NAME, *DEPS):
            subprocess.run(prior.prior.prior.prior.existing.SCP + [str(audit / name),
                f'{prior.prior.prior.prior.existing.SSH[-1]}:{OUT}/{name}'], check=True)
    code = script(args.mode, sha)
    ast.parse(code.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0])
    result = subprocess.run(prior.prior.prior.prior.existing.SSH + ['bash', '-s'],
        input=code.encode(), capture_output=True)
    (LOCAL / f'{args.mode}.stdout.txt').write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
    if args.mode == 'prepare':
        for name in ('prepared.json', 'native-owner-inspection.json', 'effective-config.yaml'):
            subprocess.run(prior.prior.prior.prior.existing.SCP + [f'{prior.prior.prior.prior.existing.SSH[-1]}:{OUT}/{name}',
                str(LOCAL / name)], check=True)
