"""Compose the verified whitening stager for one real native update and restore.

No owner, configuration, environment setup, or staging state machine is copied.
The existing guarded stager owns source checks, preparation, resource checks,
and submission; this entry only names the bounded observation and its imports.
"""
import argparse
import ast
import hashlib
import subprocess

import stage_textcraft_official_whitening as prior
from stage_environment_entry import AUDIT, ROOT, SSH, SCP

OUT = ROOT + '/receipts/textcraft-whitened-update-20261006-v1'
LOCAL = AUDIT / 'textcraft-degradation-20261005/whitened-update-20261006/v1'
NAME = 'verify_textcraft_whitened_update.py'
WHITE_ROOT = prior.OUT
WHITE_SHA = 'f20d727252d66c9c023526f4f9e91108333d74edfe6f9dbb3deec254f1b4e774'


def prepared_script(mode, hashes):
    old = {key: getattr(prior, key) for key in ('OUT', 'LOCAL', 'NAME', 'PINNED')}
    try:
        prior.OUT, prior.LOCAL, prior.NAME = OUT, LOCAL, NAME
        prior.PINNED = {**prior.PINNED,
            WHITE_ROOT + '/verify_textcraft_official_whitening.py': WHITE_SHA}
        script = prior.prepared_script(mode, hashes,
            verl_root=prior.VERL, trainer_sha=prior.TRAINER_SHA,
            environment_file=prior.ENVIRONMENT_FILE,
            environment_sha=prior.ENVIRONMENT_SHA)
    finally:
        for key, value in old.items():
            setattr(prior, key, value)
    marker = "PYTHONPATH=str(out)+':'+str(overlay_verl_root)"
    assert marker in script
    script = script.replace(marker,
        "PYTHONPATH=str(out)+':" + WHITE_ROOT + ":'+str(overlay_verl_root)")
    imports = "'verify_textcraft_native_adam','verify_textcraft_official_whitening')"
    assert imports in script
    script = script.replace(imports,
        "'verify_textcraft_native_adam','verify_textcraft_official_whitening','verify_textcraft_whitened_update')")
    old_scope = 'Prepared saved native64/checkpoint25 raw/officially whitened full-batch advantages, original PG/H/KL and cross-PG Gram; no rollout, DT or real optimizer/scheduler step'
    assert old_scope in script
    script = script.replace(old_scope,
        'Prepared saved native64/checkpoint25 official full-batch whitening, one real original VERL update, original logprob and manual checkpoint save/restore; no rollout or DT recomputation')
    script = script.replace('planned_backward_passes_per_rank=6',
        'planned_backward_passes_per_rank=1,planned_optimizer_steps_per_rank=1')
    script = script.replace("status='native_official_whitening_submitted'",
        "status='native_whitened_update_submitted'")
    if mode == 'launch':
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                           cwd=AUDIT, text=True).strip()
        script = script.replace('receipt.update(pid=child.pid,',
            'receipt.update(submission_repository_commit=' + repr(revision) + ',pid=child.pid,')
    ast.parse(script.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0])
    return script


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch'))
    args = parser.parse_args()
    hashes = {NAME: hashlib.sha256((AUDIT / NAME).read_bytes()).hexdigest()}
    if args.mode == 'prepare':
        subprocess.run(SSH + ['mkdir', '-p', OUT], check=True)
        subprocess.run(SCP + [str(AUDIT / NAME), f'{SSH[-1]}:{OUT}/{NAME}'], check=True)
    script = prepared_script(args.mode, hashes)
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / f'{args.mode}.sh').write_text(script, encoding='utf-8')
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
    (LOCAL / f'{args.mode}.stdout.txt').write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
    if args.mode == 'prepare':
        names = ('prepared.json', 'native-owner-inspection.json',
                 'native-owner-inspection.stdout.txt', 'worker-callsite-import-inspection.json',
                 'worker-callsite-import.stdout.txt')
        subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/{name}' for name in names] + [str(LOCAL)], check=True)
