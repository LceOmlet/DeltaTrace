"""Reuse the original stager; only stage the passive diagnostic modules."""
import argparse
import ast
import hashlib
import json
import subprocess

import stage_textcraft_matched_layout as prior


OUT = prior.existing.ROOT+'/receipts/textcraft-conditional-boundaries-20261006-v2'
NAME = 'verify_textcraft_conditional_boundaries.py'
LOCAL = prior.existing.AUDIT/'textcraft-degradation-20261005/conditional-boundaries-20261006/v2'
DEPS = ('observe_textcraft_conditional_boundaries.py','observe_textcraft_finite_effects.py')


def script(mode,sha):
    original = prior.script(mode,sha).replace(prior.OUT,OUT).replace(prior.NAME,NAME)
    original = prior.replace_once(original,
        "role='Original 7 paired B4 root layouts per rank; observe native selected scores before finite propagation; no rollout/backward/update'",
        "role='Original matched B4 layouts; full-EOS original finite trace and single-EOS original root boundary observation; no rollout/backward/update'")
    original = prior.replace_once(original,
        'finite_decoder_or_replay_calls=0,', 'planned_original_joint_finite_calls_per_rank=7,')
    deps = {name:hashlib.sha256((prior.existing.AUDIT/name).read_bytes()).hexdigest() for name in DEPS}
    # Explicit known diagnostic sources only. No runtime or dependency setup.
    insertion = "\ndiagnostic_dependencies = {}\nfor dependency,expected_sha in "+repr(deps)+".items():\n assert hashlib.sha256((out/dependency).read_bytes()).hexdigest()==expected_sha, dependency\n diagnostic_dependencies[str(out/dependency)] = expected_sha\n"
    original = prior.replace_once(original,"ast.parse(source.read_bytes())", "ast.parse(source.read_bytes())"+insertion)
    original = prior.replace_once(original,
        'sources=checks,diagnostic_source=',
        'sources=checks,diagnostic_dependencies=diagnostic_dependencies,diagnostic_source=')
    original = prior.replace_once(original,
        "for key in ['sources','diagnostic_source','input','config_source','checkpoint']:",
        "for key in ['sources','diagnostic_dependencies','diagnostic_source','input','config_source','checkpoint']:")
    return original


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','launch'))
    args=parser.parse_args()
    source=prior.existing.AUDIT/NAME
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    LOCAL.mkdir(parents=True,exist_ok=True)
    if args.mode=='prepare':
        subprocess.run(prior.existing.SSH+['mkdir','-p',OUT],check=True)
        for name in (NAME,*DEPS):
            subprocess.run(prior.existing.SCP+[str(prior.existing.AUDIT/name),f'{prior.existing.SSH[-1]}:{OUT}/{name}'],check=True)
    code=script(args.mode,sha)
    ast.parse(code.split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0])
    result=subprocess.run(prior.existing.SSH+['bash','-s'],input=code.encode(),capture_output=True)
    (LOCAL/f'{args.mode}.stdout.txt').write_bytes(result.stdout+result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
    if args.mode=='prepare':
        for name in ('prepared.json','native-owner-inspection.json','effective-config.yaml'):
            subprocess.run(prior.existing.SCP+[f'{prior.existing.SSH[-1]}:{OUT}/{name}',str(LOCAL/name)],check=True)
