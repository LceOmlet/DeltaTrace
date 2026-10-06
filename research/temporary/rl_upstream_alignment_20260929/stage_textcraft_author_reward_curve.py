"""Stage the original-author curve reader through the existing guarded stager.

Preparation constructs no model. Launch is a separate explicit action after
review, using the provisioned environment/cache and current physical devices.
"""
import argparse
import ast
import hashlib
import subprocess

from stage_environment_entry import AUDIT, ROOT, SSH, SCP
from stage_textcraft_native_readout import BASE, SCRIPT as ORIGINAL_SCRIPT

OUT = ROOT + '/receipts/textcraft-author-reward-curve-20261006-v1'
LOCAL = AUDIT / 'textcraft-degradation-20261005/author-reward-curve-20261006/v1'
NAME = 'verify_textcraft_author_reward_curve.py'
FILES = (NAME, 'author_reward_curve_adapter.py', 'stage_textcraft_author_reward_curve.py')

SCRIPT = ORIGINAL_SCRIPT.replace(
    "DT_TEXTCRAFT_READOUT_ROOT=str(out),DT_TEXTCRAFT_READOUT_BASE=str(base),",
    "DT_TEXTCRAFT_AUTHOR_CURVE_ROOT=str(out),DT_TEXTCRAFT_READOUT_ROOT=str(out),DT_TEXTCRAFT_READOUT_BASE=str(base),"
).replace(
    "PYTHONPATH=str(out)+':'+tail",
    "PYTHONPATH=str(out)+':'+str(base.parent/'textcraft-native-readout-20261006-v2')+':'+env['DT_OFFICIAL_ROOT']+':'+tail"
).replace(
    "source=out/name; assert hashlib.sha256(source.read_bytes()).hexdigest()=='__SHA__'",
    "source=out/name; assert hashlib.sha256(source.read_bytes()).hexdigest()=='__SHA__'\n"
    "diagnostic_sources=__DIAGNOSTIC_SOURCES__\n"
    "for path,h in diagnostic_sources.items():\n"
    " p=pathlib.Path(path); assert hashlib.sha256(p.read_bytes()).hexdigest()==h; ast.parse(p.read_bytes())"
).replace(
    'Isolated 64 real first-response prefixes; original native outcome reader; no rollout/DT/backward/update',
    'Isolated author whole-response curves; saved21 successfirstresponses padded22; saved d_from_A not freshDT; original categorical scorer; no rollout/DT/backward/update'
).replace(
    "native_batch_per_call=4,finite_trace_calls=0,config_source=",
    "native_batch_per_call=1,config_actor_microbatch=4,unique_cases=21,padded_slots=22,"
    "planned_scoring_calls_per_rank=462,finite_trace_calls=0,diagnostic_sources=diagnostic_sources,config_source="
).replace(
    " with (out/'native-owner-inspection.stdout.txt').open('wb') as log:",
    " child_code=\"import json,importlib,hashlib,inspect;names=['author_reward_curve_adapter','verify_textcraft_native_readout','verify_textcraft_author_reward_curve'];print(json.dumps({n:{'path':inspect.getsourcefile(importlib.import_module(n)),'sha256':hashlib.sha256(pathlib.Path(inspect.getsourcefile(importlib.import_module(n))).read_bytes()).hexdigest()} for n in names},indent=2))\"\n"
    " child_code='import pathlib;'+child_code\n"
    " child=subprocess.run([env['VENV_PYTHON'],'-c',child_code],cwd=out,env=dict(env,CUDA_VISIBLE_DEVICES=''),capture_output=True)\n"
    " (out/'fresh-callsite-import.json').write_bytes(child.stdout)\n"
    " (out/'fresh-callsite-import.stderr.txt').write_bytes(child.stderr)\n"
    " assert child.returncode==0,'Inspect original worker callsite imports before model init'\n"
    " receipt['fresh_callsite_import']=dict(path=str(out/'fresh-callsite-import.json'),sha256=hashlib.sha256(child.stdout).hexdigest())\n"
    " with (out/'native-owner-inspection.stdout.txt').open('wb') as log:"
).replace(
    " for key in ['sources','diagnostic_source','input','config_source','checkpoint']:",
    " for key in ['sources','diagnostic_source','diagnostic_sources','input','config_source','checkpoint']:"
).replace('native_reader_submitted', 'native_author_reward_curve_submitted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch'))
    args = parser.parse_args()
    hashes = {OUT + '/' + name: hashlib.sha256((AUDIT / name).read_bytes()).hexdigest() for name in FILES}
    sha = hashes[OUT + '/' + NAME]
    if args.mode == 'prepare':
        subprocess.run(SSH + ['mkdir', '-p', OUT], check=True)
        for name in FILES:
            subprocess.run(SCP + [str(AUDIT / name), f'{SSH[-1]}:{OUT}/{name}'], check=True)
        LOCAL.mkdir(parents=True, exist_ok=True)
        (LOCAL / 'submitted-source').mkdir(exist_ok=True)
        for name in FILES:
            (LOCAL / 'submitted-source' / name).write_bytes((AUDIT / name).read_bytes())
    script = (SCRIPT.replace('__OUT__', OUT).replace('__BASE__', BASE).replace('__MODE__', args.mode)
              .replace('__NAME__', NAME).replace('__SHA__', sha)
              .replace('__DIAGNOSTIC_SOURCES__', repr(hashes)))
    # Validate the exact generated Python body before submitting the CPU stage.
    ast.parse(script.split("\n", 1)[1].rsplit("\nPY", 1)[0])
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / f'{args.mode}.stdout.txt').write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
    if args.mode == 'prepare':
        for name in ('prepared.json', 'native-owner-inspection.json', 'native-owner-inspection.stdout.txt',
                     'fresh-callsite-import.json', 'fresh-callsite-import.stderr.txt', 'effective-config.yaml'):
            subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/{name}', str(LOCAL / name)], check=True)
