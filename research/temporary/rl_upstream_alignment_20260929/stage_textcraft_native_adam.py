"""Compose the existing guarded staging entry for one saved native update."""
import argparse
import hashlib
import json
import subprocess

import stage_textcraft_native_readout as prior
from stage_environment_entry import AUDIT, ROOT, SSH, SCP

OUT = ROOT + '/receipts/textcraft-native-adam-20261006-v1'
LOCAL = AUDIT / 'textcraft-degradation-20261005/native-adam-20261006/v1'
NAME = 'verify_textcraft_native_adam.py'
DEPS = ('observe_native_adam_update.py',)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch'))
    args = parser.parse_args()
    hashes = {name: hashlib.sha256((AUDIT / name).read_bytes()).hexdigest() for name in (NAME, *DEPS)}
    if args.mode == 'prepare':
        subprocess.run(SSH + ['mkdir', '-p', OUT], check=True)
        subprocess.run(SCP + [str(AUDIT / name) for name in hashes] + [f'{SSH[-1]}:{OUT}/'], check=True)
    script = prior.SCRIPT
    old_inputs = "cases=base/'readout-first-response-cases.json'\nassert cases.is_file(), 'CPU mapping must prepare and verify the exact original prefixes first'\ndata=json.loads(cases.read_bytes()); assert [len(rows) for rows in data['rank_cases']]==[32,32]\nassert len({row['traj_uid'] for rows in data['rank_cases'] for row in rows})==64"
    new_inputs = "cases=base/'native-optimizer-minibatch.pkl'\nboundary=json.loads((base/'native-minibatch-update-boundary.json').read_bytes())\nassert hashlib.sha256(cases.read_bytes()).hexdigest()==boundary['snapshot']['sha256']"
    assert old_inputs in script
    script = script.replace(old_inputs, new_inputs)
    script = script.replace("assert hashlib.sha256(source.read_bytes()).hexdigest()=='__SHA__'", "assert hashlib.sha256(source.read_bytes()).hexdigest()=='__SHA__'\nextra_hashes=__EXTRA_HASHES__\nfor filename, expected in extra_hashes.items():\n p=out/filename; assert hashlib.sha256(p.read_bytes()).hexdigest()==expected; ast.parse(p.read_bytes())")
    script = script.replace("diagnostic_source=dict(path=str(source),sha256='__SHA__'),", "diagnostic_source=dict(path=str(source),sha256='__SHA__'),\n diagnostic_dependencies={n:dict(path=str(out/n),sha256=h) for n,h in extra_hashes.items()},")
    script = script.replace("'sources','diagnostic_source','input','config_source','checkpoint'", "'sources','diagnostic_source','diagnostic_dependencies','input','config_source','checkpoint'")
    script = script.replace("role='Isolated 64 real first-response prefixes; original native outcome reader; no rollout/DT/backward/update'", "role='Isolated saved global64, original complete checkpoint restore and three single real original VERL updates; no rollout or DT recomputation'")
    script = script.replace("optimizer_steps=0,\n native_batch_per_call=4", "planned_optimizer_steps_per_rank=3,\n native_batch_per_call=4")
    script = script.replace("status='native_reader_submitted'", "status='native_adam_observation_submitted'")
    script = (script.replace('__OUT__', OUT).replace('__BASE__', prior.BASE)
        .replace('__MODE__', args.mode).replace('__NAME__', NAME).replace('__SHA__', hashes[NAME])
        .replace('__EXTRA_HASHES__', repr({name: hashes[name] for name in DEPS})))
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / f'{args.mode}.stdout.txt').write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
