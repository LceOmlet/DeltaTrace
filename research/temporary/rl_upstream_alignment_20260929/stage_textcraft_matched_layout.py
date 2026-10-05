"""Reuse the existing native-reader stager for original paired-root observation."""
import argparse
import hashlib
import json
import subprocess

import stage_textcraft_native_readout as existing


OUT = existing.ROOT + '/receipts/textcraft-matched-layout-20261006-v1'
NAME = 'verify_textcraft_matched_layout.py'
LOCAL = existing.AUDIT / 'textcraft-degradation-20261005/native-layout-20261006/v1'


def replace_once(source, old, new):
    assert source.count(old) == 1, ('Existing stager changed', old)
    return source.replace(old, new, 1)


def script(mode, sha):
    # Existing staging owns environment reuse, original-source validation,
    # CPU binding inspection, occupancy checks, and single submission. Only
    # the diagnostic input representation and receipt description differ.
    source = replace_once(existing.SCRIPT, "cases=base/'readout-first-response-cases.json'",
        "cases=base/'native64-first-response-matched-layout-pack.json'")
    source = replace_once(source,
        "data=json.loads(cases.read_bytes()); assert [len(rows) for rows in data['rank_cases']]==[32,32]\nassert len({row['traj_uid'] for rows in data['rank_cases'] for row in rows})==64",
        "data=json.loads(cases.read_bytes()); assert [len(rows) for rows in data['rank_groups']]==[7,7]\nassert data['coverage']['unique_fixed_random_token_probes']==42\nassert all(len(g['samples'])==4 for groups in data['rank_groups'] for g in groups)")
    source = replace_once(source,
        "role='Isolated 64 real first-response prefixes; original native outcome reader; no rollout/DT/backward/update'",
        "role='Original 7 paired B4 root layouts per rank; observe native selected scores before finite propagation; no rollout/backward/update'")
    source = replace_once(source, "native_batch_per_call=4,finite_trace_calls=0,",
        "native_logical_batch_per_call=4,native_interleaved_rows=8,finite_decoder_or_replay_calls=0,")
    return (source.replace('__OUT__', OUT).replace('__BASE__', existing.BASE).replace('__MODE__', mode)
            .replace('__NAME__', NAME).replace('__SHA__', sha))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch'))
    args = parser.parse_args()
    source = existing.AUDIT / NAME
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    LOCAL.mkdir(parents=True, exist_ok=True)
    if args.mode == 'prepare':
        subprocess.run(existing.SSH + ['mkdir', '-p', OUT], check=True)
        subprocess.run(existing.SCP + [str(source), f'{existing.SSH[-1]}:{OUT}/{NAME}'], check=True)
    result = subprocess.run(existing.SSH + ['bash', '-s'], input=script(args.mode, sha).encode(), capture_output=True)
    (LOCAL / f'{args.mode}.stdout.txt').write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
