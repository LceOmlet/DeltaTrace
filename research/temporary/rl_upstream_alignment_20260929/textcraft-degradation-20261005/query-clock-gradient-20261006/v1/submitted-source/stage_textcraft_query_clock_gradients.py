"""Reuse the completed support-gradient stager; prepare before parent review."""
import argparse
import ast
import hashlib
import subprocess

import stage_textcraft_grpo_support_gradients as prior
from stage_environment_entry import AUDIT, ROOT, SSH, SCP

OUT = ROOT + '/receipts/textcraft-query-clock-gradient-20261006-v1'
LOCAL = AUDIT / 'textcraft-degradation-20261005/query-clock-gradient-20261006/v1'
NAME = 'verify_textcraft_query_clock_gradients.py'
DEPS = ('observe_textcraft_query_clock_gradients.py', 'reward_readout_clock_candidate.py')
CANDIDATE = AUDIT.parents[2] / 'experiments/rl/reward_readout.py'
PINNED = dict(prior.PINNED)
PINNED[prior.prior.BASE + '/map_textcraft_native64_readout.py'] = 'aa5be01fd020d7f1c52d6321aaa7b8483dfe76f5c69b9203d221aaff5715fd9a'
PINNED[prior.prior.BASE + '/analyze_textcraft_native_minibatch.py'] = 'ec4512f2ce66bce1ca262075b3026080bf837909d8f62854b90c90137877c341'


def prepared_script(mode, hashes):
    # Configure the existing diagnostic staging composer, then use its intact
    # original source guards, inherited environment and physical GPU guard.
    prior.OUT, prior.LOCAL, prior.NAME, prior.DEPS, prior.PINNED = OUT, LOCAL, NAME, DEPS, PINNED
    script = prior.prepared_script(mode, hashes)
    script = script.replace("role='Prepared isolated same saved native64/checkpoint25; two support groups each two original backward passes with optimizer/scheduler no-op; no rollout or DT'",
        "role='Prepared saved native64/checkpoint25; old PG/H then original-owner query-clock DT recomputation and new PG/H; four backward passes, no rollout/optimizer/scheduler update'")
    script = script.replace("status='native_support_gradient_submitted'", "status='native_query_clock_gradient_submitted'")
    script = script.replace("names=('observe_native_optimizer_minibatch','observe_native_actor_loss_gradients','observe_textcraft_native_batches')",
        "names=('observe_native_optimizer_minibatch','observe_native_actor_loss_gradients','observe_textcraft_native_batches','observe_textcraft_query_clock_gradients','reward_readout_clock_candidate')")
    script = script.replace("assert item['sha256']==pinned[item['path']], 'Helper callsite actual path/hash mismatch'",
        "expected=pinned.get(item['path'],extra_hashes.get(pathlib.Path(item['path']).name)); assert item['sha256']==expected, 'Helper callsite actual path/hash mismatch'")
    ast.parse(script.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0])
    return script


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch'))
    args = parser.parse_args()
    files = {NAME: AUDIT / NAME, DEPS[0]: AUDIT / DEPS[0], DEPS[1]: CANDIDATE}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    assert hashes[DEPS[1]] == '94a7afbc09da72b62572d31fd32a6534f6e8f3daf656fce1011cdfa68b3c3e2b'
    if args.mode == 'prepare':
        subprocess.run(SSH + ['mkdir', '-p', OUT], check=True)
        for name, path in files.items():
            subprocess.run(SCP + [str(path), f'{SSH[-1]}:{OUT}/{name}'], check=True)
    script = prepared_script(args.mode, hashes)
    LOCAL.mkdir(parents=True, exist_ok=True)
    (LOCAL / f'{args.mode}.sh').write_text(script, encoding='utf-8')
    result = subprocess.run(SSH + ['bash', '-s'], input=script.encode(), capture_output=True)
    (LOCAL / f'{args.mode}.stdout.txt').write_bytes(result.stdout + result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()
    if args.mode == 'prepare':
        subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/{name}' for name in
            ('prepared.json', 'native-owner-inspection.json', 'native-owner-inspection.stdout.txt',
             'worker-callsite-import-inspection.json', 'worker-callsite-import.stdout.txt')] + [str(LOCAL)], check=True)
