"""Compose the existing guarded stager for labels-only native PG observation."""
import argparse
import ast
import hashlib
import subprocess

import stage_textcraft_grpo_support_gradients as prior
from stage_environment_entry import AUDIT, ROOT, SSH, SCP

OUT = ROOT + '/receipts/textcraft-label-gradient-20261006-v1'
LOCAL = AUDIT / 'textcraft-degradation-20261005/equivalent-label-gradient-20261006/v1'
NAME = 'verify_textcraft_label_gradients.py'
DEPS = ('observe_textcraft_label_gradients.py',)
PINNED = dict(prior.PINNED)
PINNED[prior.prior.BASE + '/map_textcraft_native64_readout.py'] = 'aa5be01fd020d7f1c52d6321aaa7b8483dfe76f5c69b9203d221aaff5715fd9a'
PINNED[prior.prior.BASE + '/analyze_textcraft_native_minibatch.py'] = 'ec4512f2ce66bce1ca262075b3026080bf837909d8f62854b90c90137877c341'


def prepared_script(mode, hashes):
    # The existing stager owns the source, environment, PID and physical-GPU
    # guards; this composition only names this isolated recipe and dependency.
    prior.OUT, prior.LOCAL, prior.NAME, prior.DEPS, prior.PINNED = OUT, LOCAL, NAME, DEPS, PINNED
    script = prior.prepared_script(mode, hashes)
    old_role = "role='Prepared isolated same saved native64/checkpoint25; two support groups each two original backward passes with optimizer/scheduler no-op; no rollout or DT'"
    assert old_role in script
    script = script.replace(old_role,
        "role='Prepared saved native64/checkpoint25; original PG/H then labels-only original DT recomputation and swapped PG/H plus cross-PG Gram; no rollout or optimizer/scheduler update'")
    script = script.replace("status='native_support_gradient_submitted'", "status='native_label_gradient_submitted'")
    script = script.replace('planned_backward_passes_per_rank=4,finite_trace_calls=0',
        'planned_backward_passes_per_rank=4,planned_original_DT_recompute=True,CPU_inspection_finite_trace_calls=0')
    env_marker = "env['DT_TEXTCRAFT_ADAM_RECIPE_ROOT']='" + prior.ADAM + "'"
    assert env_marker in script
    script = script.replace(env_marker, env_marker + "\nenv['DT_TEXTCRAFT_LABEL_GRADIENT_ROOT']=str(out)")
    names = "names=('observe_native_optimizer_minibatch','observe_native_actor_loss_gradients','observe_textcraft_native_batches')"
    assert names in script
    script = script.replace(names,
        "names=('observe_native_optimizer_minibatch','observe_native_actor_loss_gradients','observe_textcraft_native_batches','observe_textcraft_label_gradients','verify_textcraft_native_adam')")
    source_check = "assert item['sha256']==pinned[item['path']], 'Helper callsite actual path/hash mismatch'"
    assert source_check in script
    script = script.replace(source_check,
        "expected=pinned.get(item['path'],extra_hashes.get(pathlib.Path(item['path']).name)); assert item['sha256']==expected, 'Helper callsite actual path/hash mismatch'")
    ast.parse(script.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0])
    return script


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'launch'))
    args = parser.parse_args()
    hashes = {name: hashlib.sha256((AUDIT / name).read_bytes()).hexdigest() for name in (NAME, *DEPS)}
    if args.mode == 'prepare':
        subprocess.run(SSH + ['mkdir', '-p', OUT], check=True)
        subprocess.run(SCP + [str(AUDIT / name) for name in hashes] + [f'{SSH[-1]}:{OUT}/'], check=True)
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
