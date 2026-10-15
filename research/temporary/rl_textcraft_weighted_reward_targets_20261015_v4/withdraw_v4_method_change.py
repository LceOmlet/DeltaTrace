"""Withdraw only v4: its linear credit does not establish the original PG equivalence.

Use the existing protected process-group stop owner; never launch a replacement.
"""
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-weighted-reward-targets-20261015-v4'
owner = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1/submit_DTPO_variance_v3.py'
text = owner.read_text()
imports = text[:text.index("ROOT=Path(")]
block = text[text.index('protected=[];protected_pids=set()'):text.index('physical=subprocess.check_output')]
old = "protected_identities=[(3174448,1791793629.73),(ppo['pid'],ppo['birth'])]"
assert old in block
block = block.replace(old, "protected_identities=[(ppo['pid'],ppo['birth'])]\nfor pid in (3953367,3955296,2617231,2617232):\n    q=psutil.Process(pid);assert q.status()!=psutil.STATUS_ZOMBIE\n    protected_identities.append((pid,q.create_time()))")
block = block.replace("stop=PREPARED/'previous-DTPO-stop.json'", "stop=OUT/'DTPO-v4-method-withdrawn.json'")
old = """    marker=checkpoint/'latest_checkpointed_iteration.txt';step=int(marker.read_text())
    for rank in (0,1):
        for kind in ('model','optim','extra_state'):
            f=checkpoint/f'global_step_{step}/actor/{kind}_world_size_2_rank_{rank}.pt'
            assert f.is_file() and f.stat().st_size>0
    assert (checkpoint/f'global_step_{step}/data.pt').is_file()"""
assert old in block
new = """    marker=checkpoint/'latest_checkpointed_iteration.txt';step=int(marker.read_text()) if marker.exists() else None
    if step is not None:
        for rank in (0,1):
            for kind in ('model','optim','extra_state'):
                f=checkpoint/f'global_step_{step}/actor/{kind}_world_size_2_rank_{rank}.pt'
                assert f.is_file() and f.stat().st_size>0
        assert (checkpoint/f'global_step_{step}/data.pt').is_file()"""
block = block.replace(old, new)
block = block.replace("checkpoint_directory=str(checkpoint/f'global_step_{step}')", "checkpoint_directory=str(checkpoint/f'global_step_{step}') if step is not None else None")
block = block.replace('User requests fresh DTPO v3 restart with variance-only normalization.',
    'Withdraw implementation overreach: linear finite allocation does not preserve the original exponential Q/V strategy-gradient equivalence. User questioned the unapproved removal; preserve evidence and PPO, do not restart.')
block = block.replace('stopped_for_user_requested_fresh_v3', 'stopped_method_equivalence_not_established')
exec(compile(imports, str(owner), 'exec'), globals())
import runpy
old = json.loads((OUT / 'DTPO-v4-job.json').read_bytes())
assert (old['pid'], old['birth']) == (4022617, 1792081757.39)
ppo = json.loads((ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1/native-PPO-control-v9-job.json').read_bytes())
assert (ppo['pid'], ppo['birth']) == (3946004, 1792041120.19)
parser = runpy.run_path(str(ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1/original_plot_training_progress.py'), run_name='native_withdrawal_snapshot')
snapshot = parser['collect'](str(ROOT), [old['run'] + '/textcraft-dt'])
(OUT / 'withdrawal-source-snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
completed = max((m['step'] for j in snapshot['jobs'] for m in j['metrics']), default=0)
exec(compile(block, str(owner), 'exec'), globals())
report = json.loads((OUT / 'DTPO-v4-method-withdrawn.json').read_bytes())
report['completed_iterations_from_original_logs'] = completed
report['reason'] = 'Finite allocation linearity/conservation is not the original exponential Q-V or expected policy-gradient equivalence; do not silently deploy this substitution.'
(OUT / 'DTPO-v4-method-withdrawn.json').write_text(json.dumps(report, indent=2) + '\n')
for name in ['DTPO-v4-job.json']:
    path = OUT / name
    job = json.loads(path.read_bytes())
    job.update(status=report['status'], withdrawal_receipt=str(OUT/'DTPO-v4-method-withdrawn.json'),
               completed_iterations=completed, accepted_for_training=False)
    path.write_text(json.dumps(job, indent=2) + '\n')
run = Path(old['run'])
for name in ['launch.json', 'source.json']:
    path = run / name
    record = json.loads(path.read_bytes())
    record.update(status=report['status'], accepted_for_training=False,
                  withdrawal_receipt=str(OUT/'DTPO-v4-method-withdrawn.json'))
    path.write_text(json.dumps(record, indent=2) + '\n')
for name in ['formal-training.json', 'active-training.json', 'active-source.json']:
    path = ROOT / name
    manifest = json.loads(path.read_bytes())
    jobs = [j for j in manifest['jobs'] if j.get('task') == 'TextCraft' and j.get('method') == 'dt']
    assert len(jobs) == 1 and jobs[0]['pid'] == old['pid']
    jobs[0].update(status=report['status'], accepted_for_training=False,
                   withdrawal_receipt=str(OUT/'DTPO-v4-method-withdrawn.json'))
    path.write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(status=report['status'], survivors=report['survivors'],
    completed_iterations=completed, checkpoint=report['checkpoint_step_preserved'],
    protected=report['protected'], receipt=str(OUT/'DTPO-v4-method-withdrawn.json'))))
