"""Pin the existing native probe to immutable inputs and a nonzero native LoRA state."""
import hashlib
from pathlib import Path

source = Path(__file__).with_name('probe_native_tp.py')
assert hashlib.sha256(source.read_bytes()).hexdigest() == '249a0225aa2597a0a06e3ca8f4df9e1f2f2f50554136d03274030964135f8b9b'
code = source.read_text()
code = code.replace("OUT = AUDIT/f'TP{TP}'", "OUT = AUDIT/f'TP{TP}-state10-v2'")
code = code.replace("/f'TP{tp}'", "/f'TP{tp}-state10-v2'")
anchor = 'OUT.mkdir(parents=True, exist_ok=True)\n'
assert code.count(anchor) == 1
code = code.replace(anchor, anchor + '''INPUT_SNAPSHOT = OUT/'inputs.snapshot.json'
if not INPUT_SNAPSHOT.exists():
    INPUT_SNAPSHOT.write_bytes((AUDIT/'matched-inputs-v2-all-tail.json').read_bytes())
DIAGNOSTIC_CHECKPOINT = Path('/mnt/si0021787ci2/default/lzq/deltatrace_backups/textcraft-8gpu-20261017/textcraft-dtpo-v6-8gpu-200-20261017-v3/checkpoints/global_step_10/actor')
assert all((DIAGNOSTIC_CHECKPOINT/f'{kind}_world_size_8_rank_{rank}.pt').is_file()
    for rank in range(8) for kind in ('model', 'optim', 'extra_state'))
''')
code = code.replace("hashlib.sha256((AUDIT/'matched-inputs.json').read_bytes()).hexdigest()",
    'hashlib.sha256(INPUT_SNAPSHOT.read_bytes()).hexdigest()')
code = code.replace("cases = json.loads((AUDIT/'matched-inputs.json').read_bytes())['cases']",
    "cases = json.loads(INPUT_SNAPSHOT.read_bytes())['cases']")
anchor = '    handles = trainer.actor_rollout_wg.workers\n'
assert code.count(anchor) == 1
code = code.replace(anchor, '''    trainer.actor_rollout_wg.load_checkpoint(local_path=str(DIAGNOSTIC_CHECKPOINT), del_local_after_load=False)
    record['diagnostic_nonzero_model_state'] = str(DIAGNOSTIC_CHECKPOINT)
    record['formal_resume_or_optimizer_update'] = False
''' + anchor)
code = code.replace("w.actor_lr_scheduler.last_epoch == 0 and len(w.actor_optimizer.state) == 0",
    "w.actor_lr_scheduler.last_epoch == 10 and len(w.actor_optimizer.state) == 496")
code = code.replace("scheduler=0, optimizer_entries=0", "scheduler=w.actor_lr_scheduler.last_epoch, optimizer_entries=len(w.actor_optimizer.state)")
code = code.replace("x['scheduler'] == x['optimizer_entries'] == 0", "x['scheduler'] == 10 and x['optimizer_entries'] == 496")
code = code.replace("f'/tmp/native-rollout-TP{TP}-probe-20261018-v1'", "f'/tmp/dttp{TP}s10v2'")
try:
    exec(compile(code, str(source), 'exec'), globals())
except Exception as error:
    if 'record' in globals() and 'save' in globals():
        record.update(status='failed', error=repr(error))
        save()
    raise
