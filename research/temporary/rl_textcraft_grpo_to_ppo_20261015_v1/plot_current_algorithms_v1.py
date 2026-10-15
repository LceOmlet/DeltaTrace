"""Keep the user's stopped GRPO history in the existing three-algorithm plot."""
from pathlib import Path
import hashlib

source = Path(__file__).resolve().parent.parent/'rl_textcraft_gradient_rootcause_20261013_v1/plot_current_approved_DTPO_v63.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='f6ed1ff351b9f8ea7980577d4897acd59e953bc105defe47fe4cb3e700d46244'
code = source.read_text()
old = "assert all(j['pid_alive'] for j in snapshot['jobs'])"
new = """grpo_stop_path=ROOT/'receipts/textcraft-grpo-to-ppo-20261015-v1/GRPO-user-stopped.json'
grpo_stop=json.loads(grpo_stop_path.read_bytes())
assert grpo_stop['status']=='stopped_GRPO_at_user_request_checkpoints_and_logs_preserved' and not grpo_stop['survivors']
for j in snapshot['jobs']:
    if j['method']=='grpo':
        assert (j['pid'],j['observed_process_created_unix'])==(grpo_stop['job']['pid'],grpo_stop['job']['birth'])
        assert not j['pid_alive']
        j['training_status']='stopped_by_user_historical_control'
    else:
        assert j['pid_alive']
snapshot['GRPO_user_stop']=dict(receipt=str(grpo_stop_path),completed_checkpoint=grpo_stop['completed_iteration'])"""
assert old in code
code = code.replace(old, new)
old = "label='原生 GRPO'"
assert old in code
code = code.replace(old, "label='原生 GRPO（已停止）'")
old = "    fig.text(.075,.942,subtitle,color='#475569',fontsize=9.5)"
assert old in code
code = code.replace(old, "    subtitle+=' · GRPO已按用户要求停止，曲线保留'\n"+old)
exec(compile(code, str(source), 'exec'), globals())
