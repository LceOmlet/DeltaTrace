"""Extend the original metadata RPC with existing Adam step counters only."""
import hashlib
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
wrapper = ROOT / 'receipts/dtpo-v9-clip-u-postgroup-variance-20261019-v2/source/inspect_native_workers.py'
assert hashlib.sha256(wrapper.read_bytes()).hexdigest() == '051f8c4c7e1f7814ced95d9ce73b63e6f35ea8e30ef40aaf98cff3adf3ccc66e'
code = wrapper.read_text()
anchor = "exec(compile(code,str(owner),'exec'),globals())"
assert code.count(anchor) == 1
code = code.replace(anchor, '''
counter_anchor = "result['actor_optimizer_entries'] = len(actor.actor_optimizer.state)"
counter_extension = counter_anchor + "\\n    result['actor_optimizer_step_values'] = sorted({float(state['step'].item()) for state in actor.actor_optimizer.state.values() if 'step' in state})"
assert code.count(counter_anchor) == 1
code = code.replace(counter_anchor, counter_extension)
exec(compile(code,str(owner),'exec'),globals())
''')
exec(compile(code, str(wrapper), 'exec'), globals())
