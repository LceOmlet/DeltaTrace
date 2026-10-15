"""Reuse original read-only initial state RPC and AST-identical passive logging."""
from pathlib import Path

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
source=ROOT/'receipts/textcraft-weighted-reward-targets-20261015-v4/source/verify_DTPO_v4_workers.py'
code=source.read_text()
code=code.replace('textcraft-weighted-reward-targets-20261015-v4','textcraft-reward-target-exponential-20261016-v5')
code=code.replace('DTPO-v4-job.json','DTPO-v5-job.json')
code=code.replace('initial-state-and-existing-passive-records-v4.json','initial-state-and-existing-passive-records-v5.json')
code=code.replace('install_passive_records_v4.py','install_passive_records_v5.py')
code=code.replace('weighted_reward_target_finite_v4','reward_target_exponential_v5')
code=code.replace('read_v4_environment','read_v5_environment').replace('actual_v4_environment','actual_v5_environment')
exec(compile(code,str(source),'exec'),globals())
