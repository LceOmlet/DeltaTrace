"""Retry the unchanged full-response numerical replay after capacity ends.

The failed parallel execution was killed by kernel global OOM on allowed
NUMA node0. Change only scheduling, audit path and the Ray session directory.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
owner = ROOT/'receipts/PPO-short-numeric-reference-20261016-v1/execution2/source/run_reference.py'
assert hashlib.sha256(owner.read_bytes()).hexdigest() == 'b7a631d6ca69da90157a590050c7751aa6785941b8575c12a8c8cded3311bd4d'
capacity = ROOT/'receipts/PPO-32k-colocated-capacity-20261016-v1/execution3/status.json'
assert json.loads(capacity.read_bytes())['status'] == 'completed_native_colocated_capacity'
code = owner.read_text()
old = "AUDIT = ROOT/'receipts/PPO-short-numeric-reference-20261016-v1/execution2'"
new = "AUDIT = ROOT/'receipts/PPO-short-numeric-reference-20261016-v1/execution3'"
assert code.count(old) == 1 and code.count("'/tmp/ppo8num2'") == 1
code = code.replace(old,new).replace("'/tmp/ppo8num2'","'/tmp/ppo8num3'")
assert code.replace(new,old).replace("'/tmp/ppo8num3'","'/tmp/ppo8num2'") == owner.read_text()
exec(compile(code,str(owner),'exec'),globals())
