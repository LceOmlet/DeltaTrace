"""Fetch only completed native identity receipts, never model or seed tensors."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('strict', HERE.parent / 'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
code = """import json,pathlib,hashlib
out=pathlib.Path(%r)/'receipts/textcraft-gradient-rootcause-20261013-v1/native-full-policy-direction-exchange-v86'
result={}
for step in (40,80):
 p=out/f'captured-full{step}-seed-bindings.json'
 result[str(step)]=dict(path=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),value=json.loads(p.read_bytes()))
print(json.dumps(result))
""" % entry.ROOT
r = subprocess.run(entry.SSH + ['/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python', '-'], input=code.encode(), capture_output=True, check=True)
d = json.loads(r.stdout)
dest = HERE / 'completed-full-policy-seed-bindings-v89.json'
dest.write_bytes(r.stdout)
print(json.dumps(dict(path=str(dest), receipts=[dict(step=k, path=v['path'], SHA256=v['SHA256']) for k, v in d.items()])))
