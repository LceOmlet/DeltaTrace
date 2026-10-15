"""Read the current original PPO restore phase, without model/RPC calls."""
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('strict', HERE.parent / 'rl_upstream_alignment_20260929/stage_environment_entry.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
code = """import json,pathlib,psutil,time
o=pathlib.Path(%r)/'receipts/textcraft-gradient-rootcause-20261013-v1'
j=json.loads((o/'native-PPO-resume-after-saved-update-v90-v91-job.json').read_bytes())
p=psutil.Process(j['pid']);assert p.create_time()==j['birth'] and p.status()!=psutil.STATUS_ZOMBIE
c=o/'PPO-restore-binding-v91-client.json';client=json.loads(c.read_bytes()) if c.exists() else None
result=None if client is None else pathlib.Path(client['result'])
binding=json.loads(result.read_bytes()) if result is not None and result.exists() else None
workers=[dict(pid=x.pid,birth=x.create_time(),phase=x.name()) for x in p.children(recursive=True) if x.name().startswith('ray::WorkerDict')]
err=pathlib.Path(j['stderr'])
print(json.dumps(dict(unix=time.time(),job=dict(pid=j['pid'],birth=j['birth'],checkpoint=j['resumed_completed_iteration']),workers=workers,client=client,binding=binding,stderr_tail=err.read_text(errors='replace').splitlines()[-8:])))
""" % entry.ROOT
r = subprocess.run(entry.SSH + ['/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python', '-'], input=code.encode(), capture_output=True, check=True)
d = json.loads(r.stdout)
(HERE / 'PPO-restore-v91-progress-latest.json').write_bytes(r.stdout)
print(json.dumps(dict(unix=d['unix'],job=d['job'],workers=d['workers'],client_present=d['client'] is not None,binding_present=d['binding'] is not None,stderr_tail=d['stderr_tail']),ensure_ascii=False))
