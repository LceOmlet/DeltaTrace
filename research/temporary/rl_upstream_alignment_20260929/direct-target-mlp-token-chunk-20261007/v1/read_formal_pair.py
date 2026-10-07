"""Reuse the original bounded collector with the newly submitted source identity."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
OWNER = HERE.parents[1] / 'direct-target-head-memory-20261007/v3/inspect_active_pair.py'
spec = importlib.util.spec_from_file_location('existing_collector', OWNER)
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)
source = owner.SCRIPT.replace(
    "(670069,1791349552.74,'42bb0eb979f76527f545afd83f934cc46add4c042bf63a16dd3f7304d25789bc')",
    "(2001805,1791362313.39,'24b9e671f3533d88047fead3c709bbfddbf5200053f3cd8f3d308cb30f179f11')",
).replace('@ROOT@', repr(owner.stage.ROOT))
ast.parse(source)
command = owner.stage.ROOT + '/../deltatrace_qwen35_20260912/env/bin/python' + " - <<'PY'\n" + source + "\nPY\n"
result = subprocess.run(owner.stage.SSH + ['bash', '-s'], input=command.encode(), capture_output=True, timeout=65)
result.check_returncode()
observed = json.loads(result.stdout)
observed.update(collector_owner_path=OWNER.as_posix(),
                collector_owner_sha256=hashlib.sha256(OWNER.read_bytes()).hexdigest(),
                exact_read_script_sha256=hashlib.sha256(source.encode()).hexdigest())
output = HERE / ('formal-phase-' + str(int(observed['observed_unix'])) + '.json')
output.write_text(json.dumps(observed, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(receipt=output.as_posix(), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
 observed_unix=observed['observed_unix'], cgroup_GiB=observed['cgroup_usage_bytes']/2**30,
 jobs=[dict(task=j['job']['task'],pid=j['job']['pid'],
            logs=[dict(path=f['path'],selected=[line[:350] for line in f['selected'][-2:]])
                  for f in j['logs'] if f['selected']]) for j in observed['jobs']]), ensure_ascii=False))
