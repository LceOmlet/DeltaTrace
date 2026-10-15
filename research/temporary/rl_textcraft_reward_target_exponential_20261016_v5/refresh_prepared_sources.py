"""Update only the not-deployed candidate after completing its public seam."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/textcraft-reward-target-exponential-20261016-v5'
PREPARED=AUDIT/'v5-prepared'
launch=json.loads((PREPARED/'launch.json').read_bytes())
assert launch['status']=='prepared_only_not_deployed' and not Path(launch['run']).exists()
for path in (AUDIT/'source/candidate-source').glob('*.py'):
    destination=Path(launch['effective_environment']['DT_ENTRY_ROOT'])/path.name
    shutil.copyfile(path,destination)
    launch['source_bindings'][str(destination)]=hashlib.sha256(destination.read_bytes()).hexdigest()
(PREPARED/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
(AUDIT/'prepared-launch.json').write_bytes((PREPARED/'launch.json').read_bytes())
source=json.loads((PREPARED/'source.json').read_bytes());source['actual_inherited_sources']=launch['source_bindings']
(PREPARED/'source.json').write_text(json.dumps(source,indent=2)+'\n')
print('Updated isolated prepared-only source bindings; no formal process change.')
