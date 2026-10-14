"""Read the demonstrated native critic activation-offload assertion and owner."""
import hashlib
import inspect
import json
from pathlib import Path
import time
import verl.utils.activation_offload as owner

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT / 'receipts/textcraft-gradient-rootcause-20261013-v1'
job = json.loads((OUT / 'native-PPO-control-v8-job.json').read_bytes())
run = Path(job['run'])
p = Path(inspect.getsourcefile(owner))
report = {'unix': time.time(), 'scope': __doc__, 'GPU_model_calls': 0,
          'activation_offload': {'path': str(p), 'SHA256': hashlib.sha256(p.read_bytes()).hexdigest(),
                                 'text': p.read_text()},
          'effective_config': (run / 'effective-config.yaml').read_text()}
dest = OUT / 'PPO-activation-offload-owner-v9.json'
dest.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'receipt': str(dest), 'report': report}))
