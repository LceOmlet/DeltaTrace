"""Read-only startup receipt for the observed MetaX/Ray device-count mismatch."""
import json
import os
from pathlib import Path


def install():
    path = os.environ.get('DT_WORKER_VISIBILITY_DIR')
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)
        # Do not import torch or initialize CUDA in a worker startup hook.
        result = dict(pid=os.getpid(), visible={key: value for key, value in os.environ.items()
            if 'VISIBLE_DEVICES' in key}, rank=os.environ.get('RANK'))
        Path(path, f'{os.getpid()}.json').write_text(json.dumps(result))
