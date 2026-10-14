"""Read original first rollout dump scores, without model or reward recomputation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
report=dict(unix=time.time(),runs={},model_calls=0,reward_calls=0)
for label,name in [('old','textcraft-dt-formal-native-graph-fresh-20261012-v1'),
                   ('new','textcraft-dt-grpo-multiplier-no-whiten-20261014-v1')]:
    run=ROOT/'runs'/name;launch=json.loads((run/'launch.json').read_bytes())
    directory=Path(launch['options']['trainer.rollout_data_dir'])
    paths=sorted(directory.glob('*'))
    item=dict(directory=str(directory),first_files=[str(p) for p in paths[:5]])
    path=directory/'1.jsonl'
    if path.exists():
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        item.update(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
            rows=len(rows),first_row_keys=list(rows[0]) if rows else [],
            score_counts=dict(Counter(str(row.get('score')) for row in rows)),
            first_row_metadata={k:v for k,v in rows[0].items() if k not in ('input','output','prompt','response')} if rows else {})
    report['runs'][label]=item
dest=OUT/'native-first-rollout-metadata-v52.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(dest),report=report)))
