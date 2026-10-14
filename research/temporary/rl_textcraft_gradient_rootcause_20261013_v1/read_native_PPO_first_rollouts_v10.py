"""Compare the two existing first-iteration rollout dumps, without reconstruction."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=ROOT/'receipts/textcraft-gradient-rootcause-20261013-v1'
paths={}
records={}
for name,run in [('DTPO','textcraft-dt-formal-native-graph-fresh-20261012-v1'),
                 ('PPO','textcraft-ppo-control-fixed-verl-fresh-20261014-v2')]:
    launch=json.loads((ROOT/'runs'/run/'launch.json').read_bytes())
    path=Path(launch['options']['trainer.rollout_data_dir'])/'1.jsonl'
    data=path.read_bytes();records[name]=[json.loads(line) for line in data.splitlines()]
    paths[name]={'path':str(path),'SHA256':hashlib.sha256(data).hexdigest(),'bytes':len(data),
                 'rows':len(records[name]),'row_keys':list(records[name][0])}
assert len(records['DTPO'])==len(records['PPO'])==256
fields={key:all(a[key]==b[key] for a,b in zip(records['DTPO'],records['PPO']))
        for key in sorted(set(records['DTPO'][0])&set(records['PPO'][0]))}
report={'unix':time.time(),'scope':__doc__,'new_model_or_training_calls':0,'files':paths,
        'fields_equal_in_original_row_order':fields,
        'limits':['Equality here concerns original saved decoded text/reward artifacts, not a claim that every hidden native tensor or raw sampled ID was captured and compared.',
                  'This is an initial training-input comparison, not validation-quality measurement or repair acceptance.']}
dest=OUT/'PPO-v2-first-rollout-comparison-v10.json';dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'receipt':str(dest),'report':report}))
