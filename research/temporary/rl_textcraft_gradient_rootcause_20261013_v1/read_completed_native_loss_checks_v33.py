"""Read completed native loss provenance and unchanged forward-point checks."""
import json
from pathlib import Path
import time

OUT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-gradient-rootcause-20261013-v1')
DEST=OUT/'native-fixed-seed-propagation-v33'
state=json.loads((DEST/'result.json').read_bytes())
assert state['phase']=='complete_native_gradient_chain_localization_not_repair'
report=dict(unix=time.time(),scope=__doc__,ranks=[],GPU_model_calls=0)
for rank in (0,1):
    records={case:json.loads((DEST/f'iteration{case}-rank{rank}-PG.json').read_bytes())
             for case in ('late80_native','late5_native','late5_seed80')}
    rows={k:v['passes']['native_target']['microbatch_losses'] for k,v in records.items()}
    assert all(len(r)==8 for r in rows.values())
    report['ranks'].append(dict(rank=rank,sources=records['late80_native']['sources'],
        all_native_PPO_KL_exact_zero={k:all(r['ppo_kl']==0 for r in v) for k,v in rows.items()},
        all_native_clipping_exact_zero={k:all(r['pg_clipfrac']==0 for r in v) for k,v in rows.items()},
        checkpoint5_native_minus_fixed_seed_entropy=[a['weighted_entropy']['value']-b['weighted_entropy']['value']
            for a,b in zip(rows['late5_native'],rows['late5_seed80'])],
        native_clip_returns={k:v['passes']['native_target']['optimizer_boundary']['native_clip_return'] for k,v in records.items()}))
path=OUT/'native-fixed-seed-loss-checks-v33.json'
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(receipt=str(path),report=report)))
