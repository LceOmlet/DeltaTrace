"""Compare actual native case artifacts against the pinned offload replay criterion."""
import json
from pathlib import Path
import statistics
import sys
import time
import torch

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/PPO-32k-native-efficiency-20261016-v1'
reference_name,candidate_name=sys.argv[1:]
assert (reference_name,candidate_name) in [('baseline-fit','candidate-fit'),('baseline-fit','baseline-repeat-fit'),('reference-32k','candidate-32k')]
ref=AUDIT/reference_name;candidate=AUDIT/candidate_name
read=lambda p:json.loads(p.read_bytes())
record=dict(reference=reference_name,candidate=candidate_name,collected_unix=time.time(),
    criterion='Exact memory-path numerical replay; atol=0 and rtol=0, corresponding pinned VERL offload test. No threshold selected from observed errors.',
    compared={},failures=[])

def compare(key,left,right):
    equal=left==right
    record['compared'][key]=equal
    if not equal:record['failures'].append(key)

for rank in (0,1):
    a=read(ref/f'initial-rank{rank}.json');b=read(candidate/f'initial-rank{rank}.json')
    compare(f'rank{rank}.initial_state',a['initial'],b['initial'])
    a=read(ref/f'numerical-rank{rank}.json');b=read(candidate/f'numerical-rank{rank}.json')
    compare(f'rank{rank}.optimizer_calls',a['optimizer_calls'],b['optimizer_calls'])
    for i,(x,y) in enumerate(zip(a['captures'],b['captures'])):
        for name in ('preclip_gradient','native_grad_norm','after_native_step'):
            compare(f'rank{rank}.step{i+1}.{name}',x[name],y[name])
    compare(f'rank{rank}.native_metrics',a['native_metrics'],b['native_metrics'])
    compare(f'rank{rank}.final_native_state',a['final'],b['final'])
    if candidate_name!='baseline-repeat-fit':
        compare(f'rank{rank}.after_timing_state',read(ref/f'final-rank{rank}.json'),read(candidate/f'final-rank{rank}.json'))

left=torch.load(ref/'native-values.pt',map_location='cpu',weights_only=False).batch['values']
right=torch.load(candidate/'native-values.pt',map_location='cpu',weights_only=False).batch['values']
try:
    torch.testing.assert_close(right,left,rtol=0.0,atol=0.0)
    record['compared']['native_values']=True
except AssertionError as exc:
    record['compared']['native_values']=False;record['failures'].append('native_values')
    record['native_values_error']=str(exc)
record['numerical_status']='passed' if not record['failures'] else 'failed'
a=read(ref/'status.json');b=read(candidate/'status.json')
record['cases_completed']=a['status']==b['status']=='completed'
if 'native_warm_update_timings' in a and 'native_warm_update_timings' in b:
    x=[item['seconds'] for item in a['native_warm_update_timings']]
    y=[item['seconds'] for item in b['native_warm_update_timings']]
    record['efficiency']=dict(reference_seconds=x,candidate_seconds=y,reference_median=statistics.median(x),
        candidate_median=statistics.median(y),candidate_over_reference_median=statistics.median(y)/statistics.median(x),
        reference_mean=statistics.mean(x),candidate_mean=statistics.mean(y),
        scope='Same input and successive native states; no tensor hashing/capture during timed calls. Shared host/formal co-tenants remain. No invented performance tolerance.')
record['formal_training_mutations']=0
out=AUDIT/(reference_name+'-vs-'+candidate_name+'.json')
out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
