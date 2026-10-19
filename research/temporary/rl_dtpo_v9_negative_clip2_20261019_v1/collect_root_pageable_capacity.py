"""Read the existing native-capacity receipts; no training or model call."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'rl_dtpo_v9_clip_u_20261019_v1'))
from transport import run_python

CODE = r'''
import hashlib,json,time,psutil
from pathlib import Path
ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v4'
run=ROOT/'runs/textcraft-dtpo-v9-negative-clip2-8gpu-200-20261019-v4'
installation=run/'runtime-overrides/actual-worker-imports-root-pageable-native-capacity-v1.json'
binding=json.loads(installation.read_bytes())
rows=[]
for p in sorted(AUDIT.glob('root-pageable-capacity-rank*.json')):
    d=json.loads(p.read_bytes())
    rows.append(dict(path=str(p),SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),value=d))
completed=[r for r in rows if r['value']['status']=='completed_original_owner_capacity']
failed=[r for r in rows if r['value']['status']=='candidate_capacity_failed']
result=dict(collected=time.time(),run=str(run),installation=str(installation),
    installation_SHA256=hashlib.sha256(installation.read_bytes()).hexdigest(),
    rows=rows,completed_ranks=[r['value']['rank'] for r in completed],
    failed_ranks=[r['value']['rank'] for r in failed],
    candidate_deployed=False,
    status='failed' if failed else 'completed' if len(completed)==8 else 'pending',
    capacity_scope='Original B4/exact32768 on ranks0/1, small original B4 on ranks2..7; not simultaneous eight-rank all-32k capacity',
    numerical_scope='Original owner interface and finite outputs only, not a new official numerical tolerance test',
    additional_production_model_calls=0,additional_optimizer_calls=0,
    excluded_validation_intervals=[dict(rank=r['value']['rank'],start=r['value']['started'],end=r['value'].get('finished')) for r in rows])
if len(completed)==8:
    assert sorted(result['completed_ranks'])==list(range(8))
    for r in completed:
        d=r['value']
        assert d['original_options_restored'] and d['original_method_restored']
        assert all(d['all_QVA_finite'].values())
        if d['rank'] in (0,1):
            assert d['input_shape']==[4,32768]
            assert d['native_report']['actual_context_lengths']==[32768]*4
            assert d['native_report']['minibatch_size']==4
    receipt=AUDIT/'root-pageable-native-capacity-results.json'
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    result['remote_receipt']=str(receipt)
print(json.dumps(result))
'''

def main():
    result=json.loads(run_python(CODE,timeout=45))
    path=HERE.parents[2]/'experiments/rl/results_dtpo_v9_negative2_root_capacity_20261019_v4.json'
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    compact=[]
    for row in result['rows']:
        d=row['value']
        compact.append({k:d.get(k) for k in ('rank','status','input_shape','candidate_seconds','error','RNG_unchanged','original_options_restored','original_method_restored')})
    print(json.dumps(dict(collected=result['collected'],status=result['status'],rows=compact,local_result=str(path)),indent=2))

if __name__=='__main__':
    main()
