"""Collect the original native harness reports; no extra model calls or tests."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
AUDIT=ROOT/'receipts/dtpo-v9-negative-clip2-postgroup-variance-20261019-v3'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    state=json.loads((AUDIT/'native-behavior-status.json').read_bytes())
    assert state['status']=='completed'
    reports=[]
    for stage in ('native-small','native-multi-target','capacity-multi-32768'):
        for rank in (0,1):
            path=AUDIT/f'{stage}-rank{rank}.json';record=json.loads(path.read_bytes())
            assert record['status']=='completed' and record['before']==record['after']
            assert record['report']['credit_transform']=='per_event_clipped_exponential_then_credit_sum'
            assert record['report']['event_u_clip']==[-2.0,1.0]
            for binding in record['bindings'].values():assert sha(binding['path'])==binding['SHA256']
            reports.append(dict(path=str(path),SHA256=sha(path),rank=rank,stage=stage,
                seconds=record['seconds'],finite_calls=record['finite_calls'],credit_calls=record['credit_calls'],
                bindings=record['bindings'],optimizer_updates=0))
    summary=dict(status='passed_native_v9_DT_and_exact32768_B4_per_rank',collected=time.time(),
        physical_gpus=[2,3],ranks=2,event_u_clip=[-2,1],reports=reports,
        optimizer_updates=0,generated_rollouts=0,quality_evaluations=0,
        scope='Actual negative2 DT interface and two-rank B4/32768 capacity; not eight-rank full training or quality acceptance',
        native_status=str(AUDIT/'native-behavior-status.json'))
    (AUDIT/'native-verification-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(status=summary['status'],native_reports=len(reports),event_u_clip=[-2,1])))

if __name__=='__main__':main()
