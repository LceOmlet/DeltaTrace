"""Collect owner-emitted verification and the previous stopped run facts."""
import hashlib
import json
from pathlib import Path
import time
import psutil
from prepare_sources import ROOT,AUDIT

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
            for binding in record['bindings'].values():assert sha(binding['path'])==binding['SHA256']
            reports.append(dict(path=str(path),SHA256=sha(path),rank=rank,stage=stage,
                seconds=record['seconds'],finite_calls=record['finite_calls'],credit_calls=record['credit_calls'],
                bindings=record['bindings'],optimizer_updates=0))
    summary=dict(status='passed_native_v9_DT_and_exact32768_B4_per_rank',collected=time.time(),
        physical_gpus=[2,3],ranks=2,reports=reports,optimizer_updates=0,generated_rollouts=0,quality_evaluations=0,
        scope='Actual DT interface and two-rank B4/32768 capacity; not eight-rank full training or quality acceptance',
        original_allocator_unchanged=True,native_status=str(AUDIT/'native-behavior-status.json'))
    (AUDIT/'native-verification-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    parent=ROOT/'receipts/textcraft-v5-eight-gpu-20261019-v1'
    old=json.loads((parent/'campaign-state.json').read_bytes());job=old['jobs'][0]
    prior=dict(observed=time.time(),status=old['status'],version=job['version'],pid=job['pid'],birth=job['birth'],
        alive=psutil.pid_exists(job['pid']) and psutil.Process(job['pid']).create_time()==job['birth']
            and psutil.Process(job['pid']).status()!=psutil.STATUS_ZOMBIE,
        completed_iterations=job['completed_iterations'],exit_code=(Path(job['run'])/'exit-code.txt').read_text().strip(),
        stderr=job['stderr'],stderr_SHA256=sha(job['stderr']),stopping_signal_sent=False,
        failure='CUDA invalid argument during original native capture in first DT stage; cause unestablished; native v9 capacity passed unchanged runtime')
    (AUDIT/'prior-v5-observation.json').write_text(json.dumps(prior,indent=2)+'\n')
    print(json.dumps(dict(native_status=summary['status'],native_reports=len(reports),prior_v5=prior)))

if __name__=='__main__':main()
