"""Run all 48 answer-only cases with exact old-development overlap controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('environment','control','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();plan=json.loads((HERE/'protocol.json').read_bytes())
    identity=dict(plan_sha256=sha(HERE/'protocol.json'),driver_sha256=sha(HERE/'evaluate_answer.py'),
                  controller_sha256=sha(Path(__file__)),environment_sha256=sha(a.environment))
    assert not a.output.exists();a.output.mkdir(parents=True)
    state=dict(status='running',identity=identity,completed=[],case_count=0,controls=0,current=None)
    def save():
        p=a.output/'progress.partial';p.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');p.replace(a.output/'progress.json')
    try:
        for name,indices in plan['chunks'].items():
            state['current']=name;save();folder=a.output/name
            with (a.output/(name+'.log')).open('wb') as log:
                result=subprocess.run([sys.executable,'-u',str(HERE/'evaluate_answer.py'),'--environment',str(a.environment.resolve()),
                    '--control',str(a.control.resolve()),'--stage','fairness','--datasets','hotpotqa_long',
                    '--chunk',name,'--output',str(folder.resolve())],env=dict(os.environ,MACA_PATH='/opt/maca'),stdout=log,stderr=subprocess.STDOUT)
            assert result.returncode==0,f'Failed {name}; preserve failed files'
            r=json.loads((folder/'results.json').read_bytes())
            assert r['status']=='complete' and r['fair_plan_sha256']==identity['plan_sha256'] and r['driver_sha256']==identity['driver_sha256']
            assert len(r['cases'])==len(indices) and {x['index'] for x in r['cases']}==set(indices)
            assert r['vectors_sha256']==sha(folder/'vectors.npz')
            controls=sum(x.get('development_overlap_bitwise_equal',False) for x in r['cases'])
            state['completed'].append(dict(chunk=name,results_sha256=sha(folder/'results.json'),vectors_sha256=r['vectors_sha256'],cases=len(indices),controls=controls))
            state['case_count']+=len(indices);state['controls']+=controls;save()
        assert state['case_count']==48 and state['controls']==8
        state.update(status='complete',current=None)
    except BaseException as e:
        state.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:save()


if __name__=='__main__':main()
