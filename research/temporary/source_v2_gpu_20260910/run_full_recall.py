"""Complete deterministic Recall shards, with verified resume and overlap checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('environment','parent','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();plan=json.loads((HERE/'full_recall_plan.json').read_bytes())
    assert sha(a.parent/'results.json')==plan['parent_results_sha256']
    assert sha(a.parent/'vectors.npz')==plan['parent_vectors_sha256']
    fingerprint=dict(plan_sha256=sha(HERE/'full_recall_plan.json'),spec_sha256=sha(HERE/'FULL_RECALL.md'),
        environment_sha256=sha(a.environment),driver_sha256=sha(HERE/'evaluate_full_recall.py'),controller_sha256=sha(Path(__file__)),
        sources={n:sha(HERE/n) for n in ('weighted_secant.py','weighted_paired.py','scope_choice.json')})
    a.output.mkdir(parents=True,exist_ok=True)
    identity=a.output/'execution_identity.json'
    if identity.exists():assert json.loads(identity.read_bytes())==fingerprint,'Execution identity changed'
    else:identity.write_text(json.dumps(fingerprint,indent=2)+'\n',encoding='utf-8')
    state=dict(status='running',identity=fingerprint,unique_target=448,reused_cases=80,new_completed=0,
        control_completed=0,completed_chunks=[],current_chunk=None,child_pid=None)
    def save():
        temp=a.output/'progress.partial';temp.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8');temp.replace(a.output/'progress.json')
    env=dict(os.environ,MACA_PATH='/opt/maca')
    try:
        for name,chunk in plan['chunks'].items():
            state.update(current_chunk=name,child_pid=None);save()
            folder=a.output/name
            if not folder.exists():
                command=[sys.executable,'-u',str(HERE/'evaluate_full_recall.py'),'--environment',str(a.environment.resolve()),
                    '--parent',str(a.parent.resolve()),'--choice',str(HERE/'scope_choice.json'),'--stage','full',
                    '--datasets',chunk['dataset'],'--chunk',name,'--output',str(folder.resolve())]
                with (a.output/(name+'.log')).open('wb') as log:
                    proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    state['child_pid']=proc.pid;save();exit_code=proc.wait()
                assert exit_code==0,f'Failed shard {name}; preserve its output (exit {exit_code})'
            r=json.loads((folder/'results.json').read_bytes())
            assert r['status']=='complete' and r['experiment']=='target-full-v1'
            assert r['driver_sha256']==fingerprint['driver_sha256'] and r['full_plan_sha256']==fingerprint['plan_sha256']
            assert r['chunk_spec']==chunk and r['parent_results_sha256']==plan['parent_results_sha256']
            assert r['vectors_sha256']==sha(folder/'vectors.npz')
            assert len(r['cases'])==len(chunk['indices'])
            assert {x['index'] for x in r['cases']}==set(chunk['indices'])
            assert all(x['dataset']==chunk['dataset'] and x['status']=='complete' for x in r['cases'])
            assert all(x['overlap_control_bitwise_equal'] for x in r['cases'] if x['index'] in chunk['control_indices'])
            receipt=dict(chunk=name,dataset=chunk['dataset'],new_cases=len(chunk['new_indices']),controls=len(chunk['control_indices']),
                results_sha256=sha(folder/'results.json'),vectors_sha256=r['vectors_sha256'])
            state['completed_chunks'].append(receipt);state['new_completed']+=receipt['new_cases'];state['control_completed']+=receipt['controls'];state['child_pid']=None
            save();print(json.dumps(receipt),flush=True)
        assert state['new_completed']==368 and state['control_completed']==5
        state.update(status='complete',current_chunk=None,child_pid=None)
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:save()


if __name__=='__main__':main()
