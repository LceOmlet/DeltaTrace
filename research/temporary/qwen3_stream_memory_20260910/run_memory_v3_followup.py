from pathlib import Path
import hashlib,json,os,subprocess,time,traceback
T=Path(__file__).resolve().parent;state=T/'memory_v3_followup.json';assert not state.exists()
summary_path=T/'memory_v3_confirmation_summary.json';summary=json.loads(summary_path.read_bytes())
assert summary['acceptance_passed'] and summary['all_timed_rows_count']==120 and len(summary['verified_cells'])==24
assert json.loads((T/'memory_v3_confirmation/queue.json').read_bytes())['status']=='complete'
assert not (T/'memory_v3_author').exists() and not (T/'memory_v3_rollout').exists()
report={'status':'running','started':time.time(),'jobs':[],'verified_confirmation_sha256':hashlib.sha256(summary_path.read_bytes()).hexdigest()}
def save():state.write_text(json.dumps(report,indent=2)+'\n')
save()
try:
    env=dict(os.environ,MACA_PATH='/opt/maca',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    job={'name':'author','started':time.time(),'status':'running'};report['jobs'].append(job);save()
    with (T/'memory_v3_author.log').open('w') as log:
        c=subprocess.run(['/mnt/geogpt-doc-new/deepresearch/lzq/contrastive_flashtrace_probe/env/bin/python','-u',str(T/'check_memory_v3_author.py'),'--root',str(T)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,timeout=5400)
    result=json.loads((T/'memory_v3_author/results.json').read_bytes());job.update(status=result['status'],returncode=c.returncode,ended=time.time());save()
    assert c.returncode==0 and result['status']=='complete' and result['all_vectors_exact']
    subprocess.run(['/opt/conda/bin/python',str(T/'collect_memory_v3.py'),'--phase','author'],check=True)
    job={'name':'rollout','started':time.time(),'status':'running'};report['jobs'].append(job);save()
    with (T/'memory_v3_rollout_controller.log').open('w') as log:
        c=subprocess.run(['/opt/conda/bin/python','-u',str(T/'run_memory_v3_rollout.py')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,timeout=43200)
    result=json.loads((T/'memory_v3_rollout/queue.json').read_bytes());job.update(status=result['status'],returncode=c.returncode,ended=time.time());save()
    assert c.returncode==0 and result['status']=='complete' and len(result['jobs'])==21
    subprocess.run(['/opt/conda/bin/python',str(T/'collect_memory_v3.py'),'--phase','rollout'],check=True)
    report['status']='complete'
except BaseException:report['status']='failed';report['error']=traceback.format_exc()
finally:report['ended']=time.time();save();print(json.dumps(report),flush=True)
