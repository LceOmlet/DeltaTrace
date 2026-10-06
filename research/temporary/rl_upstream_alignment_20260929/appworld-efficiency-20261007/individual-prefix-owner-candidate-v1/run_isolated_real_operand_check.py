"""Run the bounded real-operand check on a currently free physical GPU2."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('entry',HERE.parents[1]/'stage_environment_entry.py')
entry=importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
REMOTE=entry.ROOT+'/candidates/appworld-row-cuts-finite-20261007-v1'


def main():
    source=HERE/'verify_row_layout_real_operands.py'
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    subprocess.run(entry.SCP+[str(source),entry.SSH[-1]+':'+REMOTE+'/'+source.name],check=True)
    code=r'''
import os,pathlib,json,time,subprocess,hashlib,psutil,re,signal
candidate=pathlib.Path(@REMOTE@)
source=candidate/'verify_row_layout_real_operands.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()==@SHA@
out=candidate/'real-operands-v1'
if out.exists():
    raise FileExistsError('Preserve the existing real-operands attempt; inspect it instead of repeating the test')
physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
processes=physical.split('| Process:')[-1]
assert not re.search(r'\|\s*2\s+\d+\s',processes),'Physical GPU2 has an existing process'
out.mkdir()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',MACA_VISIBLE_DEVICES='2')
cmd=[os.environ['VENV_PYTHON'],'-u',str(source),'--out',str(out)]
start=time.time();peak_pss=0;peak_children=[];samples=[];next_gpu=start;timed_out=False
with (out/'stdout.txt').open('wb') as stdout,(out/'stderr.txt').open('wb') as stderr:
    p=subprocess.Popen(cmd,env=env,cwd=candidate,stdout=stdout,stderr=stderr,start_new_session=True)
    birth=psutil.Process(p.pid).create_time()
    while p.poll() is None:
        children=[]
        try:
            parent=psutil.Process(p.pid)
            for c in [parent,*parent.children(recursive=True)]:
                try:children.append(dict(pid=c.pid,pss_bytes=c.memory_full_info().pss))
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
        except psutil.NoSuchProcess:pass
        size=sum(v['pss_bytes'] for v in children)
        if size>peak_pss:peak_pss=size;peak_children=children
        if time.time()>=next_gpu:
            samples.append(dict(unix=time.time(),physical_mx_smi=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout))
            next_gpu=time.time()+1
        if time.time()-start>120:
            timed_out=True
            assert psutil.Process(p.pid).create_time()==birth and os.getpgid(p.pid)==p.pid
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
            break
        time.sleep(.2)
report=dict(scope='Bounded original saved-operand FA assertions plus real finite layout transport; no model, RL, optimizer, rollout, checkpoint or production change',
    command=cmd,source_sha256=@SHA@,started_unix=start,finished_unix=time.time(),pid=p.pid,birth=birth,returncode=p.returncode,
    timed_out=timed_out,physical_GPU=2,sampled_peak_tree_PSS_bytes=peak_pss,peak_processes=peak_children,physical_before=physical,physical_samples=samples,
    physical_after=subprocess.run(['mx-smi'],capture_output=True,text=True).stdout,
    stdout_tail=(out/'stdout.txt').read_text(errors='replace')[-5000:],stderr=(out/'stderr.txt').read_text(errors='replace'))
for name in ['original-official-saved-operand-check.json','real-layout-result.json']:
    q=out/name
    if q.exists():report[name]=dict(path=str(q),sha256=hashlib.sha256(q.read_bytes()).hexdigest(),value=json.loads(q.read_text()))
(out/'execution.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
'''.replace('@REMOTE@',repr(REMOTE)).replace('@SHA@',repr(sha))
    compile(code,'<bounded real-operand launch>','exec')
    script='source '+entry.ENTRY+'/metax-entry.env.sh\n/opt/conda/bin/python - <<\'PY\'\n'+code+'\nPY\n'
    run=subprocess.run(entry.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=145)
    (HERE/'real-operands-launch.stderr.txt').write_bytes(run.stderr)
    run.check_returncode()
    (HERE/'real-operands-execution.json').write_bytes(run.stdout)
    report=json.loads(run.stdout)
    for name in ['original-official-saved-operand-check.json','real-layout-result.json']:
        if name in report:(HERE/name).write_text(json.dumps(report[name]['value'],indent=2)+'\n',encoding='utf8')
    summary={key:report[key] for key in ['returncode','timed_out','started_unix','finished_unix','sampled_peak_tree_PSS_bytes']}
    if report['returncode']:
        summary['stderr']=report['stderr'];summary['stdout_tail']=report['stdout_tail']
    if 'real-layout-result.json' in report:
        value=report['real-layout-result.json']['value']
        summary.update(official_status=value['original_official_coincident_check_status'],same_actual_range=value['same_actual_complete_range'],mixed_index_transport=value['mixed_index_transport'],resource=value['resource'])
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
