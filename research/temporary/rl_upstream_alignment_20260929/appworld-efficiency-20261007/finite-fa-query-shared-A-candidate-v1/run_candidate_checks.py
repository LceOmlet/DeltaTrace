"""Execute the saved-operand owner comparison on free GPU2, bounded per rank."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('existing_builder', HERE/'compile_candidate.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
entry, OUT = builder.owner.entry, builder.owner.OUT


def main():
    build = json.loads((HERE/'compiled-owner.json').read_bytes())
    assert build['returncode'] == 0
    sources = {}
    for name in ('check_query_shared_A.py', 'source-preparation.json'):
        if name.endswith('.py'):
            ast.parse((HERE/name).read_text())
        sources[name] = hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        subprocess.run(entry.SCP+[str(HERE/name),entry.SSH[-1]+':'+OUT+'/'+name],check=True)
    code = r'''
import hashlib,json,os,pathlib,psutil,re,subprocess,sys,time
root=pathlib.Path(@ROOT@);out=pathlib.Path(@OUT@)
job=next(j for j in json.loads((root/'active-training.json').read_bytes())['jobs'] if j['task']=='AppWorld')
assert job['pid']==2360541 and psutil.Process(job['pid']).create_time()==1791325655.01
assert hashlib.sha256(pathlib.Path(job['source_receipt']).read_bytes()).hexdigest()=='c83b96debba90d85476e58b56b2c8c7f3900901f26ce41094e8f60ac489c7789'
for name,sha in @SOURCES@.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==sha
physical=subprocess.check_output(['mx-smi'],text=True)
assert not re.search(r'^\|\s+2\s+\d+\s+',physical.split('| Process:')[-1],re.M),'GPU2 occupied; no process changed'
run=out/'saved-operand-checks-v1';run.mkdir(exist_ok=False)
receipt=dict(status='running',started_unix=time.time(),checks=[],physical_before=physical,
    formal_pid=job['pid'],formal_birth=1791325655.01,formal_source_sha256='c83b96debba90d85476e58b56b2c8c7f3900901f26ce41094e8f60ac489c7789',
    source_sha256=@SOURCES@,candidate_library_sha256=@LIBSHA@,production_changes=0,checkpoint_operations=0,model_loads=0)
target=run/'execution.json'
def save():target.write_text(json.dumps(receipt,indent=2)+'\n')
save()
for rank in (0,1):
 command=[sys.executable,'-u',str(out/'check_query_shared_A.py'),'--assets',str(out/'source-preparation.json'),
  '--rank',str(rank),'--candidate-library',str(out/'libfinite_row_query_starts.so'),'--candidate-library-sha256',@LIBSHA@,'--output',str(run/f'rank{rank}.json')]
 record=dict(rank=rank,command=command,started_unix=time.time(),sampled_peak_tree_PSS_bytes=0);receipt['checks'].append(record);save()
 with (run/f'rank{rank}.stdout.txt').open('xb') as log,(run/f'rank{rank}.stderr.txt').open('xb') as err:
  child=subprocess.Popen(command,stdout=log,stderr=err,env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',MACA_VISIBLE_DEVICES='2'),start_new_session=True)
  record.update(pid=child.pid,birth=psutil.Process(child.pid).create_time())
  while child.poll() is None:
   try:
    p=psutil.Process(child.pid);pss=0
    for c in [p,*p.children(recursive=True)]:
     try:pss+=c.memory_full_info().pss
     except psutil.Error:pass
    record['sampled_peak_tree_PSS_bytes']=max(record['sampled_peak_tree_PSS_bytes'],pss)
   except psutil.Error:pass
   if time.time()-record['started_unix']>90:
    child.terminate();child.wait(timeout=10);record['bounded_timeout']=True;break
   time.sleep(.5)
  record.update(returncode=child.returncode,finished_unix=time.time());save()
 if child.returncode:
  receipt['status']='diagnostic_failed_not_deployed';break
else:receipt['status']='saved_operand_comparison_completed_not_deployed'
receipt['finished_unix']=time.time();receipt['physical_after']=subprocess.check_output(['mx-smi'],text=True);save()
print(json.dumps(dict(execution=str(target),receipt=receipt)))
'''.replace('@ROOT@',repr(entry.ROOT)).replace('@OUT@',repr(OUT)).replace('@SOURCES@',repr(sources)).replace('@LIBSHA@',repr(build['library_sha256']))
    compile(code,'<bounded original-owner check>','exec')
    script='source '+entry.ENTRY+'/metax-entry.env.sh\n'+entry.ROOT.replace('/deltatrace_rl_20260922','/deltatrace_qwen35_20260912/env/bin/python')+" - <<'PY'\n"+code+'\nPY\n'
    result=subprocess.run(entry.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=210)
    (HERE/'check-launch.stderr.txt').write_bytes(result.stderr)
    result.check_returncode()
    report=json.loads(result.stdout)
    (HERE/'execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(execution=report['execution'],status=report['receipt']['status'],checks=report['receipt']['checks'])))


if __name__=='__main__':
    main()
