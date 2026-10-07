"""Bounded original-kernel profiling on free GPU2; formal training is untouched."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('existing_remote_entry', AUDIT / 'stage_environment_entry.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
OUT = entry.ROOT + '/receipts/finite-fa-current-owner-profile-20261007-v1'


def main():
    prior = HERE.parent / 'finite-fa-center-query-fusion-candidate-v1/source-preparation.json'
    old = json.loads(prior.read_bytes())
    base = entry.ROOT + '/candidates/appworld-row-cuts-finite-20261007-v1'
    assets = dict(scope='Only actual saved input provenance reused; no fusion candidate is loaded.',
        prior_asset_metadata=dict(path=str(prior), sha256=hashlib.sha256(prior.read_bytes()).hexdigest()),
        transport_helper=dict(path=base+'/real-b8-v3-official-checks-v1/check_actual_row_finite_fa_offline.py',
            sha256='e32fc0098c5cccb0cb8a5e87150484e407be35c279f5035eb52958c068c6dad0'),
        deployed_wrapper=dict(path=base+'/production-wiring-v1/deltatrace/clean/qwen35/vendor_fa_finite_bf16_d256.py',
            sha256='3e1d61037be22a1cc826b004d49f854e3c246a149642a4f9167c204181f34089'),
        deployed_library=dict(path=base+'/libfinite_row_query_starts.so',
            sha256='4f42c391055afec0a0fee9ee698c0820163ff413e42f1c4909b2961ce81e5157'),
        saved_operands=[dict(rank=x['rank'], finite_operands=x['finite_operands']) for x in old['actual_saved_assets']])
    (HERE/'assets.json').write_text(json.dumps(assets, indent=2)+'\n')
    ast.parse((HERE/'profile_saved_owner.py').read_text())
    create = 'from pathlib import Path; Path('+repr(OUT)+').mkdir(exist_ok=False,parents=True)'
    subprocess.run(entry.SSH+['/opt/conda/bin/python','-c',shlex.quote(create)],check=True)
    sources = {}
    for name in ('profile_saved_owner.py', 'assets.json'):
        sources[name] = hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        subprocess.run(entry.SCP+[str(HERE/name),entry.SSH[-1]+':'+OUT+'/'+name],check=True)
    script = r'''
import hashlib,json,os,pathlib,psutil,re,subprocess,sys,time
root=pathlib.Path(@ROOT@);out=pathlib.Path(@OUT@)
job=next(j for j in json.loads((root/'active-training.json').read_bytes())['jobs'] if j['task']=='AppWorld')
assert job['pid']==2360541 and psutil.Process(job['pid']).create_time()==1791325655.01
assert hashlib.sha256(pathlib.Path(job['source_receipt']).read_bytes()).hexdigest()=='c83b96debba90d85476e58b56b2c8c7f3900901f26ce41094e8f60ac489c7789'
physical=subprocess.check_output(['mx-smi'],text=True)
if re.search(r'^\|\s+2\s+\d+\s+',physical.split('| Process:')[-1],re.M):
    raise RuntimeError('GPU2 has another process; no process was changed.')
for name,sha in @SOURCES@.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==sha
receipt=dict(started_unix=time.time(),formal_pid=job['pid'],formal_birth=1791325655.01,
    formal_source_sha256='c83b96debba90d85476e58b56b2c8c7f3900901f26ce41094e8f60ac489c7789',
    device=2,physical_before=physical,checks=[],source_sha256=@SOURCES@,model_loads=0,
    checkpoint_operations=0,production_changes=0,status='running')
target=out/'execution.json'
def save():target.write_text(json.dumps(receipt,indent=2)+'\n')
save()
for rank in (0,1):
    command=[sys.executable,'-u',str(out/'profile_saved_owner.py'),'--assets',str(out/'assets.json'),
        '--rank',str(rank),'--output',str(out/f'rank{rank}')]
    record=dict(rank=rank,command=command,started_unix=time.time(),sampled_peak_tree_PSS_bytes=0)
    receipt['checks'].append(record);save()
    with (out/f'rank{rank}.stdout.txt').open('xb') as log,(out/f'rank{rank}.stderr.txt').open('xb') as err:
        child=subprocess.Popen(command,stdout=log,stderr=err,env=dict(os.environ,CUDA_VISIBLE_DEVICES='2'),start_new_session=True)
        record.update(pid=child.pid,birth=psutil.Process(child.pid).create_time());save()
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
        receipt['status']='diagnostic_failed_no_production_change';break
else:receipt['status']='original_saved_B4_profiles_completed'
receipt['finished_unix']=time.time();receipt['physical_after']=subprocess.check_output(['mx-smi'],text=True);save()
print(json.dumps(dict(execution=str(target),receipt=receipt)))
'''.replace('@ROOT@',repr(entry.ROOT)).replace('@OUT@',repr(OUT)).replace('@SOURCES@',repr(sources))
    compile(script, '<original-kernel profiling launcher>', 'exec')
    command='source '+entry.ENTRY+'/metax-entry.env.sh\n'+entry.ROOT.replace('/deltatrace_rl_20260922','/deltatrace_qwen35_20260912/env/bin/python')+" - <<'PY'\n"+script+'\nPY\n'
    result=subprocess.run(entry.SSH+['bash','-s'],input=command.encode(),capture_output=True,timeout=210)
    (HERE/'launch.stderr.txt').write_bytes(result.stderr)
    result.check_returncode()
    report=json.loads(result.stdout)
    (HERE/'execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(execution=report['execution'],status=report['receipt']['status'],checks=report['receipt']['checks'])))


if __name__ == '__main__':
    main()
