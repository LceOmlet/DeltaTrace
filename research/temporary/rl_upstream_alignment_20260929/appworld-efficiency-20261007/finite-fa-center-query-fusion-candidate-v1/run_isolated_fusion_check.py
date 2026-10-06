"""Stage one bounded saved-operand check on unused GPU2; never alter training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('existing_entry', AUDIT / 'stage_environment_entry.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
OUT = entry.ROOT + '/candidates/finite-fa-center-query-fusion-candidate-20261007-v1'


def main():
    sources = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
               for name in ('check_fused_fa_offline.py', 'source-preparation.json')}
    for name in sources:
        subprocess.run(entry.SCP + [str(HERE / name), entry.SSH[-1] + ':' + OUT + '/' + name], check=True)
    remote = r'''
import hashlib,json,os,pathlib,psutil,re,subprocess,sys,time
root=pathlib.Path(@ROOT@);out=pathlib.Path(@OUT@)
job=next(j for j in json.loads((root/'active-training.json').read_bytes())['jobs'] if j['task']=='AppWorld')
assert job['pid']==2360541 and psutil.Process(job['pid']).create_time()==1791325655.01
assert hashlib.sha256(pathlib.Path(job['source_receipt']).read_bytes()).hexdigest()=='c83b96debba90d85476e58b56b2c8c7f3900901f26ce41094e8f60ac489c7789'
physical=subprocess.check_output(['mx-smi'],text=True)
if re.search(r'^\|\s+2\s+\d+\s+',physical.split('| Process:')[-1],re.M):
    raise RuntimeError('Physical GPU2 occupied; no process was changed.')
for name,sha in @SOURCES@.items():
    assert hashlib.sha256((out/name).read_bytes()).hexdigest()==sha
prepared=json.loads((out/'source-preparation.json').read_bytes())
run=out/('actual-saved-operands-'+str(time.time_ns()));run.mkdir()
receipt=dict(started_unix=time.time(),physical_gpu_before=physical,checks=[],production_changes=0,
    model_loads=0,checkpoint_operations=0,device=2,source_sha256=@SOURCES@,status='running')
target=run/'execution.json'
def save():target.write_text(json.dumps(receipt,indent=2)+'\n')
save()
base=root/'candidates/appworld-row-cuts-finite-20261007-v1'
for assets in prepared['actual_saved_assets']:
    rank=assets['rank']
    command=[sys.executable,'-u',str(out/'check_fused_fa_offline.py'),
        '--reuse-helper',str(base/'real-b8-v3-official-checks-v1/check_actual_row_finite_fa_offline.py'),
        '--source-preparation',str(out/'source-preparation.json'),
        '--candidate-source',str(out/'vendor_fa_finite_p1_bf16_d256.cu'),
        '--native-operands',assets['native_operands']['path'],
        '--finite-operands',assets['finite_operands']['path'],
        '--saved-verifier',str(root/'releases/c9cd147/experiments/rl/verify_saved_fa_dtypes.py'),
        '--sources',str(root/'receipts/training-setup/official-kernel-tests'),
        '--row-wrapper',str(out/'vendor_fa_finite_bf16_d256.py'),
        '--baseline-library',str(base/'libfinite_row_query_starts.so'),
        '--candidate-library',str(out/'libfinite_row_query_starts.so'),
        '--candidate-library-sha256','60b9a5dfad574817ca13468b28cec18682bd54db604d2e83c70db6513fe216fc',
        '--environment-json',str(root/'candidates/appworld-native-conv-initial-states-20261007-v1/environment.json'),
        '--output-dir',str(run/f'rank{rank}'),'--rank',str(rank)]
    for row in assets['existing_logical_B1_rows']:command+=['--complete-row',row['operands']['path']]
    record=dict(rank=rank,command=command,started_unix=time.time());receipt['checks'].append(record);save()
    with (run/f'rank{rank}.stdout.txt').open('xb') as log,(run/f'rank{rank}.stderr.txt').open('xb') as err:
        child=subprocess.Popen(command,stdout=log,stderr=err,env=dict(os.environ,CUDA_VISIBLE_DEVICES='2'),start_new_session=True)
        record.update(pid=child.pid,birth=psutil.Process(child.pid).create_time(),sampled_peak_tree_PSS_bytes=0);save()
        while child.poll() is None:
            try:
                parent=psutil.Process(child.pid)
                values=[]
                for process in [parent,*parent.children(recursive=True)]:
                    try:values.append(process.memory_full_info().pss)
                    except psutil.Error:pass
                record['sampled_peak_tree_PSS_bytes']=max(record['sampled_peak_tree_PSS_bytes'],sum(values))
            except psutil.Error:pass
            if time.time()-record['started_unix']>90:
                child.terminate();child.wait(timeout=10);record['bounded_timeout']=True;break
            time.sleep(.5)
        record.update(returncode=child.returncode,finished_unix=time.time());save()
    if child.returncode:
        receipt['status']='candidate_failed_not_deployed';break
else:receipt['status']='bounded_saved_operand_checks_completed_not_deployed'
receipt['finished_unix']=time.time();receipt['physical_gpu_after']=subprocess.check_output(['mx-smi'],text=True);save()
print(json.dumps(dict(execution=str(target),receipt=receipt)))
'''.replace('@ROOT@', repr(entry.ROOT)).replace('@OUT@', repr(OUT)).replace('@SOURCES@', repr(sources))
    compile(remote, '<bounded fusion diagnostic>', 'exec')
    script = 'source ' + entry.ENTRY + '/metax-entry.env.sh\n' + entry.ROOT.replace('/deltatrace_rl_20260922', '/deltatrace_qwen35_20260912/env/bin/python') + " - <<'PY'\n" + remote + '\nPY\n'
    result = subprocess.run(entry.SSH + ['bash', '-s'], input=script.encode(), capture_output=True, timeout=210)
    (HERE / 'gpu-stage.stderr.txt').write_bytes(result.stderr)
    result.check_returncode()
    report = json.loads(result.stdout)
    (HERE / 'gpu-stage.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(execution=report['execution'],status=report['receipt']['status'],checks=report['receipt']['checks'])))


if __name__ == '__main__':
    main()
