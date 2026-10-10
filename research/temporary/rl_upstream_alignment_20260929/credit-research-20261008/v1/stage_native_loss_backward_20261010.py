"""Check a passive loss observer on CPU, then queue one original-worker RPC.

The original formal source environment is inherited before Ray imports. A
submission is not reported as installation; observe the same sender handle.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('transport',HERE.parents[1]/'stage_environment_entry.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
OUT = transport.ROOT+'/receipts/textcraft-native-actor-incidents-20261010-v1/loss-backward-v1'
RECEIPT = HERE/'direct-credit-records-20261009-v1/native-loss-backward-stage-20261010.json'


def remote(code):
    command = 'source '+transport.ENTRY+'/metax-entry.env.sh\nCUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
    result = subprocess.run(transport.SSH+['bash','-s'],input=command.encode(),capture_output=True,timeout=55)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    return json.loads(result.stdout)


def main():
    assert not RECEIPT.exists(),'Inspect the existing sender instead of resubmitting.'
    remote('import json\nfrom pathlib import Path\np=Path('+repr(OUT)+')\np.mkdir(parents=True,exist_ok=True)\nprint(json.dumps(dict(path=str(p))))')
    files = {}
    for name in ['capture_native_loss_backward_20261010.py','test_capture_native_loss_backward_20261010.py']:
        path = HERE/name
        subprocess.run(transport.SCP+[str(path),transport.SSH[-1]+':'+OUT+'/'+name],check=True,timeout=30)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    environment = ('source_path=Path('+repr(transport.ROOT+'/runs/textcraft-formal-stable-20261009-v1/source.json')+')\n'
        'assert hashlib.sha256(source_path.read_bytes()).hexdigest()=="1c08b57bf83506d3e69d865f74377e3baa624358c678e0ac92cb6ebfb2d73658"\n'
        'source=json.loads(source_path.read_bytes())\n'
        'env=dict(os.environ,**source["environment"]);env.pop("RAY_ADDRESS",None)\n'
        'env["CUDA_VISIBLE_DEVICES"]="-1";env["PYTHONPATH"]=source["pythonpath"]\n')
    prefix = 'import hashlib,json,os,psutil,subprocess,time\nfrom pathlib import Path\n'+environment+'out=Path('+repr(OUT)+')\n'
    check = remote(prefix+'expected='+repr(files)+'\n'
        'for name,sha in expected.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==sha\n'
        'r=subprocess.run([env["VENV_PYTHON"],str(out/"test_capture_native_loss_backward_20261010.py")],env=env,capture_output=True,text=True,timeout=40)\n'
        'if r.returncode:raise RuntimeError(r.stderr)\n'
        'print(r.stdout)')
    RECEIPT.write_text(json.dumps(dict(status='CPU_delegation_passed_not_installed',files=files,cpu_check=check),indent=2)+'\n')
    query = r'''
import json,os,psutil,time
from pathlib import Path
import ray
import capture_native_loss_backward_20261010 as observer
ray.cloudpickle.register_pickle_by_value(observer)
out=Path(__file__).parent
ray.init(address='127.0.0.1:58837',log_to_driver=False,ignore_reinit_error=False)
named=[x for x in ray.util.list_named_actors(all_namespaces=True)
 if x['name'].startswith('zagXiMWorkerDict_') and 'register_center' not in x['name']]
assert len(named)==2,named
record=dict(pid=os.getpid(),birth=psutil.Process().create_time(),started_unix=time.time(),named_actors=named,complete=False)
path=out/'query-result.json';path.write_text(json.dumps(record,indent=2)+'\n')
try:
    refs=[ray.get_actor(x['name'],namespace=x['namespace']).execute_with_func_generator.remote(observer.install) for x in named]
    record.update(results=ray.get(refs),complete=True,completed_unix=time.time())
    path.write_text(json.dumps(record,indent=2)+'\n')
except Exception as error:
    record.update(error=repr(error),failed_unix=time.time())
    path.write_text(json.dumps(record,indent=2)+'\n')
    raise
finally:
    ray.shutdown()
'''
    commit = subprocess.run(['git','log','-1','--format=%H','--',str(HERE/'capture_native_loss_backward_20261010.py')],capture_output=True,text=True,check=True).stdout.strip()
    assert commit,'Commit the observer source before submitting it.'
    launch = remote(prefix+
        'assert psutil.Process(982372).create_time()==1791553809.84\n'
        'script=out/"install_loss_backward_query.py"\n'
        'assert not script.exists(),"Inspect the existing sender instead of resubmitting."\n'
        'script.write_text('+repr(query)+')\n'
        'with (out/"query.log").open("xb") as stream:\n'
        ' p=subprocess.Popen([env["VENV_PYTHON"],str(script)],env=env,cwd=out,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)\n'
        'd=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),launched_unix=time.time(),path=str(script),'
        'script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),out=str(out),observer_commit='+repr(commit)+',status="submitted_not_verified")\n'
        '(out/"launch.json").write_text(json.dumps(d,indent=2)+"\\n")\n'
        'print(json.dumps(d))')
    data = dict(status='submitted_native_RPC_not_verified',files=files,cpu_check=check,launch=launch,
        model_calls=0,production_numerical_changes=0,scope=__doc__)
    RECEIPT.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(saved=str(RECEIPT),**data)))


if __name__ == '__main__':
    main()
