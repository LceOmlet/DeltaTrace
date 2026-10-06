
import hashlib, json, os, resource, site, sys, time, traceback
from pathlib import Path
import psutil

out = Path(os.environ['DT_LOOP_READINESS_CHECK_OUT'])
root = Path(os.environ['LOOP_ROOT'])
site.addsitedir(os.environ['LOOP_EXTRAS'])
from phi_agents.appworld import interface, server
import pytest

def identity(path):
    p=Path(path);b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))

def snapshot(label):
    p=psutil.Process()
    cg=Path('/sys/fs/cgroup/memory/memory.usage_in_bytes')
    lim=Path('/sys/fs/cgroup/memory/memory.limit_in_bytes')
    return dict(label=label,observed_unix=time.time(),pid=p.pid,pid_birth=p.create_time(),rss_bytes=p.memory_info().rss,
                max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                host_available_bytes=psutil.virtual_memory().available,
                cgroup_actual_bytes=int(cg.read_text()) if cg.exists() else None,
                cgroup_limit_bytes=int(lim.read_text()) if lim.exists() else None,
                scope='host/cgroup shared; RSS is this CPU checker only')

def describe_service(world, label):
    child=psutil.Process(world.server.pid)
    return dict(label=label,observed_unix=time.time(),pid=child.pid,pid_birth=child.create_time(),url=world.remote_environment_url,
                port=world.port,argv=child.cmdline(),rss_bytes=child.memory_info().rss,
                readiness_task_id=interface.DUMMY_TASK_ID,
                readiness_task_id_source='original _wait_for_server_ready JSON assertion passed; no task initialized',
                owner_task_id=world._task_id,timeout_seconds=world.timeout_seconds,
                max_wait_tries=world._max_wait_tries,wait_seconds=world._wait_seconds,
                max_restarts_on_error=world._max_restarts_on_error)

report=dict(schema='original-loop-isolated-readiness-cpu-service-v1',phase='starting',started_unix=time.time(),
            operation_counts=dict(model_forwards=0,GPU_computation=0,training=0,environment_actions=0),
            source_commit=os.environ['DT_LOOP_READINESS_SOURCE_COMMIT'],
            actual_imports=dict(interface=identity(interface.__file__),server=identity(server.__file__),
                                patch=identity(out/'patch_loop_readiness.py'),test=identity(out/'test_loop_readiness.py')),
            environment=dict(VENV_PYTHON=sys.executable,LOOP_ROOT=str(root),LOOP_EXTRAS=os.environ['LOOP_EXTRAS'],
                             APPWORLD_ROOT=os.environ['APPWORLD_ROOT'],CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                             MACA_VISIBLE_DEVICES=os.environ.get('MACA_VISIBLE_DEVICES')),
            resources=[snapshot('before_pytest')],services=[],cleanup=[])
assert Path(interface.__file__).resolve()==(root/'phi_agents/appworld/interface.py').resolve()
assert Path(server.__file__).resolve()==(root/'phi_agents/appworld/server.py').resolve()
spawned=[]
original_launch=interface.launch_appworld_environment_server

def observe_launch(*args,**kwargs):
    value=original_launch(*args,**kwargs)
    spawned.append(value)
    print(json.dumps(dict(phase='original_service_spawned',pid=value.pid,observed_unix=time.time())),flush=True)
    return value

world=None
interface.launch_appworld_environment_server=observe_launch
try:
    print(json.dumps(dict(phase='actual_owner_pytest',observed_unix=time.time())),flush=True)
    code=pytest.main(['-q',str(out/'test_loop_readiness.py'),'--junitxml='+str(out/'pytest-results.xml')])
    report['pytest_exit_code']=int(code)
    assert code==0, 'Original candidate-owner pytest failed; native service check not executed'
    report['resources'].append(snapshot('after_pytest'))
    start=time.perf_counter()
    print(json.dumps(dict(phase='original_init',observed_unix=time.time())),flush=True)
    world=interface.AppWorldInterface(stdout_to_devnull=False)
    init=describe_service(world,'init');init['seconds']=time.perf_counter()-start
    report['services'].append(init)
    first=world.server
    start=time.perf_counter()
    print(json.dumps(dict(phase='original_restart',observed_unix=time.time())),flush=True)
    world.restart()
    restart=describe_service(world,'restart');restart['seconds']=time.perf_counter()-start
    report['services'].append(restart)
    report['cleanup'].append(dict(label='original_restart_closed_first',pid=first.pid,returncode=first.poll()))
    report['resources'].append(snapshot('after_restart'))
    report['phase']='completed'
except BaseException:
    report['phase']='failed'
    report['error']=traceback.format_exc()
    print(report['error'],flush=True)
finally:
    interface.launch_appworld_environment_server=original_launch
    if world is not None:
        try:
            print(json.dumps(dict(phase='original_close_server',observed_unix=time.time())),flush=True)
            start=time.perf_counter();world.close_server()
            report['close_server_seconds']=time.perf_counter()-start
            report['owner_clean_after_close']=world.clean
        except BaseException:
            report['phase']='failed';report['close_error']=traceback.format_exc()
    for child in spawned:
        rc=child.poll()
        if rc is None:
            try:
                server.stop_appworld_environment_server(child)
                rc=child.poll()
            except BaseException:
                report.setdefault('cleanup_errors',[]).append(traceback.format_exc())
                server.force_stop_appworld_environment_server(child)
                rc=child.poll()
        report['cleanup'].append(dict(label='original_service_returncode',pid=child.pid,returncode=rc))
    report['resources'].append(snapshot('after_native_cleanup'))
    report['torch_imported']='torch' in sys.modules
    report['completed_unix']=time.time()
    report['seconds']=report['completed_unix']-report['started_unix']
    (out/'native-service-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(phase=report['phase'],out=str(out),seconds=report['seconds'],service_pids=[x['pid'] for x in report['services']])),flush=True)
if report['phase']!='completed':
    raise SystemExit(1)
