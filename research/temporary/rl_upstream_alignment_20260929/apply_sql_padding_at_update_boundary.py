"""Queue the tested owner forward after SQL's first original update RPC.

No process stop, model replacement, optimizer update, or task-config change.
The original sequential worker RPC runs this only after its current update.
"""
import hashlib
from pathlib import Path
import subprocess
from stage_environment_entry import remote, ROOT, ENTRY, REPO

revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
remote(r'''set -e
source @ENTRY@/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'
from pathlib import Path
import hashlib,json,psutil,ray,subprocess,time
root=Path('@ROOT@');base=root/'receipts/owner-b8-dispatch-20260930/actor-response-padding'
out=base/'sql-live';out.mkdir(exist_ok=True)
assert not (out/'submitted.json').exists(), 'Inspect the existing submission instead of queuing twice'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proof=read(base/'candidate.json');assert proof['test_returncode']==0
owner_test=read(base/'real-model/owner-padding-assertion.json');assert owner_test['passed']
candidate=Path(proof['candidate'])/'verl/workers/actor/dp_actor.py'
assert sha(candidate)==proof['after_sha256']
job=next(j for j in read(root/'active-training.json')['jobs'] if j['task']=='SkyRL-SQL')
assert job['pid']==proof['prior_driver_pid'] and job['devices']==[0,1]
driver=psutil.Process(job['pid']);assert driver.create_time()==job['observed_process_created_unix']
children=driver.children(recursive=True)
workers=[p for p in children if 'WorkerDict' in p.name()]
assert len(workers)==2 and all('actor_rollout_update_actor' in p.name() for p in workers), 'Only queue behind the observed original update'
task_runner=next(p for p in children if 'TaskRunner' in p.name())
stack=subprocess.run(['/opt/conda/bin/py-spy','dump','--pid',str(task_runner.pid)],capture_output=True,text=True,timeout=15)
(out/'before-taskrunner.txt').write_text(stack.stdout)
gcs=next(p for p in children if p.name()=='gcs_server')
port=next(s.split('=',1)[1] for s in gcs.cmdline() if s.startswith('--gcs_server_port='))

def apply(worker,path,expected,before_expected,receipt):
    import hashlib,importlib.util,inspect,json,os,sys,time
    from pathlib import Path
    from types import MethodType
    path=Path(path);digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert digest(path)==expected
    records=[]
    for role,w in worker.worker_dict.items():
        if not hasattr(w,'actor'):continue
        actor=w.actor
        steps=sorted({float(s['step'].item()) for s in actor.actor_optimizer.state.values() if 'step' in s})
        assert steps==[1.0], ('Expected completed original SQL update 1',steps)
        assert actor.config.ppo_micro_batch_size_per_gpu==4
        assert w.config.model.lora_rank==8 and w.config.model.lora_alpha==16
        assert actor.use_fused_kernels and os.environ.get('VERL_TRIM_SHARED_PADDING')=='1'
        old_source=inspect.getsourcefile(actor._forward_micro_batch)
        assert digest(old_source)==before_expected,old_source
        before_config=repr(actor.config);model_id=id(actor.actor_module);optim_id=id(actor.actor_optimizer)
        name='_verified_response_padding_'+expected[:12]
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
        actor._forward_micro_batch=MethodType(module.DataParallelPPOActor._forward_micro_batch,actor)
        assert repr(actor.config)==before_config and id(actor.actor_module)==model_id and id(actor.actor_optimizer)==optim_id
        record=dict(task='SkyRL-SQL',pid=os.getpid(),rank=w.rank,unix=time.time(),role=role,
            original_optimizer_steps=steps,effective_forward_source=str(path),effective_source_sha256=expected,
            prior_forward_source=old_source,prior_forward_sha256=before_expected,
            actor_microbatch=4,lora_rank=8,lora_alpha=16,
            boundary='Original Worker.execute_with_func_generator queued behind completed update_actor 1',
            scope='Only bound original-owner forward changed; model/optimizer/config/DT/vLLM/loss unchanged')
        (Path(receipt)/f'rank{w.rank}.json').write_text(json.dumps(record,indent=2)+'\n')
        records.append(record)
    assert len(records)==1
    return records[0]

ray.init(address=f'127.0.0.1:{port}',log_to_driver=False)
try:
    actors=[n for n in ray.util.list_named_actors(all_namespaces=True) if 'WorkerDict' in n['name']]
    assert len(actors)==2
    submission=dict(task=job['task'],driver_pid=job['pid'],driver_created_unix=driver.create_time(),
        worker_pids=[p.pid for p in workers],submitted_unix=time.time(),actors=actors,
        source=str(candidate),source_sha256=proof['after_sha256'],
        owner_comparison_receipt=str(base/'real-model/owner-padding-assertion.json'),
        owner_comparison_sha256=sha(base/'real-model/owner-padding-assertion.json'),
        submission_repository_commit=@REVISION@,submission_script_sha256=@SCRIPT_SHA@,
        status='queued_after_update_not_yet_applied')
    (out/'submitted.json').write_text(json.dumps(submission,indent=2)+'\n')
    refs=[ray.get_actor(n['name'],namespace=n['namespace']).execute_with_func_generator.remote(
        apply,str(candidate),proof['after_sha256'],proof['before_sha256'],str(out)) for n in actors]
    print(json.dumps(submission),flush=True)
    records=ray.get(refs)
    stack=subprocess.run(['/opt/conda/bin/py-spy','dump','--pid',str(task_runner.pid)],capture_output=True,text=True,timeout=15)
    (out/'after-taskrunner.txt').write_text(stack.stdout)
    result=dict(submission,status='applied_after_original_update_1',completed_unix=time.time(),workers=records,
        after_taskrunner_stack=str(out/'after-taskrunner.txt'))
    (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
finally:
    ray.shutdown()
PY
'''.replace('@ROOT@',ROOT).replace('@ENTRY@',ENTRY)
   .replace('@REVISION@',repr(revision)).replace('@SCRIPT_SHA@',repr(script_sha)))
