"""Prepare isolated native two-iteration speed trials; never modify live jobs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
VERSION='appworld-sql-native-speed-20261016-v1'
AUDIT=ROOT/'receipts'/VERSION
DEST=ROOT/'candidates'/VERSION
BASE=ROOT/'candidates/appworld-stable-memory-entry-20261010-v3/appworld'
V5=ROOT/'runs/textcraft-dt-reward-target-exponential-20261016-v5'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')

assert not DEST.exists(),'Inspect existing preparation; never overwrite a candidate.'
v5=read(V5/'launch.json');app=read(BASE/'source-template.json')
assert v5['local_patch_commit']=='4545c55807e7d002287717b6a4254f9b083dd5fc'
authorities={str(ROOT/n):sha(ROOT/n) for n in ('formal-training.json','active-training.json','active-source.json')}
DEST.mkdir();AUDIT.mkdir(exist_ok=True)
prepared=[]
for task in ('SQL','AppWorld'):
    directory=DEST/task;entry=directory/'entry';entry.mkdir(parents=True)
    for path in Path(app['entry']).iterdir():
        if path.is_file() and (path.suffix=='.py' or path.name in ('owner_environment_configs.json','qwen3_acc_thinking.jinja2','loop-project-environment-comparison.json')):
            shutil.copyfile(path,entry/path.name)
    for name in ('dt_training_batch.py','reward_readout.py','counterfactual.py'):
        shutil.copyfile(Path(v5['effective_environment']['DT_ENTRY_ROOT'])/name,entry/name)
    producer=next(Path(p) for p in v5['source_bindings'] if Path(p).name=='deltatrace_rollout.py')
    assert sha(producer)==v5['source_bindings'][str(producer)]
    shutil.copyfile(producer,entry/'deltatrace_rollout.py')
    env=dict(os.environ);env.update(app['environment']);env.update(v5['effective_environment'])
    env.update(DT_ENTRY_ROOT=str(entry),DT_TASK='AppWorld' if task=='AppWorld' else 'SkyRL-SQL',
        CUDA_VISIBLE_DEVICES='2,3',DT_CREDIT_MODE='reward_target_exponential_v5',
        DT_TARGET_SEMANTICS='native_joint_action_target')
    env['LOOP_ROOT']=app['environment']['LOOP_ROOT'];env['LOOP_EXTRAS']=app['environment']['LOOP_EXTRAS']
    parts=[str(entry),env['VERL_ROOT'],env['DT_ROOT']+'/clean/qwen35',env['LOOP_ROOT']]
    for source in (v5['effective_environment']['PYTHONPATH'],app['environment']['PYTHONPATH']):
        for part in source.split(':'):
            if part and part not in parts and not any(part==old or part.startswith(old+'/') for old in
                (app['entry'],app['verl_root'],app['dt_root'],v5['effective_environment']['DT_ENTRY_ROOT'])):
                parts.append(part)
    env['PYTHONPATH']=':'.join(parts)
    for key in ('RAY_ADDRESS','RAY_TMPDIR','MACA_VISIBLE_DEVICES'):env.pop(key,None)
    run=ROOT/'runs'/VERSION/task.lower()
    if task=='AppWorld':
        path=entry/'loop_owner_recipe.py';source=path.read_text()
        old='return get_config("train", training_overrides(owner_root) + overrides)'
        assert source.count(old)==1
        source=source.replace(old,'import json, os\n    bounded = json.loads(os.environ.get("DT_LOOP_OVERRIDES_JSON", "[]"))\n    return get_config("train", training_overrides(owner_root) + bounded + overrides)')
        path.write_text(source)
        env['DT_LOOP_OVERRIDES_JSON']=json.dumps(['rl.params.scenarios_per_iteration=4'])
        path=entry/'loop_owner_rollout.py';source=path.read_text()
        old='if key not in executions:';assert source.count(old)==1
        source=source.replace(old,'if key not in executions or key != turns[-1][0]:')
        path.write_text(source)
        options=dict(app['startup_options'])
        options.update({'data.train_batch_size':4,'+data.loop_train_groups':4,
            'data.custom_cls.path':str(entry/'loop_iteration_dataset.py')})
    else:
        # The original SQL generator owns exact final-action boundaries and IDs.
        # SQLEnv._get_reward returns zero until done, then scores that action's
        # completed history, including native format -1 and execution 0/1.
        path=entry/'owner_trajectory_batch.py';source=path.read_text()
        marker="    return output\n\n\n def trajectory_credit".replace(' def','def')
        assert source.count(marker)==1
        seam='''    if dt:
        from executed_target_spans import full_trajectory_artifact
        artifacts=[]
        for trajectory,turns in zip(trajectories,manager.rollouts.records):
            full=list(trajectory.prompt_ids)+list(trajectory.response_ids)
            policy=[False]*len(trajectory.prompt_ids)+list(trajectory.loss_mask)
            target=[False]*len(full)
            if turns:
                turn=turns[-1]
                start,end=turn['start'],turn['end']
                if full[start:end] != turn['output_ids'][:end-start]:
                    raise ValueError('Native SQL terminal action IDs differ')
                for offset in range(start,end):target[offset]=bool(policy[offset])
            item=full_trajectory_artifact(trajectory.prompt_ids,full,policy,target,trajectory.response_ids)
            item['retained_response_positions']=[-1]*(response_width-len(trajectory.response_ids))+item['retained_response_positions']
            item['reward_settlement_owner']='SQLEnv._get_reward(done=True); native final action'
            artifacts.append(item)
        array=np.empty(len(artifacts),dtype=object);array[:]=artifacts
        output.non_tensor_batch['dt_direct_target_artifact']=array
'''
        source=source.replace(marker,seam+marker);path.write_text(source)
        # Compose the already existing original bounded SQL entry on CPU.
        cfg_script='''import argparse,json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import launch_sql_native as owner
a=argparse.Namespace(method="dt",phase="bounded",data=sys.argv[2],output=sys.argv[3],resume_from=None)
print(json.dumps(owner.command(a)[1]))
'''
        cpuenv=dict(env);cpuenv['CUDA_VISIBLE_DEVICES']=''
        output=subprocess.check_output([env['VENV_PYTHON'],'-c',cfg_script,str(entry),
            str(ROOT/'datasets/skyrl-sql-7e5e665'),str(run)],env=cpuenv,text=True)
        options=json.loads(output)
        options['data.train_batch_size']=8
    options.update({'trainer.experiment_name':VERSION+'-'+task.lower(),
        'trainer.default_local_dir':str(run/'checkpoints'),'trainer.rollout_data_dir':str(run/'rollouts'),
        'trainer.validation_data_dir':str(run/'validation'),'trainer.total_training_steps':2,
        'trainer.save_freq':-1,'trainer.test_freq':-1,'trainer.val_before_train':False,
        'trainer.resume_mode':'disable','trainer.resume_from_path':None,
        'actor_rollout_ref.rollout.enforce_eager':False,'actor_rollout_ref.rollout.free_cache_engine':False,
        '+ray_init.runtime_env.worker_process_setup_hook':'metax_fsdp_event_wait.worker_setup',
        '+ray_init.runtime_env.env_vars.DT_FSDP_EVENT_WAIT_RECEIPTS':str(run/'worker-runtime-overrides'),
        '+ray_init.runtime_env.env_vars.DT_CREDIT_MODE':'reward_target_exponential_v5',
        '+ray_init.runtime_env.env_vars.VERL_RELEASE_UNUSED_HOST_CACHE':'1'})
    options.pop('+ray_init.runtime_env.env_vars.DT_WORKER_VISIBILITY_DIR',None)
    env['DT_MAX_STEPS']=str(options['env.max_steps'])
    env['DT_MAX_LENGTH']='32768'
    env['DT_SAMPLING_JSON']=json.dumps(options['+env.sql']['sampling'] if task=='SQL' else
        dict(temperature=options['actor_rollout_ref.rollout.temperature'],
             max_tokens=options['data.max_response_length']))
    env['DT_FSDP_EVENT_WAIT_RECEIPTS']=str(run/'worker-runtime-overrides')
    # Original owner serializes Hydra values; no competing launcher/trainer.
    script='''import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from owner_runtime_options import owner_command
print(json.dumps(owner_command(json.loads(sys.stdin.read()))))
'''
    cpuenv=dict(env);cpuenv['CUDA_VISIBLE_DEVICES']=''
    argv=json.loads(subprocess.check_output([env['VENV_PYTHON'],'-c',script,str(entry)],
        input=json.dumps(options),env=cpuenv,text=True))
    bindings={str(p):sha(p) for p in entry.iterdir() if p.is_file()}
    for path,expected in v5['source_bindings'].items():
        if path.startswith(env['VERL_ROOT']+'/') or path.startswith(env['DT_ROOT']+'/'):
            assert sha(path)==expected;bindings[path]=expected
    record=dict(task=task,version=VERSION,run=str(run),entry=str(entry),argv=argv,options=options,
        effective_environment=env,physical_gpus=[2,3],status='prepared_only',
        original_task_limits_preserved=True,source_bindings=bindings,
        upstream_commits={'VERL':'20bd331b','LOOP':'f14107a','SkyRL':'7d94cc'},
        baseline_v5_commit=v5['local_patch_commit'],budget=dict(iterations=2,
            prompt_groups=4 if task=='AppWorld' else 8,rollouts_per_group=6 if task=='AppWorld' else 5,
            checkpoints=False,evaluation=False),
        scope='Bounded native training speed trial; not a formal quality experiment or full-batch throughput claim.')
    write(directory/'prepared.json',record);prepared.append(dict(task=task,path=str(directory/'prepared.json')))
assert authorities=={p:sha(p) for p in authorities}
write(AUDIT/'prepared-index.json',dict(version=VERSION,tasks=prepared,formal_authorities_unchanged=authorities))
print(json.dumps(dict(version=VERSION,tasks=prepared,status='prepared_not_launched')))
