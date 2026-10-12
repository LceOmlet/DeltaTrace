"""Submit the recorded native commands after the bounded graph checks pass."""
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import psutil

ROOT = Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT = ROOT/'receipts/textcraft-native-graph-deployment-20261012-v1'


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    verification = json.loads((OUT/'integration-result.json').read_bytes())
    assert verification.get('status') == 'passed_native_graph_B8_32768_DT_update_checkpoint_roundtrip'
    assert verification['dataloader_next_batch_exact']
    assert all(all(w['restoration_matches'].values()) for w in verification['restored_worker_state'])
    checkpoint_extension=json.loads((OUT/'checkpoint-rng-extension.json').read_bytes())
    assert all(w['restoration_matches'].get('rollout_rng') for w in verification['restored_worker_state'])
    assert all(json.loads((OUT/f'gate-transfer-rank{rank}.json').read_bytes())['all_original_elements_exact'] for rank in range(2))
    gpu = subprocess.check_output(['mx-smi'], text=True)
    assert 'no process found' in gpu, 'Inspect current occupancy; do not disturb unrelated processes'
    (OUT/'formal-gpu-before.txt').write_text(gpu)
    authoritative = {name: json.loads((ROOT/name).read_bytes()) for name in
                     ('formal-training.json','active-training.json','active-source.json')}
    raw = subprocess.check_output(['bash','-c','source "$1"; env -0','bash',
                                  str(ROOT/'receipts/environment-only-20260930/entry/metax-entry.env.sh')])
    base_env = dict(x.decode().split('=',1) for x in raw.split(b'\0') if b'=' in x)
    jobs=[]
    for method, previous, devices in [
        ('dt','textcraft-dt-formal-eventwait-fresh-20261012-v2',[2,3]),
        ('grpo','textcraft-grpo-formal-eventwait-fresh-20261012-v1',[4,5]),
    ]:
        old=ROOT/'runs'/previous
        launch=json.loads((old/'launch.json').read_bytes())
        source=json.loads((old/'source.json').read_bytes())
        old_job=next(j for j in authoritative['formal-training.json']['jobs'] if j.get('output')==str(old/('textcraft-'+method)))
        if psutil.pid_exists(old_job['pid']):
            assert psutil.Process(old_job['pid']).create_time()!=old_job['observed_process_created_unix'] or psutil.Process(old_job['pid']).status()=='zombie'
        run=ROOT/'runs'/f'textcraft-{method}-formal-native-graph-fresh-20261012-v1'
        run.mkdir(exist_ok=False)
        options=copy.deepcopy(launch['options'])
        for key,value in list(options.items()):
            if isinstance(value,str):options[key]=value.replace(str(old),str(run))
        options.update({'actor_rollout_ref.rollout.enforce_eager':False,
                        'actor_rollout_ref.rollout.free_cache_engine':False,
                        'trainer.save_freq':5})
        argv=[x.replace(str(old),str(run)) for x in launch['argv']]
        argv=[('trainer.save_freq=5' if x=='trainer.save_freq=25' else x) for x in argv]
        for key in ('actor_rollout_ref.rollout.enforce_eager','actor_rollout_ref.rollout.free_cache_engine'):
            assert not any(x.startswith(key+'=') for x in argv)
            argv.append(key+'=false')
        expected={'trainer.default_local_dir','trainer.rollout_data_dir','trainer.validation_data_dir',
                  '+ray_init.runtime_env.env_vars.DT_FSDP_EVENT_WAIT_RECEIPTS',
                  'actor_rollout_ref.rollout.enforce_eager','actor_rollout_ref.rollout.free_cache_engine','trainer.save_freq'}
        changes={k:dict(before=launch['options'].get(k,'upstream_default'),after=v) for k,v in options.items() if launch['options'].get(k,'upstream_default')!=v}
        assert set(changes)==expected,changes
        assert options['actor_rollout_ref.model.lora_rank']==8
        assert options['actor_rollout_ref.model.lora_alpha']==16
        assert options['actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu']==4
        assert options['actor_rollout_ref.rollout.max_model_len']==32768
        assert options['actor_rollout_ref.actor.ppo_mini_batch_size']==64
        assert options['trainer.total_epochs']==30 and options['trainer.resume_from_path'] is None
        env=dict(base_env);env.update(launch.get('effective_environment') or source['environment'])
        memory=json.loads((OUT/'memory-inheritance.json').read_bytes())
        env['PYTHONPATH']=env['PYTHONPATH'].replace(env['VERL_ROOT'],checkpoint_extension['new_VERL_ROOT'])
        env['VERL_ROOT']=checkpoint_extension['new_VERL_ROOT']
        if method=='dt':
            env['PYTHONPATH']=env['PYTHONPATH'].replace(env['DT_ROOT'],memory['new_DT_ROOT'])
            env['DT_ROOT']=memory['new_DT_ROOT']
            env['DT_ENVIRONMENT_JSON']=memory['environment']
        env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,devices)), DT_FSDP_EVENT_WAIT_RECEIPTS=str(run/'worker-runtime-overrides'))
        for k in ('RAY_ADDRESS','RAY_TMPDIR','MACA_VISIBLE_DEVICES'):env.pop(k,None)
        with (run/'driver.log').open('wb') as log:
            process=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        birth=psutil.Process(process.pid).create_time()
        record=dict(unix=time.time(),pid=process.pid,birth=birth,run=str(run),log=str(run/'driver.log'),
                    local_patch_commit=os.environ.get('DT_PATCH_SOURCE_COMMIT'),
                    argv=argv,options=options,effective_environment={k:env[k] for k in source['environment']},
                    physical_gpus=devices,budget=launch['budget'],fresh_base_model=True,resume_checkpoint=None,
                    previous_launch=str(old/'launch.json'),previous_launch_SHA256=digest(old/'launch.json'),
                    configuration_changes=changes,numerical_commit='26bef6c8b2e49db118f46e3c05e86944dcf8e293',
                    FSDP_fix_commit='2534a670974e388bc2b9a01d4f304d5c23b37534',
                    checkpoint_extension=checkpoint_extension,
                    execution_verification=str(OUT/'integration-result.json'),execution_verification_SHA256=digest(OUT/'integration-result.json'))
        save(run/'launch.json',record)
        for k in ('runtime_override','deployment','passive_credit_recording_runtime_override','initial_worker_state_verification',
                  'actual_initial_worker_state_receipt','actual_initial_worker_state_verified','actual_initial_worker_states'):
            source.pop(k,None)
        source.update(verl_root=checkpoint_extension['new_VERL_ROOT'],
                      local_patch_commit=record['local_patch_commit'],
                      current_checkpoint_extension=checkpoint_extension,
                      environment=record['effective_environment'],inherited_source_receipt=str(old/'source.json'),
                      inherited_source_sha256=digest(old/'source.json'),fresh_base_model=True,resume_from_path=None,
                      deployment=dict(pid=process.pid,birth=birth,status='formal_initializing',launch=str(run/'launch.json')),
                      current_native_graph_verification=record['execution_verification'],
                      actual_import_scope='Inherited accepted files; actual new-worker import verification pending',
                      passive_recording_status='pending installation in new DT workers')
        if method=='dt':
            source.update(dt_root=memory['new_DT_ROOT'],memory_runtime=memory,
                          inherited_binding_scope='Historical bindings retained as provenance; current imports are recorded separately below',
                          current_DT_bindings={str(p):digest(p) for p in (Path(memory['new_DT_ROOT'])/'clean/qwen35').glob('*.py')})
            source['current_DT_bindings'].update(memory['gate_capture_host_fix']['changed_files'])
        # The existing backup selector consumes this canonical source map.
        # Keep historical bindings explicitly separate from the current files.
        source['historical_source_bindings']=source.get('source_bindings',{})
        current_bindings={}
        old_env=json.loads((old/'source.json').read_bytes())['environment']
        for name in source['historical_source_bindings']:
            current=name.replace(old_env['VERL_ROOT'],env['VERL_ROOT'])
            if method=='dt':current=current.replace(old_env['DT_ROOT'],env['DT_ROOT'])
            if Path(current).is_file():current_bindings[current]=digest(current)
        current_bindings.update(source.get('current_DT_bindings',{}))
        current_bindings.update({str(Path(env['VERL_ROOT'])/name):item['after_sha256']
                                for name,item in checkpoint_extension['changed_files'].items()})
        for name in ('integration-result.json','integration-launch.json','memory-inheritance.json',
                     'checkpoint-rng-extension.json','dt-gate-host.patch','checkpoint-rollout-rng.patch',
                     'gate-transfer-rank0.json','gate-transfer-rank1.json'):
            current_bindings[str(OUT/name)]=digest(OUT/name)
        source['source_bindings']=current_bindings
        save(run/'source.json',source)
        job=copy.deepcopy(old_job)
        for k in ('ray_session','completed_iterations','last_metric','runtime_override',
                  'passive_credit_recording_runtime_override','initial_worker_state_verification','stop_receipt',
                  'workers','actual_initial_worker_state_receipt','actual_initial_worker_state_verified','actual_initial_worker_states'):
            job.pop(k,None)
        job.update(pid=process.pid,birth=birth,observed_process_created_unix=birth,started_unix=record['unix'],
                   dt_root=env['DT_ROOT'],verl_root=env['VERL_ROOT'],
                   output=str(run/('textcraft-'+method)),log=str(run/'driver.log'),
                   checkpoints=str(run/('textcraft-'+method)/'checkpoints'),source_receipt=str(run/'source.json'),
                   argv=argv,status='formal_initializing',fresh_base_model=True,resume_checkpoint=None,
                   completed_iterations=0,devices=devices,passive_recording_status='pending new worker installation',
                   save_freq=5,native_graph_verification=str(OUT/'integration-result.json'))
        jobs.append(job)
    for name,value in authoritative.items():
        updated=copy.deepcopy(value)
        updated['jobs']=[j for j in value['jobs'] if not(j.get('task')=='TextCraft' and j.get('method') in ('dt','grpo'))]+jobs
        updated.update(unix=time.time(),manifest=str(OUT/'formal-training.json'))
        save(ROOT/name,updated);save(OUT/name,updated)
    result=dict(unix=time.time(),status='submitted_full_formal_budget_native_graph_save5',
                submission_script_sha256=digest(__file__),jobs=[{k:j[k] for k in
                    ('method','pid','observed_process_created_unix','devices','output','log','save_freq')} for j in jobs])
    save(OUT/'formal-submission.json',result);print(json.dumps(result))


if __name__=='__main__':
    main()
