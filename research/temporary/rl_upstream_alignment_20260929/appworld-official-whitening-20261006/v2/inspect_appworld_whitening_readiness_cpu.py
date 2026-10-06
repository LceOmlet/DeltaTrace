"""CPU composition only: frozen App white seam plus original LOOP repair."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import time


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_delegate(path,expected):
    assert sha(path)==expected
    spec=importlib.util.spec_from_file_location('app_whitening_v1_cpu',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def native_inspection(args):
    original=load_delegate(args['delegate']['path'],args['delegate']['sha256'])
    original.native_inspection(args)
    import phi_agents.appworld.interface as interface
    import torch
    p=Path(args['inspection']);record=json.loads(p.read_bytes())
    record['imports']['loop_interface']=dict(path=interface.__file__,sha256=sha(interface.__file__))
    assert record['imports']['loop_interface']==args['expected_interface']
    record['delegate']=args['delegate']
    record['readiness_role']='Verified isolated original LOOP interface; no service/init/restart call in this preparation'
    record['readiness_candidate_combined']=True
    record['cuda_initialized']=torch.cuda.is_initialized()
    assert not record['cuda_initialized']
    p.write_text(json.dumps(record,indent=2)+'\n')


def prepare(input_path):
    import psutil
    started=time.time();payload=json.loads(input_path.read_bytes());out=Path(payload['receipt']);v1=Path(payload['v1_receipt'])
    assert not (out/'prepared.json').exists(),'Preserve completed preparation'
    for name,record in payload['sources'].items():assert sha(out/name)==record['sha256'],name
    read=lambda name:json.loads((out/'source/inputs'/name).read_bytes())
    prior=read('app-v1-prepared.json');assert sha(v1/'prepared.json')==payload['sources']['source/inputs/app-v1-prepared.json']['sha256']
    v1_source=json.loads((v1/'candidate-source.json').read_bytes())
    assert sha(v1/'candidate-source.json')==prior['receipts']['candidate-source.json']['sha256']
    original=json.loads((v1/'source/inputs/original-prepared.json').read_bytes())
    job=json.loads((v1/'source/inputs/deployment-job.json').read_bytes())
    actual_source=json.loads((v1/'source/inputs/deployment-source.json').read_bytes())
    actual_launch=json.loads((v1/'source/inputs/deployment-launch.json').read_bytes())
    readiness=read('readiness-prepared.json');readiness_review=read('readiness-verification-review.json')
    cleanup=read('readiness-completed-resource-status.json');native_service=read('readiness-native-service-results.json')
    loop=Path(readiness['candidate_root']);entry=Path(prior['entry']);owner=Path(prior['verl_root']);dt=Path(prior['dt_root'])
    for item in readiness['files']:assert sha(loop/item['path'])==item['candidate_sha256'],item['path']
    interface=loop/'phi_agents/appworld/interface.py'
    assert sha(interface)=='17adc301aae395b789b5528abc167fd025b6cbb22379e954c600ed586987610e'
    unchanged={}
    for label,manifest in v1_source['manifests'].items():
        for name,pair in manifest['all_file_sha256'].items():assert sha(Path(manifest['candidate'])/name)==pair['candidate'],name
        unchanged[label]=dict(file_count=len(manifest['all_file_sha256']),changed_from_actual1856052=manifest['changed_files'],
            additional_v2_source_changes=[])
    for name,expected in original['dt_source_sha256'].items():assert sha(dt/name)==expected,name
    checkpoint=Path(prior['checkpoint_root']);marker=Path(prior['marker']['path'])
    assert marker.read_text().strip()=='28' and sha(marker)==prior['marker']['sha256']
    for item in prior['checkpoint_files']:
        p=Path(item['path']);assert p.stat().st_size==item['bytes']
        if item['recorded_sha256']:assert sha(p)==item['recorded_sha256']
    assert not Path(payload['formal_output']).exists(),'Preparation must not own a prior run directory'
    env=os.environ.copy();env.update(prior['environment'])
    old_loop=original['loop_root'];env['PYTHONPATH']=':'.join(str(loop) if p==old_loop else p for p in env['PYTHONPATH'].split(':'))
    env.update(LOOP_ROOT=str(loop),**original['resource_environment'])
    # Same original device assignment is only recorded in the future plan.
    env['CUDA_VISIBLE_DEVICES']=','.join(map(str,job['devices']));env.pop('MACA_VISIBLE_DEVICES',None)
    cpu=env.copy();cpu.update(CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='')
    args=dict(delegate=payload['v1_native_source'],expected_interface=dict(path=str(interface),sha256=sha(interface)),
        test_source=str(v1/'source/tests/test_dt_official_whitening.py'),
        baseline_trainer=str(Path(original['verl_root'])/'verl/trainer/ppo/ray_trainer.py'),
        baseline_trainer_sha256='b174dbfa6e1ca6a514a126aa2fe64399e1876103dbbcc500a27c4035966b0066',
        pristine_trainer=str(v1/'source/owner-pristine-trainer.py'),test_stdout=str(out/'focused-owner-tests.stdout.txt'),
        test_json=str(out/'focused-owner-tests.json'),inspection=str(out/'native-interface-inspection.json'),
        checkpoint=str(checkpoint),output=payload['formal_output'])
    with (out/'native-interface-inspection.stdout.txt').open('wb') as log:
        subprocess.run([cpu['VENV_PYTHON'],__file__,'--native-inspection',json.dumps(args)],cwd=entry,env=cpu,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    inspection=json.loads(Path(args['inspection']).read_bytes());options=inspection['options']
    baseline=actual_launch['options']
    changes={k:dict(actual1856052=baseline.get(k),prepared=options.get(k))
        for k in baseline.keys()|options.keys() if baseline.get(k)!=options.get(k)}
    output_keys={'trainer.default_local_dir','trainer.rollout_data_dir','trainer.validation_data_dir',
        '+ray_init.runtime_env.env_vars.DT_WORKER_VISIBILITY_DIR'}
    assert set(changes)==output_keys|{'data.custom_cls.path','trainer.resume_from_path','+env.loop'},changes
    assert options['+env.loop']==dict(owner_root=str(loop))
    assert inspection['sampling']==original['unchanged_sampling']
    argv=[env['VENV_PYTHON'],str(entry/'launch_appworld_native.py'),'--output',payload['formal_output'],
        '--resume-from',str(checkpoint)]
    dry_output=out/'config-only';dry_argv=list(argv);dry_argv[dry_argv.index('--output')+1]=str(dry_output);dry_argv+=['--config-only']
    with (out/'effective-config.yaml').open('wb') as log:
        subprocess.run(dry_argv,env=cpu,cwd=entry,stdout=log,stderr=subprocess.STDOUT,check=True)
    dry=json.loads((dry_output/'launch.json').read_bytes())['options']
    dry_changes={k:dict(formal=options.get(k),dry=dry.get(k)) for k in options.keys()|dry.keys() if options.get(k)!=dry.get(k)}
    assert set(dry_changes)==output_keys,dry_changes
    assert not Path(payload['formal_output']).exists()
    relevant=('VENV_PYTHON','PATH','LD_LIBRARY_PATH','LD_PRELOAD','PYTHONPATH','LOOP_ROOT','LOOP_EXTRAS','APPWORLD_ROOT',
        'CUDA_VISIBLE_DEVICES','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR','HF_HOME',
        'HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
        'PYTORCH_CUDA_ALLOC_CONF','OMP_NUM_THREADS','MKL_NUM_THREADS','MACA_TORCH_COMPILE_CONF')
    configured={k:env[k] for k in sorted(env) if k.startswith(('DT_','VERL_','FLA_')) or k in relevant}
    launch_plan=dict(argv=argv,working_directory=payload['formal_output'],environment=configured,
        environment_source=prior['environment_source'],remove_environment_keys=['MACA_VISIBLE_DEVICES'],
        environment_scope='Source provisioned environment then these exact configured native overrides; not a dump of inherited credentials',
        formal_output=payload['formal_output'],checkpoint_root=str(checkpoint),devices=job['devices'],
        role='Prepared plan only; standard original launcher/submitter owns future submission')
    (out/'launch-plan.json').write_text(json.dumps(launch_plan,indent=2)+'\n')
    comparison=dict(configuration_changes=changes,dry_vs_formal_output_only_changes=dry_changes,
        original_budget=job['budget'],unchanged_sampling=inspection['sampling'],options=options,
        scope='Only source/output paths and explicit checkpoint20-to28 resume differ; +env.loop.owner_root is a source path change')
    (out/'configuration-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    author=dict(actual_source['author_sha256']);author['phi_agents/appworld/interface.py']=sha(interface)
    for name,expected in author.items():assert sha(loop/name)==expected,name
    readiness_binding=dict(candidate_root=str(loop),interface=dict(path=str(interface),sha256=sha(interface)),
        receipts={name:dict(path=str(out/'source/inputs'/name),sha256=sha(out/'source/inputs'/name)) for name in
            ('readiness-prepared.json','readiness-native-service-results.json','readiness-completed-resource-status.json','readiness-verification-review.json')},
        scope='Verified original CPU service lifecycle candidate; no readiness/service call repeated by this prepare')
    binding=dict(sources=payload['sources'],v1_delegate=payload['v1_native_source'],v1_manifest=prior['receipts']['candidate-source.json'],
        unchanged_v1_owner_entry=unchanged,author_sha256=author,readiness=readiness_binding,
        context=dict(current_dated=payload['sources']['source/inputs/current-runtime-1791290776.json'],
            historical_root_snapshot=payload['sources']['source/inputs/current-runtime-20261005-historical.json']))
    (out/'source-binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    # Preserve the existing submitter's source schema, replacing obsolete prior
    # identities explicitly with the actual ended 1856052/28 boundary.
    record=dict(original,prepared_unix=time.time(),status='prepared_only_combined_CPU_verified_not_launched',
        prior_driver_pid=job['pid'],prior_entry=job['entry'],prior_verl_root=job['verl_root'],
        future_checkpoint_root=job['checkpoints'],minimum_completed_checkpoint=28,
        prior_source_receipt=job['output']+'/source.json',prior_source_sha256=sha(v1/'source/inputs/deployment-source.json'),
        entry=str(entry),verl_root=str(owner),dt_root=str(dt),loop_root=str(loop),
        entry_sha256={name:sha(entry/name) for name in actual_source['entry_sha256']},
        owner_sha256={name:pair['candidate'] for name,pair in v1_source['manifests']['verl']['all_file_sha256'].items()},
        owner_head_sha256={name:sha(owner/name) for name in actual_source['owner_head_sha256']},
        dt_source_sha256=original['dt_source_sha256'],author_sha256=author,
        author_readiness_binding=readiness_binding,resource_environment=original['resource_environment'],
        trainer_sha256=prior['trainer_sha256'],reward_readout_sha256=prior['reward_readout_sha256'],
        resume_checkpoint=str(checkpoint),checkpoint_root=str(checkpoint),actor_checkpoint=str(checkpoint/'actor'),
        checkpoint_files=prior['checkpoint_files'],marker=prior['marker'],configuration_changes=changes,
        unchanged_formal_options=options,unchanged_sampling=inspection['sampling'],prior_budget=job['budget'],
        resume_launcher=dict(path=str(entry/'launch_appworld_native.py'),sha256=sha(entry/'launch_appworld_native.py')),
        preparation_repository_commit=payload['repository_commit'],preparation_source_sha256=payload['sources'],
        v1_source_only_preparation=prior['receipts']['candidate-source.json'],readiness_candidate_combined=True,
        formal_output=payload['formal_output'],formal_output_created=False,launch_plan=dict(path=str(out/'launch-plan.json'),sha256=sha(out/'launch-plan.json')),
        process=dict(pid=os.getpid(),pid_birth=psutil.Process().create_time(),max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
        wall_seconds=time.time()-started,native_inspection_resource={k:inspection[k] for k in
            ('pid','pid_birth','wall_seconds','max_rss_bytes','cuda_initialized','distributed_initialized')},
        no_launch=True,active_manifest_modified=False,model_initializations=0,DT_calls=0,rollout_calls=0,backward_calls=0,optimizer_steps=0,
        receipts={name:dict(path=str(out/name),sha256=sha(out/name)) for name in
            ('source-binding.json','native-interface-inspection.json','native-interface-inspection.stdout.txt',
             'focused-owner-tests.json','focused-owner-tests.stdout.txt','effective-config.yaml','configuration-comparison.json','launch-plan.json')},
        scope='Prepared only: reuse frozen App white/clock entry+VERL and isolated verified LOOP readiness path; existing official launcher/optimizer/DT/task budgets untouched.')
    (out/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('status','entry','verl_root','loop_root','resume_checkpoint','wall_seconds','native_inspection_resource')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path);parser.add_argument('--native-inspection')
    args=parser.parse_args()
    if args.native_inspection:native_inspection(json.loads(args.native_inspection))
    else:prepare(args.input)
