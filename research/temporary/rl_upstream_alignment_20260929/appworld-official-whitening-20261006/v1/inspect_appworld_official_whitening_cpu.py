"""Copy frozen AppWorld sources and exercise existing CPU owner interfaces only."""
import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
        if p.is_file() and '.git' not in p.relative_to(root).parts
        and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def native_inspection(args):
    """Fresh CPU process: actual imports plus the unchanged six owner tests."""
    import psutil
    import torch
    import unittest
    import verl.trainer.ppo.ray_trainer as trainer
    import verl.utils.torch_functional as functional
    import launch_appworld_native as launcher
    import owner_runtime_options as runtime
    import reward_readout
    import deltatrace_rollout
    import native_prefix_leases
    import deltatrace_credit

    started = time.time()
    spec = importlib.util.spec_from_file_location('app_owner_whitening_tests', args['test_source'])
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    # Reuse the unchanged tests on the exact App baseline; only its identity
    # differs from Text's 8816 predecessor (the async-manager binding).
    tests.SOURCE_PATHS['actual'] = Path(args['baseline_trainer'])
    tests.SOURCE_PATHS['pristine'] = Path(args['pristine_trainer'])
    tests.SOURCE_PATHS['functional'] = Path(functional.__file__)
    tests.SOURCE_SHAS['actual'] = args['baseline_trainer_sha256']
    with Path(args['test_stdout']).open('w') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromModule(tests))
    test_result = dict(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=result.skipped, successful=result.wasSuccessful(), source_path=args['test_source'],
        source_sha256=sha(args['test_source']), baseline_owner_identity=tests.SOURCE_SHAS,
        identity_binding_scope='Only existing tests SOURCE_PATHS and actual predecessor SHA bind App b174; test definitions unchanged')
    Path(args['test_json']).write_text(json.dumps(test_result, indent=2)+'\n')
    assert result.wasSuccessful(), Path(args['test_stdout']).read_text()
    assert tests.patch.patch_dt_advantage_preprocessing(Path(args['baseline_trainer']).read_text()) == Path(trainer.__file__).read_text()
    options, sampling = launcher.options_for(Path(args['output']), resume_from=Path(args['checkpoint']))
    modules = dict(trainer=trainer, official_helper=functional, launcher=launcher, runtime_options=runtime,
        reward_readout=reward_readout, producer=deltatrace_rollout, prefix_leases=native_prefix_leases,
        dt_credit=deltatrace_credit)
    record = dict(observed_unix=time.time(), pid=os.getpid(), pid_birth=psutil.Process().create_time(),
        wall_seconds=time.time()-started, max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        cuda_initialized=torch.cuda.is_initialized(), distributed_initialized=torch.distributed.is_initialized(),
        imports={k:dict(path=v.__file__, sha256=sha(v.__file__)) for k,v in modules.items()},
        options=options, sampling=sampling, owner_command=runtime.owner_command(options),
        focused_owner_tests=test_result, model_initializations=0, DT_calls=0, rollout_calls=0,
        backward_calls=0, optimizer_steps=0, readiness_candidate_combined=False,
        scope='Actual source imports, unchanged existing CPU tests and original options_for; no model or task execution')
    assert not record['cuda_initialized'] and not record['distributed_initialized']
    Path(args['inspection']).write_text(json.dumps(record, indent=2)+'\n')


def prepare(input_path):
    import psutil
    started = time.time()
    payload = json.loads(input_path.read_bytes()); out = Path(payload['receipt'])
    assert not (out/'prepared.json').exists(), 'Preserve completed preparation'
    for name, expected in payload['source_sha256'].items():
        assert sha(out/name) == expected, name
    read = lambda name: json.loads((out/'source/inputs'/name).read_bytes())
    original = read('original-prepared.json'); source = read('deployment-source.json')
    job = read('deployment-job.json'); launch = read('deployment-launch.json'); ending = read('ending-review.json')
    baseline_entry = Path(job['entry']); baseline_owner = Path(job['verl_root']); dt = Path(job['dt_root'])
    assert sha(Path(job['output'])/'source.json') == payload['source_sha256']['source/inputs/deployment-source.json']
    assert sha(Path(job['output'])/'job.json') == payload['source_sha256']['source/inputs/deployment-job.json']
    assert sha(Path(source['prepared_receipt'])) == source['prepared_receipt_sha256']
    for name, expected in source['entry_sha256'].items():
        assert sha(baseline_entry/name) == expected, name
    for name, expected in source['verl_sha256'].items():
        assert sha(baseline_owner/name) == expected, name
    for name, expected in original['dt_source_sha256'].items():
        assert sha(dt/name) == expected, name
    actor_checkpoint = ending['latest_completed_native_checkpoint']
    checkpoint = Path(actor_checkpoint['checkpoint_path'])
    assert checkpoint.name == 'global_step_28'
    marker = Path(actor_checkpoint['marker']['path'])
    assert marker.read_text().strip() == '28' and sha(marker) == actor_checkpoint['marker']['sha256']
    checkpoint_files = []
    for saved in actor_checkpoint['required_original_files']:
        p = Path(saved['path']); assert p.is_file() and p.stat().st_size == saved['bytes']
        if 'sha256' in saved: assert sha(p) == saved['sha256']
        checkpoint_files.append(dict(path=str(p), bytes=p.stat().st_size, recorded_sha256=saved.get('sha256')))
    baseline_trainer = baseline_owner/'verl/trainer/ppo/ray_trainer.py'
    assert sha(baseline_trainer) == 'b174dbfa6e1ca6a514a126aa2fe64399e1876103dbbcc500a27c4035966b0066'
    assert sha(baseline_entry/'reward_readout.py') == '8acf94d46cef8ca03ce1e92352723b2490c76e66a075f6b2a5f969163bcdc774'
    spec = importlib.util.spec_from_file_location('patch_verl_agent2', out/'source/experiments/rl/patch_verl_agent2.py')
    patch = importlib.util.module_from_spec(spec); spec.loader.exec_module(patch)
    base = Path(payload['candidate']); entry=base/'entry'; owner=base/'verl'
    assert not base.exists(), 'Use a distinct attempt; do not replace an earlier candidate'
    shutil.copytree(baseline_owner, owner, ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
    shutil.copytree(baseline_entry, entry, ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
    trainer = owner/'verl/trainer/ppo/ray_trainer.py'
    trainer.write_text(patch.patch_dt_advantage_preprocessing(baseline_trainer.read_text()))
    original_readout=(baseline_entry/'reward_readout.py').read_bytes()
    new_readout=(out/'source/accepted-reward-readout-94a7afbc.py').read_bytes()
    assert sha(out/'source/accepted-reward-readout-94a7afbc.py') == '94a7afbc09da72b62572d31fd32a6534f6e8f3daf656fce1011cdfa68b3c3e2b'
    old_tree=ast.parse(original_readout);new_tree=ast.parse(new_readout)
    old_class=next(n for n in old_tree.body if isinstance(n,ast.ClassDef) and n.name=='RewardAlphabet')
    new_class=next(n for n in new_tree.body if isinstance(n,ast.ClassDef) and n.name=='RewardAlphabet')
    old_query=next(n for n in old_class.body if isinstance(n,ast.FunctionDef) and n.name=='query_ids')
    new_query=next(n for n in new_class.body if isinstance(n,ast.FunctionDef) and n.name=='query_ids')
    old_lines=original_readout.splitlines(keepends=True);new_lines=new_readout.splitlines(keepends=True)
    assert old_lines[:old_query.lineno-1]==new_lines[:new_query.lineno-1]
    assert old_lines[old_query.end_lineno:]==new_lines[new_query.end_lineno:]
    (entry/'reward_readout.py').write_bytes(new_readout)
    manifests={}
    for label,before,after,expected in [('verl',baseline_owner,owner,['verl/trainer/ppo/ray_trainer.py']),
                                      ('entry',baseline_entry,entry,['reward_readout.py'])]:
        a,b=tree(before),tree(after);assert a.keys()==b.keys()
        changed=[name for name in a if a[name]!=b[name]];assert changed==expected,changed
        manifests[label]=dict(baseline=str(before),candidate=str(after),changed_files=changed,
            all_file_sha256={name:dict(baseline=a[name],candidate=b[name]) for name in a})
    assert 'self.traj_collector.async_rollout_manager = self.async_rollout_manager' in trainer.read_text()
    assert "self.readout_options['prefix_lease_factory'] = prepare_native_prefix_leases" in (entry/'deltatrace_rollout.py').read_text()
    candidate_source=dict(status='prepared_source_only_not_deployed', upstream_commit='20bd331',
        white_patch_commit='8f52b95658d37fc7a0cb69647de801188f70be02',query_clock_commit='afe59dd',
        source_sha256=payload['source_sha256'],manifests=manifests, dt_root=str(dt),dt_source_sha256=original['dt_source_sha256'],
        query_clock_diff=''.join(difflib.unified_diff(original_readout.decode().splitlines(True),new_readout.decode().splitlines(True))),
        trainer_diff=''.join(difflib.unified_diff(baseline_trainer.read_text().splitlines(True),trainer.read_text().splitlines(True))),
        no_whole_Text_trainer_copy=True,readiness_candidate_combined=False)
    (out/'candidate-source.json').write_text(json.dumps(candidate_source,indent=2)+'\n')
    env=os.environ.copy(); provisioned_dt=Path(env['DT_ROOT'])
    paths={str(provisioned_dt):str(dt),str(provisioned_dt/'experiments/rl'):str(dt/'experiments/rl')}
    inherited=[paths.get(p,p) for p in env['PYTHONPATH'].split(':')]
    env.update(VERL_ROOT=str(owner),DT_ROOT=str(dt),DT_ENTRY_ROOT=str(entry),LOOP_ROOT=original['loop_root'],
        APPWORLD_ROOT=str(Path(payload['root'])/'receipts/environment-only-20260930/loop-entry/appworld-root'),
        CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',**original['resource_environment'])
    env['PYTHONPATH']=':'.join([str(entry),str(owner),original['loop_root'],*inherited])
    args=dict(test_source=str(out/'source/tests/test_dt_official_whitening.py'),baseline_trainer=str(baseline_trainer),
        baseline_trainer_sha256=sha(baseline_trainer),pristine_trainer=str(out/'source/owner-pristine-trainer.py'),
        test_stdout=str(out/'focused-owner-tests.stdout.txt'),test_json=str(out/'focused-owner-tests.json'),
        inspection=str(out/'native-interface-inspection.json'),checkpoint=str(checkpoint),output=str(out/'config-only'))
    with (out/'native-interface-inspection.stdout.txt').open('wb') as log:
        subprocess.run([env['VENV_PYTHON'],__file__,'--native-inspection',json.dumps(args)],env=env,cwd=entry,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    inspection=json.loads(Path(args['inspection']).read_bytes())
    assert inspection['imports']['trainer']['path']==str(trainer)
    assert inspection['imports']['trainer']['sha256']==sha(trainer)
    assert inspection['imports']['official_helper']['sha256']==source['verl_sha256']['verl/utils/torch_functional.py']
    original_options=launch['options'];options=inspection['options']
    changes={k:dict(original=original_options.get(k),prepared=options.get(k))
        for k in original_options.keys()|options.keys() if original_options.get(k)!=options.get(k)}
    allowed={'data.custom_cls.path','trainer.default_local_dir','trainer.rollout_data_dir',
        'trainer.validation_data_dir','+ray_init.runtime_env.env_vars.DT_WORKER_VISIBILITY_DIR','trainer.resume_from_path'}
    assert set(changes)==allowed,changes
    assert inspection['sampling']==original['unchanged_sampling']
    for name,value in [('actor_rollout_ref.model.lora_rank',8),('actor_rollout_ref.model.lora_alpha',16),
                       ('actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu',4),
                       ('actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu',4)]: assert options[name]==value
    dry_argv=[env['VENV_PYTHON'],str(entry/'launch_appworld_native.py'),'--output',args['output'],
        '--resume-from',str(checkpoint),'--config-only']
    with (out/'effective-config.yaml').open('wb') as log:
        subprocess.run(dry_argv,env=env,cwd=entry,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert json.loads((Path(args['output'])/'launch.json').read_bytes())['options']==options
    comparison=dict(configuration_changes=changes,unchanged_sampling=inspection['sampling'],
        original_budget=job['budget'], options=options,DT_minibatch_size=4,
        DT_minibatch_source='Unchanged producer/readout default minibatch_size=4; no runtime override added',
        transport_limit=options['actor_rollout_ref.rollout.max_model_len'],
        scope='Only source/output paths and explicit checkpoint20-to28 resume differ; no task/actor/loss/DT budget option change')
    (out/'configuration-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    record=dict(prepared_unix=time.time(),status='prepared_only_CPU_owner_interface_verified_not_launched',
        repository_commit=payload['repository_commit'],upstream_commit='20bd331',source_sha256=payload['source_sha256'],
        input_context_sha256=payload['context_sources'],entry=str(entry),verl_root=str(owner),dt_root=str(dt),
        trainer_sha256=sha(trainer),reward_readout_sha256=sha(entry/'reward_readout.py'),
        checkpoint_root=str(checkpoint),actor_checkpoint=str(checkpoint/'actor'),checkpoint_files=checkpoint_files,
        marker=dict(path=str(marker),value=28,sha256=sha(marker)),configuration_changes=changes,
        environment_source=dict(path=payload['provisioned_entry']+'/metax-entry.env.sh',
            sha256=sha(Path(payload['provisioned_entry'])/'metax-entry.env.sh')),
        environment={k:env[k] for k in sorted(env) if k.startswith(('DT_','VERL_','FLA_')) or k in
            ('VENV_PYTHON','PYTHONPATH','LOOP_ROOT','LOOP_EXTRAS','APPWORLD_ROOT','CUDA_VISIBLE_DEVICES',
             'MACA_VISIBLE_DEVICES','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','MACA_TORCH_COMPILE_CONF')},
        process=dict(pid=os.getpid(),pid_birth=psutil.Process().create_time(),
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),wall_seconds=time.time()-started,
        native_inspection_resource={k:inspection[k] for k in ('pid','pid_birth','wall_seconds','max_rss_bytes','cuda_initialized','distributed_initialized')},
        no_launch=True,model_initializations=0,DT_calls=0,rollout_calls=0,backward_calls=0,optimizer_steps=0,
        readiness_candidate_combined=False,active_manifest_modified=False,
        receipts={name:dict(path=str(out/name),sha256=sha(out/name)) for name in
            ('candidate-source.json','native-interface-inspection.json','native-interface-inspection.stdout.txt',
             'focused-owner-tests.stdout.txt','focused-owner-tests.json','effective-config.yaml','configuration-comparison.json')},
        scope='Frozen App b174 owner and route/prefix entry copied; only existing official DT whitening seam and accepted two-sentence query-clock change. CPU interfaces/configuration only, not App training or numerical acceptance.')
    (out/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('status','entry','verl_root','trainer_sha256','reward_readout_sha256','checkpoint_root','wall_seconds','native_inspection_resource')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--native-inspection')
    args=parser.parse_args()
    if args.native_inspection:
        native_inspection(json.loads(args.native_inspection))
    else:
        prepare(args.input)
