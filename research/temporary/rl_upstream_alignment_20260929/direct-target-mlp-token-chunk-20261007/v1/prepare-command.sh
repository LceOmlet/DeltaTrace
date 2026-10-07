set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'

from pathlib import Path
import copy,hashlib,importlib.util,json,os,psutil,subprocess,sys,time
P=Path;R=P('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');O=P('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/direct-target-mlp-token-chunk-20261007-v1')
read=lambda p:json.loads(P(p).read_bytes())
sha=lambda p:hashlib.sha256(P(p).read_bytes()).hexdigest()
binding=lambda p:dict(path=str(p),sha256=sha(p))
prior_path=R/'runs/direct-target-gpu-lifetime-20261007-v1/appworld/appworld-dt/source.json'
assert sha(prior_path)=='942c2d686a701317e8441f5b299dd86dfb55deb7356c7427e62bc4cbdcd36488'
prior=read(prior_path)
active=read(R/'active-training.json')
old=next(j for j in active['jobs'] if j['task']=='AppWorld')
assert (old['pid'],old['observed_process_created_unix'])==(1468126,1791357190.77)
assert old['source_receipt']==str(prior_path)
submit_path=O/'setup/submit_prepared_direct_targets.py'
assert sha(submit_path)=='8066517300717a55ca340f70a284cfc0051091c77291a72ba1fb029c0518a8f4'
spec=importlib.util.spec_from_file_location('prepared_submit_owner',submit_path)
submit=importlib.util.module_from_spec(spec);spec.loader.exec_module(submit)
submit.check_previous_owner(active,'AppWorld')
try:
 p=psutil.Process(1468126)
 previous=dict(pid=p.pid,birth=p.create_time(),status=p.status())
except psutil.NoSuchProcess:
 previous=dict(pid=1468126,birth=1791357190.77,status='NoSuchProcess')

base=O/'appworld';dt=O/'deltatrace';old_dt=P(prior['dt_root'])
assert not base.exists() and not dt.exists(),'Preserve previous preparation attempts'
for directory,mapping in [(prior['entry'],prior['entry_sha256']),
                          (prior['verl_root'],prior['owner_head_sha256']),
                          (old_dt,prior['dt_source_sha256'])]:
 for name,digest in mapping.items():assert sha(P(directory)/name)==digest,(directory,name)
for path,digest in prior['source_bindings'].items():assert sha(path)==digest,path
replacements={'clean/qwen35/qwen35_dense_finite_runner.py': ('ba639b2876827cc250f4806af278065181ef78d28c8b22da97377c34b9e168d7', '628006b637516f8d62e95583a9eb51fe9038ea7931798e2c1f42c28e154cf24f'), 'clean/qwen35/qwen35_decoder_finite.py': ('047c38e6b180adb350083ff370693ed20b95e2df82e4a78cd59038bc0803a197', '1c58c33c3d9120c846f571a5bfac80c07362e2e8247305cc38db5303d90d3234')}
assert prior['dt_source_sha256']['clean/qwen35/qwen35_answer_finite.py']=='1e20956a774d34917f2a31290945830bf757062bd8006881c21883e20bc3541e'
assert prior['entry_sha256']['reward_readout.py']=='7900a369a2d716b60e4b3cc24de13919f3c61eaeef6d4fa4511eecb088f67b27'
for name,(before,after) in replacements.items():
 assert prior['dt_source_sha256'][name]==before,name
 assert sha(O/'setup'/P(name).name)==after,name

# Same linked-tree transport pattern as the frozen prepare_direct_entries.
# Directories remain local; file symlinks retain the exact accepted owners.
dt.mkdir(parents=True)
for original in old_dt.rglob('*'):
 relative=original.relative_to(old_dt)
 if '__pycache__' in relative.parts or '.git' in relative.parts:continue
 path=dt/relative
 if original.is_dir() and original.is_symlink():
  path.symlink_to(original.resolve(),target_is_directory=True)
 elif original.is_dir():path.mkdir(exist_ok=True,parents=True)
 elif original.is_file():
  path.parent.mkdir(exist_ok=True,parents=True);path.symlink_to(original.resolve())
for name,(before,after) in replacements.items():
 replacement=dt/name
 assert replacement.is_symlink()
 replacement.unlink();replacement.write_bytes((O/'setup'/P(name).name).read_bytes())
 assert sha(replacement)==after,name
dt_hashes={name:sha(dt/name) for name in prior['dt_source_sha256']}
changed=[name for name,digest in dt_hashes.items() if digest!=prior['dt_source_sha256'][name]]
assert set(changed)==set(replacements) and len(changed)==2,changed
assert sha(dt/'clean/qwen35/qwen35_answer_finite.py')=='1e20956a774d34917f2a31290945830bf757062bd8006881c21883e20bc3541e'
assert sha(dt/'clean/qwen35/qwen35_gdn_finite.py')=='448ef32c773f8cda20be56c75fc181944e7efd18db69061928dedeed6d73ab72'
assert sha(P(prior['entry'])/'reward_readout.py')=='7900a369a2d716b60e4b3cc24de13919f3c61eaeef6d4fa4511eecb088f67b27'
base.mkdir()
output=R/'runs/direct-target-mlp-token-chunk-20261007-v1/appworld/appworld-dt'
assert not output.exists(),'No working output is created during preparation'
env=copy.deepcopy(prior['environment'])
def remap(path):
 value=str(path);prefix=str(old_dt)
 return str(dt)+value[len(prefix):] if value==prefix or value.startswith(prefix+'/') else value
env['DT_ROOT']=str(dt)
env['PYTHONPATH']=':'.join(remap(path) for path in prior['pythonpath'].split(':'))
assert env['DT_ENTRY_ROOT']==prior['entry'] and env['VERL_ROOT']==prior['verl_root']
assert env['LOOP_ROOT']==prior['loop_root']
assert env['CUDA_VISIBLE_DEVICES']=='4,5' and env['DT_MAX_LENGTH']=='32768'
cpu=dict(os.environ,**env);cpu.update(CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='-1')
cpu.pop('RAY_ADDRESS',None)
inspect_code=r"""
import hashlib,importlib,inspect,json,os,pathlib,resource,sys,time
P=pathlib.Path;out=P(sys.argv[1]);output=P(sys.argv[2]);dt=P(os.environ['DT_ROOT'])
import torch
assert not torch.cuda.is_initialized()
import owner_runtime_options as runtime
launcher=importlib.import_module('launch_appworld_native')
options,sampling=launcher.options_for(output)
baseline_options,baseline_sampling=launcher.options_for(P(sys.argv[3]))
assert sampling==baseline_sampling
modules={name:importlib.import_module(name) for name in ['counterfactual','reward_readout','dt_training_batch',
 'owner_trajectory_batch','deltatrace_rollout','executed_target_spans','verl.trainer.ppo.ray_trainer',
 'verl.workers.actor.dp_actor','verl.utils.torch_functional','loop_owner_rollout']}
import site
if os.environ.get('LOOP_EXTRAS'):site.addsitedir(os.environ['LOOP_EXTRAS'])
modules['official_parser']=importlib.import_module('phi_agents.utils.appworld')
modules['launcher']=launcher
# Producer.__init__ imports use these exact recorded paths. Reuse only its
# import path setup, without constructing its model/tokenizer/runner.
qwen=json.loads(P(os.environ['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
official=P(os.environ.get('DT_OFFICIAL_ROOT') or qwen['official_root'])
sys.path[:0]=[str(dt),str(official),str(dt/'clean/qwen35')]
sys.path.append(qwen['ft_extension_root'])
for name in ['qwen35_answer_finite','qwen35_dense_finite_runner','qwen35_gdn_finite','qwen35_decoder_finite']:
 modules[name]=importlib.import_module(name)
def source(module):
 path=P(inspect.getfile(module))
 return dict(path=str(path),resolved_path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
imports={name:source(module) for name,module in modules.items()}
assert imports['qwen35_answer_finite']['path']==str(dt/'clean/qwen35/qwen35_answer_finite.py')
assert imports['qwen35_answer_finite']['sha256']=='1e20956a774d34917f2a31290945830bf757062bd8006881c21883e20bc3541e'
assert imports['qwen35_dense_finite_runner']['path']==str(dt/'clean/qwen35/qwen35_dense_finite_runner.py')
assert imports['qwen35_dense_finite_runner']['sha256']=='628006b637516f8d62e95583a9eb51fe9038ea7931798e2c1f42c28e154cf24f'
assert imports['qwen35_decoder_finite']['path']==str(dt/'clean/qwen35/qwen35_decoder_finite.py')
assert imports['qwen35_decoder_finite']['resolved_path']==str(dt/'clean/qwen35/qwen35_decoder_finite.py')
assert imports['qwen35_decoder_finite']['sha256']=='1c58c33c3d9120c846f571a5bfac80c07362e2e8247305cc38db5303d90d3234'
assert imports['qwen35_gdn_finite']['path']==str(dt/'clean/qwen35/qwen35_gdn_finite.py')
assert imports['qwen35_gdn_finite']['sha256']=='448ef32c773f8cda20be56c75fc181944e7efd18db69061928dedeed6d73ab72'
assert imports['reward_readout']['sha256']=='7900a369a2d716b60e4b3cc24de13919f3c61eaeef6d4fa4511eecb088f67b27'
assert imports['verl.workers.actor.dp_actor']['sha256']=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
assert not torch.cuda.is_initialized() and not torch.distributed.is_initialized()
record=dict(status='CPU_exact_owner_import_and_original_options_only',observed_unix=time.time(),
 options=options,sampling=sampling,baseline_options=baseline_options,baseline_sampling=baseline_sampling,imports=imports,owner_command=runtime.owner_command(options),
 cuda_initialized=False,distributed_initialized=False,
 max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
 scope='Original preparation CPU imports/configuration, plus explicit candidate runner/decoder and unchanged head/GDN imports using original producer path setup. No producer instance, model, DT, episode, optimizer, checkpoint or GPU call.')
(out/'cpu-imports.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(imports=len(imports),max_rss_bytes=record['max_rss_bytes'])))
"""
with (base/'cpu-imports.log').open('wb') as log:
 result=subprocess.run([env['VENV_PYTHON'],'-c',inspect_code,str(base),str(output),old['output']],
                       cwd=prior['entry'],env=cpu,stdout=log,stderr=subprocess.STDOUT,timeout=120)
assert result.returncode==0,str(base/'cpu-imports.log')
inspection=read(base/'cpu-imports.json');options=inspection['options']
old_launch=P(old['output'])/'launch.json';old_options=read(old_launch)['options']
assert inspection['baseline_options']==old_options
assert inspection['sampling']==inspection['baseline_sampling']
prior_imports=prior['actual_CPU_imports']
for name,before in prior_imports.items():
 after=inspection['imports'][name]
 if name in ('qwen35_dense_finite_runner','qwen35_decoder_finite'):continue
 assert after['sha256']==before['sha256'],name
 assert after['path']==remap(before['path']),name
 assert after['resolved_path']==before['resolved_path'],name
differences={key:dict(before=old_options.get(key),after=options.get(key))
             for key in old_options.keys()|options.keys() if old_options.get(key)!=options.get(key)}
allowed={'trainer.default_local_dir','trainer.rollout_data_dir','trainer.validation_data_dir',
         '+ray_init.runtime_env.env_vars.DT_WORKER_VISIBILITY_DIR'}
assert set(differences)<=allowed,differences
for key,change in differences.items():
 assert change['after']==change['before'].replace(old['output'],str(output)),key
assert env['DT_ENVIRONMENT_JSON']==prior['environment']['DT_ENVIRONMENT_JSON']
environment_differences={key for key in env.keys()|prior['environment'].keys()
 if env.get(key)!=prior['environment'].get(key)}
assert environment_differences<= {'DT_ROOT','PYTHONPATH'},environment_differences
assert options['trainer.resume_mode']=='disable' and 'trainer.resume_from_path' not in options
for key,value in {'actor_rollout_ref.model.lora_rank':8,'actor_rollout_ref.model.lora_alpha':16,
                  'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':4,
                  'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu':4}.items():
 assert options[key]==value,key
bindings={path:digest for path,digest in prior['source_bindings'].items()
          if not (path==str(old_dt) or path.startswith(str(old_dt)+'/'))}
bindings.update({str(dt/name):digest for name,digest in dt_hashes.items()})
bindings.update({str(O/'setup'/name):sha(O/'setup'/name) for name in
                 ('qwen35_dense_finite_runner.py','qwen35_decoder_finite.py','prepare_appworld_mlp_token_chunk.py','submit_prepared_direct_targets.py')})
source=dict(prior,dt_root=str(dt),dt_source_sha256=dt_hashes,source_bindings=bindings,
            environment=env,pythonpath=env['PYTHONPATH'],startup_options=options,
            unix=time.time(),prepared_only=True,checkpoint_restore_requested=False,resume_mode='disable',
            runtime_verification_status='Prepared CPU imports only; new model/DT/update not executed',
            local_patch_commit='d5b879d7acd49a5e7ba7550a52cb53a0f993d0d5',baseline_source=binding(prior_path),
            actual_CPU_imports=inspection['imports'],official_configuration_differences=differences,
            source_scope='Real executed joint action targets; isolated finite-MLP token chunking at2048; original B4/LoRA/task/PPO/FA/FLA; fresh base without checkpoint',
            mlp_token_chunk_preparation=dict(changed_dt_files=changed,mlp_token_chunk_size=2048,previous_driver=previous,
                baseline_launch=binding(old_launch),candidate_files={name:binding(dt/name) for name in replacements},
                retained_head=binding(dt/'clean/qwen35/qwen35_answer_finite.py'),
                retained_gdn=binding(dt/'clean/qwen35/qwen35_gdn_finite.py'),
                retained_readout=binding(P(prior['entry'])/'reward_readout.py'),
                setup_owner=binding(O/'setup/prepare_appworld_mlp_token_chunk.py'),
                original_prepare_owner=dict(path='D:/Users/Administrator/Documents/ChatGPT/DeltaTrace/research/temporary/rl_upstream_alignment_20260929/direct-target-head-memory-20261007/v2/prepare_appworld_head_memory.py',sha256='aaaaddd13aef5b37c4b3d7d0cadf5807fc14f5fe6356d5352e554a8a06608a26'),
                status='prepared_only_no_submission_or_numerical_acceptance'))
if 'resource_environment' in source:
 resource_env=copy.deepcopy(source['resource_environment'])
 if 'DT_ROOT' in resource_env:resource_env['DT_ROOT']=str(dt)
 if 'PYTHONPATH' in resource_env:resource_env['PYTHONPATH']=':'.join(remap(p) for p in resource_env['PYTHONPATH'].split(':'))
 source['resource_environment']=resource_env
argv=[str(output) if value==old['output'] else value for value in old['argv']]
plan=dict(task='AppWorld',devices=[4,5],argv=argv,
          working_directory=str(output),environment=env,entry=prior['entry'],
          verl_root=prior['verl_root'],dt_root=str(dt),budget=old['budget'],resume_mode='disable',
          checkpoint_restore_requested=False)
for name,value in [('source-template.json',source),('launch-plan.json',plan),('run-env.json',env)]:
 path=base/name;path.write_text(json.dumps(value,indent=2)+'\n');os.chmod(path,0o600)
receipt=dict(status='prepared_CPU_imports_passed_not_submitted',task='AppWorld',devices=[4,5],
             observed_unix=time.time(),repository_commit='d5b879d7acd49a5e7ba7550a52cb53a0f993d0d5',source_template=binding(base/'source-template.json'),
             launch_plan=binding(base/'launch-plan.json'),CPU_imports=binding(base/'cpu-imports.json'),
             run_environment=binding(base/'run-env.json'),source_bindings=bindings,
             configuration_differences=differences,entry=prior['entry'],task_owner=prior['loop_root'],
             dt_root=str(dt),previous_driver=previous,
             scope='Prepared-only linked DT tree; exactly runner/decoder replaced; CPU imports/configuration only; no stop/submit, training/default source mutation, model/DT/GPU, environment episode, optimizer, checkpoint or new numerical acceptance.')
(base/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
checked=submit.check_prepared(O,'AppWorld')
print(json.dumps(dict(status='prepared_CPU_imports_passed_not_submitted',prepared=binding(base/'prepared.json'),
    source_template=binding(base/'source-template.json'),launch_plan=binding(base/'launch-plan.json'),
    source_bindings_verified=checked['source_bindings_verified'],configuration_differences=differences,
    head=inspection['imports']['qwen35_answer_finite'],runner=inspection['imports']['qwen35_dense_finite_runner'],
    gdn=inspection['imports']['qwen35_gdn_finite'],decoder=inspection['imports']['qwen35_decoder_finite'],
    actor=inspection['imports']['verl.workers.actor.dp_actor'],previous_driver=previous,
    output=str(output),devices=[4,5])))

PY
