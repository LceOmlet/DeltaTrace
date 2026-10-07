"""Prepare an isolated answer-owner replacement around the exact AppWorld v3.

Setup/provenance only. This helper never submits/stops training or calls a
model, DT, environment, optimizer or checkpoint. Running it stages a new
candidate and performs CPU imports/configuration composition; it does not
change the accepted v3 entry, VERL, LOOP, DT tree or active manifests.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('stage', AUDIT / 'stage_environment_entry.py')
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)
ANSWER_SHA = '1e20956a774d34917f2a31290945830bf757062bd8006881c21883e20bc3541e'

CODE = r'''
from pathlib import Path
import copy,hashlib,importlib.util,json,os,psutil,subprocess,sys,time
P=Path;R=P(@ROOT@);O=P(@OUT@)
read=lambda p:json.loads(P(p).read_bytes())
sha=lambda p:hashlib.sha256(P(p).read_bytes()).hexdigest()
binding=lambda p:dict(path=str(p),sha256=sha(p))
prior_path=R/'runs/direct-action-target-20261007-v3/appworld/appworld-dt/source.json'
assert sha(prior_path)=='70ffdcfd05fccb94c0cec70e5b1d83d8728e3335bb9f1a4e6bbec9299f53847b'
prior=read(prior_path)
active=read(R/'active-training.json')
old=next(j for j in active['jobs'] if j['task']=='AppWorld')
assert (old['pid'],old['observed_process_created_unix'])==(167065,1791344807.34)
assert old['source_receipt']==str(prior_path)
submit_path=O/'setup/submit_prepared_direct_targets.py'
assert sha(submit_path)=='8066517300717a55ca340f70a284cfc0051091c77291a72ba1fb029c0518a8f4'
spec=importlib.util.spec_from_file_location('prepared_submit_owner',submit_path)
submit=importlib.util.module_from_spec(spec);spec.loader.exec_module(submit)
submit.check_previous_owner(active,'AppWorld')
try:
 p=psutil.Process(167065)
 previous=dict(pid=p.pid,birth=p.create_time(),status=p.status())
except psutil.NoSuchProcess:
 previous=dict(pid=167065,birth=1791344807.34,status='NoSuchProcess')

base=O/'appworld';dt=O/'deltatrace';old_dt=P(prior['dt_root'])
assert not base.exists() and not dt.exists(),'Preserve previous preparation attempts'
for directory,mapping in [(prior['entry'],prior['entry_sha256']),
                          (prior['verl_root'],prior['owner_head_sha256']),
                          (old_dt,prior['dt_source_sha256'])]:
 for name,digest in mapping.items():assert sha(P(directory)/name)==digest,(directory,name)
for path,digest in prior['source_bindings'].items():assert sha(path)==digest,path
answer_relative='clean/qwen35/qwen35_answer_finite.py'
assert prior['dt_source_sha256'][answer_relative]=='d47333ea68fb7a332e7d1dce7913c989d875ea262dfe49cfa4f20f7c35ebe03e'
assert sha(O/'setup/qwen35_answer_finite.py')=='@ANSWER_SHA@'

# Same linked-tree transport pattern as the frozen prepare_direct_entries.
# Directories remain local; file symlinks retain the exact accepted owners.
dt.mkdir(parents=True)
for original in old_dt.rglob('*'):
 relative=original.relative_to(old_dt)
 if '__pycache__' in relative.parts or '.git' in relative.parts:continue
 path=dt/relative
 if original.is_dir():path.mkdir(exist_ok=True,parents=True)
 elif original.is_file():
  path.parent.mkdir(exist_ok=True,parents=True);path.symlink_to(original.resolve())
answer=dt/answer_relative
assert answer.is_symlink()
answer.unlink();answer.write_bytes((O/'setup/qwen35_answer_finite.py').read_bytes())
dt_hashes={name:sha(dt/name) for name in prior['dt_source_sha256']}
changed=[name for name,digest in dt_hashes.items() if digest!=prior['dt_source_sha256'][name]]
assert changed==[answer_relative],changed
base.mkdir()
output=R/'runs/direct-target-head-memory-20261007-v1/appworld/appworld-dt'
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
for name in ['qwen35_answer_finite','qwen35_dense_finite_runner']:
 modules[name]=importlib.import_module(name)
def source(module):
 path=P(inspect.getfile(module))
 return dict(path=str(path),resolved_path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
imports={name:source(module) for name,module in modules.items()}
assert imports['qwen35_answer_finite']['path']==str(dt/'clean/qwen35/qwen35_answer_finite.py')
assert imports['qwen35_answer_finite']['sha256']=='@ANSWER_SHA@'
assert imports['qwen35_dense_finite_runner']['sha256']=='5f14bb3cdb491e4d5b3b531607e00936bae76e055e2286ed60e096c6c5d2e555'
assert imports['verl.workers.actor.dp_actor']['sha256']=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
assert not torch.cuda.is_initialized() and not torch.distributed.is_initialized()
record=dict(status='CPU_exact_owner_import_and_original_options_only',observed_unix=time.time(),
 options=options,sampling=sampling,imports=imports,owner_command=runtime.owner_command(options),
 cuda_initialized=False,distributed_initialized=False,
 max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
 scope='Original preparation CPU imports/configuration, plus explicit answer/runner imports using original producer path setup. No producer instance, model, DT, episode, optimizer, checkpoint or GPU call.')
(out/'cpu-imports.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(imports=len(imports),max_rss_bytes=record['max_rss_bytes'])))
"""
with (base/'cpu-imports.log').open('wb') as log:
 result=subprocess.run([env['VENV_PYTHON'],'-c',inspect_code,str(base),str(output)],
                       cwd=prior['entry'],env=cpu,stdout=log,stderr=subprocess.STDOUT,timeout=120)
assert result.returncode==0,str(base/'cpu-imports.log')
inspection=read(base/'cpu-imports.json');options=inspection['options']
old_launch=P(old['output'])/'launch.json';old_options=read(old_launch)['options']
differences={key:dict(before=old_options.get(key),after=options.get(key))
             for key in old_options.keys()|options.keys() if old_options.get(key)!=options.get(key)}
allowed={'trainer.default_local_dir','trainer.rollout_data_dir','trainer.validation_data_dir',
         '+ray_init.runtime_env.env_vars.DT_WORKER_VISIBILITY_DIR'}
assert set(differences)<=allowed,differences
assert options['trainer.resume_mode']=='disable' and 'trainer.resume_from_path' not in options
for key,value in {'actor_rollout_ref.model.lora_rank':8,'actor_rollout_ref.model.lora_alpha':16,
                  'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu':4,
                  'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu':4}.items():
 assert options[key]==value,key
bindings={path:digest for path,digest in prior['source_bindings'].items()
          if not (path==str(old_dt) or path.startswith(str(old_dt)+'/'))}
bindings.update({str(dt/name):digest for name,digest in dt_hashes.items()})
bindings.update({str(O/'setup'/name):sha(O/'setup'/name) for name in
                 ('qwen35_answer_finite.py','prepare_appworld_head_memory.py','submit_prepared_direct_targets.py')})
source=dict(prior,dt_root=str(dt),dt_source_sha256=dt_hashes,source_bindings=bindings,
            environment=env,pythonpath=env['PYTHONPATH'],startup_options=options,
            unix=time.time(),prepared_only=True,checkpoint_restore_requested=False,resume_mode='disable',
            runtime_verification_status='Prepared CPU imports only; new model/DT/update not executed',
            local_patch_commit='@COMMIT@',baseline_source=binding(prior_path),
            actual_CPU_imports=inspection['imports'],official_configuration_differences=differences,
            source_scope='Real executed joint action targets; isolated full-vocabulary target-row temporary tiling; original B4/LoRA/task/PPO/FA/FLA; fresh base without checkpoint',
            head_memory_preparation=dict(changed_dt_files=changed,previous_driver=previous,
                baseline_launch=binding(old_launch),candidate_answer=binding(answer),
                setup_owner=binding(O/'setup/prepare_appworld_head_memory.py'),
                original_prepare_owner=dict(path='@ORIGINAL_PREPARE_PATH@',sha256='@ORIGINAL_PREPARE_SHA@'),
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
             observed_unix=time.time(),repository_commit='@COMMIT@',source_template=binding(base/'source-template.json'),
             launch_plan=binding(base/'launch-plan.json'),CPU_imports=binding(base/'cpu-imports.json'),
             run_environment=binding(base/'run-env.json'),source_bindings=bindings,
             configuration_differences=differences,entry=prior['entry'],task_owner=prior['loop_root'],
             dt_root=str(dt),previous_driver=previous,
             scope='Preparation and CPU imports only; no stop/submit, training/default source mutation, model/DT/GPU, environment episode, optimizer, checkpoint or new numerical acceptance.')
(base/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
checked=submit.check_prepared(O,'AppWorld')
print(json.dumps(dict(status='prepared_CPU_imports_passed_not_submitted',prepared=binding(base/'prepared.json'),
    source_template=binding(base/'source-template.json'),launch_plan=binding(base/'launch-plan.json'),
    source_bindings_verified=checked['source_bindings_verified'],configuration_differences=differences,
    answer=inspection['imports']['qwen35_answer_finite'],runner=inspection['imports']['qwen35_dense_finite_runner'],
    actor=inspection['imports']['verl.workers.actor.dp_actor'],previous_driver=previous,
    output=str(output),devices=[4,5])))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.commit):
        parser.error('--commit must be the actual 40-character code commit')
    candidate = HERE / 'candidate/qwen35_answer_finite.py'
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == ANSWER_SHA
    submit = AUDIT / 'direct-target-semantics-20261007/submit_prepared_direct_targets.py'
    assert hashlib.sha256(submit.read_bytes()).hexdigest() == '8066517300717a55ca340f70a284cfc0051091c77291a72ba1fb029c0518a8f4'
    original = AUDIT / 'direct-action-target-20261007/prepare_direct_entries.py'
    out = stage.ROOT + '/candidates/direct-target-head-memory-20261007-v1'
    bundle = HERE / 'setup-source.tar'
    with tarfile.open(bundle, 'w') as archive:
        for path, name in [(candidate, 'qwen35_answer_finite.py'),
                           (Path(__file__), 'prepare_appworld_head_memory.py'),
                           (submit, 'submit_prepared_direct_targets.py')]:
            archive.add(path, arcname=name)
    code = CODE.replace('@ROOT@', repr(stage.ROOT)).replace('@OUT@', repr(out))
    code = code.replace('@ANSWER_SHA@', ANSWER_SHA).replace('@COMMIT@', args.commit)
    code = code.replace('@ORIGINAL_PREPARE_PATH@', str(original.resolve()).replace('\\', '/'))
    code = code.replace('@ORIGINAL_PREPARE_SHA@', hashlib.sha256(original.read_bytes()).hexdigest())
    ast.parse(code)
    script = 'set -eu\nsource ' + stage.ENTRY + '/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n' + code + '\nPY\n'
    (HERE / 'prepare-command.sh').write_text(script, encoding='utf-8')
    subprocess.run(stage.SSH + ['mkdir', '-p', out + '/setup'], check=True)
    subprocess.run(stage.SCP + [str(bundle), stage.SSH[-1] + ':' + out + '/setup-source.tar'], check=True)
    subprocess.run(stage.SSH + ['tar', '-xf', out + '/setup-source.tar', '-C', out + '/setup'], check=True)
    result = subprocess.run(stage.SSH + ['bash', '-s'], input=script.encode(), capture_output=True, timeout=240)
    (HERE / 'prepare.stdout.txt').write_bytes(result.stdout)
    (HERE / 'prepare.stderr.txt').write_bytes(result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()


if __name__ == '__main__':
    main()
