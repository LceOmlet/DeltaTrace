"""Prepare one isolated reward-readout overlay around the current owner tree.

Setup only: reuse the existing CPU import/config inspection and submission
owners. No model, task episode, optimizer, checkpoint or GPU call is made.
"""
import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path
import re
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('existing_stage', AUDIT/'stage_environment_entry.py')
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)

CODE = r'''
from pathlib import Path
import copy,hashlib,importlib.util,json,os,subprocess,sys,time
P=Path;R=P(@ROOT@);O=P(@OUT@)
read=lambda p:json.loads(P(p).read_bytes())
sha=lambda p:hashlib.sha256(P(p).read_bytes()).hexdigest()
binding=lambda p:dict(path=str(p),sha256=sha(p))
prior_path=R/'runs/direct-target-head-memory-20261007-v2/appworld/appworld-dt/source.json'
assert sha(prior_path)=='42bb0eb979f76527f545afd83f934cc46add4c042bf63a16dd3f7304d25789bc'
prior=read(prior_path);active=read(R/'active-training.json')
old=next(j for j in active['jobs'] if j['task']=='AppWorld')
assert (old['pid'],old['observed_process_created_unix'])==(670069,1791349552.74)
assert old['source_receipt']==str(prior_path)
submit_path=O/'setup/submit_prepared_direct_targets.py'
assert sha(submit_path)=='8066517300717a55ca340f70a284cfc0051091c77291a72ba1fb029c0518a8f4'
spec=importlib.util.spec_from_file_location('prepared_submit_owner',submit_path)
submit=importlib.util.module_from_spec(spec);spec.loader.exec_module(submit)
submit.check_previous_owner(active,'AppWorld')
for directory,mapping in [(prior['entry'],prior['entry_sha256']),
                          (prior['verl_root'],prior['owner_head_sha256']),
                          (prior['dt_root'],prior['dt_source_sha256'])]:
 for name,digest in mapping.items():assert sha(P(directory)/name)==digest,(directory,name)
for path,digest in prior['source_bindings'].items():assert sha(path)==digest,path
base=O/'appworld';entry=base/'entry';old_entry=P(prior['entry'])
assert not base.exists(),'Preserve existing preparation attempts'
entry.mkdir(parents=True)
for original in old_entry.iterdir():
 if original.is_file():(entry/original.name).symlink_to(original.resolve())
candidate=entry/'reward_readout.py';candidate.unlink()
candidate.write_bytes((O/'setup/reward_readout.py').read_bytes())
assert sha(candidate)=='@READOUT_SHA@'
entry_hashes={name:sha(entry/name) for name in prior['entry_sha256']}
changed=[name for name in entry_hashes if entry_hashes[name]!=prior['entry_sha256'][name]]
assert changed==['reward_readout.py'],changed
output=R/'runs/direct-target-causal-prefix-20261007-@VERSION@/appworld/appworld-dt'
assert not output.exists()
def remap(path):
 value=str(path);prefix=str(old_entry)
 return str(entry)+value[len(prefix):] if value==prefix or value.startswith(prefix+'/') else value
env=copy.deepcopy(prior['environment']);env['DT_ENTRY_ROOT']=str(entry)
env['PYTHONPATH']=':'.join(remap(p) for p in prior['pythonpath'].split(':'))
assert env['DT_ROOT']==prior['dt_root'] and env['VERL_ROOT']==prior['verl_root']
assert env['LOOP_ROOT']==prior['loop_root'] and env['CUDA_VISIBLE_DEVICES']=='4,5'
assert env['DT_MAX_LENGTH']=='32768'
cpu=dict(os.environ,**env);cpu.update(CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='-1')
cpu.pop('RAY_ADDRESS',None)
inspect_code=@INSPECT_CODE@
with (base/'cpu-imports.log').open('wb') as log:
 result=subprocess.run([env['VENV_PYTHON'],'-c',inspect_code,str(base),str(output)],
                       cwd=entry,env=cpu,stdout=log,stderr=subprocess.STDOUT,timeout=120)
assert result.returncode==0,str(base/'cpu-imports.log')
inspection=read(base/'cpu-imports.json');options=inspection['options']
assert inspection['imports']['reward_readout']['path']==str(candidate)
assert inspection['imports']['reward_readout']['sha256']=='@READOUT_SHA@'
# Reuse the existing CPU contract suite against these exact frozen imports.
# Its owner selection-class path is supplied by a symlink, not a copied owner.
checks=O/'checks/experiments/rl';checks.mkdir(parents=True)
(O/'checks/deltatrace').symlink_to(prior['dt_root'],target_is_directory=True)
(checks/'test_reward_readout.py').write_bytes((O/'setup/test_reward_readout.py').read_bytes())
with (base/'cpu-tests.log').open('wb') as log:
 tests=subprocess.run([env['VENV_PYTHON'],'-m','pytest','-q',str(checks/'test_reward_readout.py')],
                      cwd=entry,env=cpu,stdout=log,stderr=subprocess.STDOUT,timeout=120)
assert tests.returncode==0,str(base/'cpu-tests.log')
old_launch=P(old['output'])/'launch.json';old_options=read(old_launch)['options']
differences={key:dict(before=old_options.get(key),after=options.get(key))
             for key in old_options.keys()|options.keys() if old_options.get(key)!=options.get(key)}
allowed={'data.custom_cls.path','trainer.default_local_dir','trainer.rollout_data_dir',
         'trainer.validation_data_dir','+ray_init.runtime_env.env_vars.DT_WORKER_VISIBILITY_DIR'}
assert set(differences)<=allowed,differences
assert options['data.custom_cls.path']==str(entry/'loop_iteration_dataset.py')
bindings={path:digest for path,digest in prior['source_bindings'].items()
          if not (path==str(old_entry) or path.startswith(str(old_entry)+'/'))}
bindings.update({str(entry/name):digest for name,digest in entry_hashes.items()})
bindings.update({str(O/'setup'/name):sha(O/'setup'/name) for name in
                 ('reward_readout.py','test_reward_readout.py','prepare_appworld_causal_prefix.py',
                  'submit_prepared_direct_targets.py')})
source=dict(prior,entry=str(entry),entry_sha256=entry_hashes,source_bindings=bindings,
 environment=env,pythonpath=env['PYTHONPATH'],startup_options=options,unix=time.time(),prepared_only=True,
 checkpoint_restore_requested=False,resume_mode='disable',local_patch_commit='@COMMIT@',
 baseline_source=binding(prior_path),actual_CPU_imports=inspection['imports'],
 official_configuration_differences=differences,
 runtime_verification_status='Prepared CPU imports and packing contracts only; full joint DT/update not executed',
 source_scope='Original full trajectory/reward/scatter; compute causal prefix through last real target only; unchanged head/VERL/LOOP/task/PPO/B4/LoRA; fresh base',
 causal_prefix_preparation=dict(changed_entry_files=changed,prior_driver=old['pid'],
   baseline_launch=binding(old_launch),CPU_tests=binding(base/'cpu-tests.log'),
   owner_CPU_inspection=dict(path='@INSPECT_OWNER@',sha256='@INSPECT_OWNER_SHA@'),
   scope='No full failed-trajectory replay; CPU packing tests are not DT numerical/capacity evidence'))
if 'resource_environment' in source:
 resource_env=copy.deepcopy(source['resource_environment'])
 if 'DT_ENTRY_ROOT' in resource_env:resource_env['DT_ENTRY_ROOT']=str(entry)
 if 'PYTHONPATH' in resource_env:resource_env['PYTHONPATH']=':'.join(remap(p) for p in resource_env['PYTHONPATH'].split(':'))
 source['resource_environment']=resource_env
argv=[remap(value) if value!=old['output'] else str(output) for value in old['argv']]
plan=dict(task='AppWorld',devices=[4,5],argv=argv,working_directory=str(output),environment=env,
 entry=str(entry),verl_root=prior['verl_root'],dt_root=prior['dt_root'],budget=old['budget'],
 resume_mode='disable',checkpoint_restore_requested=False)
for name,value in [('source-template.json',source),('launch-plan.json',plan),('run-env.json',env)]:
 path=base/name;path.write_text(json.dumps(value,indent=2)+'\n');os.chmod(path,0o600)
receipt=dict(status='prepared_CPU_imports_passed_not_submitted',task='AppWorld',devices=[4,5],
 observed_unix=time.time(),repository_commit='@COMMIT@',source_template=binding(base/'source-template.json'),
 launch_plan=binding(base/'launch-plan.json'),CPU_imports=binding(base/'cpu-imports.json'),
 run_environment=binding(base/'run-env.json'),configuration_differences=differences,
 entry=str(entry),dt_root=prior['dt_root'],scope='Setup/CPU only; no submit or model/GPU/episode call')
(base/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
checked=submit.check_prepared(O,'AppWorld')
print(json.dumps(dict(status=receipt['status'],prepared=binding(base/'prepared.json'),
 source_template=binding(base/'source-template.json'),configuration_differences=differences,
 source_bindings_verified=checked['source_bindings_verified'],imports=inspection['imports'],
 CPU_tests=binding(base/'cpu-tests.log'),output=str(output),devices=[4,5])))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--version', choices=('v1', 'v2'), default='v1')
    args = parser.parse_args()
    assert re.fullmatch(r'[0-9a-f]{40}', args.commit)
    owner = AUDIT/'direct-target-head-memory-20261007/v2/prepare_appworld_head_memory.py'
    owner_tree = ast.parse(owner.read_bytes())
    owner_code = ast.literal_eval(next(n.value for n in owner_tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CODE' for t in n.targets)))
    inspect_code = ast.literal_eval(next(n.value for n in ast.walk(ast.parse(owner_code.replace('@ROOT@', "''").replace('@OUT@', "''")))
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inspect_code' for t in n.targets)))
    inspect_code = inspect_code.replace('@ANSWER_SHA@', '1e20956a774d34917f2a31290945830bf757062bd8006881c21883e20bc3541e')
    submit = AUDIT/'direct-target-semantics-20261007/submit_prepared_direct_targets.py'
    readout = stage.REPO/'experiments/rl/reward_readout.py'
    tests = stage.REPO/'experiments/rl/test_reward_readout.py'
    results = HERE.parent/args.version
    results.mkdir(parents=True, exist_ok=True)
    out = stage.ROOT+'/candidates/direct-target-causal-prefix-20261007-'+args.version
    bundle = results/'setup-source.tar'
    with tarfile.open(bundle, 'w') as archive:
        for path in (readout, tests, Path(__file__), submit):
            archive.add(path, arcname=path.name)
    replacements = {'@ROOT@':repr(stage.ROOT), '@OUT@':repr(out), '@COMMIT@':args.commit,
        '@VERSION@':args.version,
        '@READOUT_SHA@':hashlib.sha256(readout.read_bytes()).hexdigest(),
        '@INSPECT_CODE@':repr(inspect_code), '@INSPECT_OWNER@':str(owner.resolve()).replace('\\','/'),
        '@INSPECT_OWNER_SHA@':hashlib.sha256(owner.read_bytes()).hexdigest()}
    code = CODE
    for key,value in replacements.items(): code=code.replace(key,value)
    ast.parse(code)
    script = 'set -eu\nsource '+stage.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
    (results/'prepare-command.sh').write_text(script, encoding='utf-8')
    subprocess.run(stage.SSH+['mkdir','-p',out+'/setup'],check=True)
    subprocess.run(stage.SCP+[str(bundle),stage.SSH[-1]+':'+out+'/setup-source.tar'],check=True)
    subprocess.run(stage.SSH+['tar','-xf',out+'/setup-source.tar','-C',out+'/setup'],check=True)
    result = subprocess.run(stage.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=300)
    (results/'prepare.stdout.json').write_bytes(result.stdout)
    (results/'prepare.stderr.txt').write_bytes(result.stderr)
    print(result.stdout.decode(errors='replace'))
    print(result.stderr.decode(errors='replace'))
    result.check_returncode()


if __name__ == '__main__':
    main()
