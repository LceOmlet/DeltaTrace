"""Freeze new credit adapters around the stopped, verified owner releases.

Preparation only. Original task launchers compose their configurations; no
model, environment episode, optimizer, checkpoint or GPU is started here.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import tarfile
import argparse

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
spec = importlib.util.spec_from_file_location('stage', AUDIT/'stage_environment_entry.py')
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)

CODE = r'''
from pathlib import Path
import ast,copy,hashlib,json,os,subprocess,sys,time
P=Path; R=P(@ROOT@); O=P(@OUT@)
read=lambda p:json.loads(P(p).read_bytes())
sha=lambda p:hashlib.sha256(P(p).read_bytes()).hexdigest()
binding=lambda p:dict(path=str(p),sha256=sha(p))
active=read(R/'active-training.json')
overlay=O/'overlay'
sys.path.insert(0,str(overlay))
from patch_executed_payload_metadata import textcraft_parser,loop_extractor

def linked_tree(source,destination):
    source,destination=P(source),P(destination)
    destination.mkdir(parents=True)
    for original in source.rglob('*'):
        relative=original.relative_to(source)
        if '__pycache__' in relative.parts or '.git' in relative.parts:continue
        path=destination/relative
        if original.is_dir():path.mkdir(exist_ok=True,parents=True)
        elif original.is_file():
            path.parent.mkdir(exist_ok=True,parents=True);path.symlink_to(original.resolve())

prepared=[]
for task in ('TextCraft','AppWorld'):
    old=next(j for j in active['jobs'] if j['task']==task)
    expected_status = ('stopped_direct_target_dtype_interface' if task=='AppWorld' and O.name.endswith('-v3')
                       else 'stopped_auxiliary_label_target_semantics')
    assert old['status']==expected_status,(task,old['status'])
    prior=read(old['source_receipt'])
    expected={'TextCraft':'5faa6e2d','AppWorld':('3f34b191' if O.name.endswith('-v3') else 'c83b96de')}[task]
    assert sha(old['source_receipt']).startswith(expected),(task,sha(old['source_receipt']))
    base=O/task.lower();entry=base/'entry';owner=base/'verl'
    assert not base.exists(),'Preserve previous preparation attempts'
    entry.mkdir(parents=True)
    for root,mapping in [(old['entry'],prior['entry_sha256']),
                         (old['verl_root'],prior['owner_head_sha256']),
                         (old['dt_root'],prior['dt_source_sha256'])]:
        for name,digest in mapping.items():assert sha(P(root)/name)==digest,(root,name)
    for original in P(old['entry']).iterdir():
        if original.is_file():(entry/original.name).write_bytes(original.read_bytes())
    owner.mkdir()
    for name in prior['owner_head_sha256']:
        path=owner/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.symlink_to((P(old['verl_root'])/name).resolve())
    for name in ['counterfactual.py','dt_training_batch.py','owner_trajectory_batch.py',
                 'reward_readout.py','executed_target_spans.py','deltatrace_rollout.py']:
        (entry/name).write_bytes((overlay/name).read_bytes())
    if task=='TextCraft':
        env=dict(prior['environment'])
        (entry/'textcraft_owner_rollout.py').write_bytes((overlay/'textcraft_owner_rollout.py').read_bytes())
        agentenv=R/'third_party/AgentGym-d014732d9fe39b975c368c03749bfd50950067f6/agentenv'
        task_owner=base/'agentenv-owner';linked_tree(agentenv,task_owner)
        parser=task_owner/'agentenv/envs/textcraft.py'
        original=parser.read_text();parser.unlink();parser.write_text(textcraft_parser(original))
        launcher=entry/'launch_textcraft_native.py'
        data=old['argv'][old['argv'].index('--data')+1]
        prefix=[str(task_owner)]
    else:
        prep=read(prior['formal_preparation']['path'])
        assert sha(prior['formal_preparation']['path'])==prior['formal_preparation']['sha256']
        env=read(prep['run_environment']['path'])
        assert sha(prep['run_environment']['path'])==prep['run_environment']['sha256']
        for name in ['loop_owner_rollout.py','loop_owner_worker.py']:
            (entry/name).write_bytes((overlay/name).read_bytes())
        task_owner=base/'loop-owner';linked_tree(old['loop_root'],task_owner)
        parser=task_owner/'phi_agents/utils/appworld.py'
        original=parser.read_text();parser.unlink();parser.write_text(loop_extractor(original))
        env['LOOP_ROOT']=str(task_owner)
        launcher=entry/'launch_appworld_native.py'
        data=None;prefix=[str(task_owner)]
    output=R/'runs'/O.name/task.lower()/(task.lower()+'-dt')
    env.update(VERL_ROOT=str(owner),DT_ENTRY_ROOT=str(entry),DT_ROOT=old['dt_root'],
               DT_TARGET_SEMANTICS='native_joint_action_target',
               CUDA_VISIBLE_DEVICES=','.join(map(str,old['devices'])))
    for name in ('MACA_VISIBLE_DEVICES','RAY_ADDRESS','RAY_TMPDIR','DT_PREFIX_CHECKPOINT'):
        env.pop(name,None)
    replacements={old['entry']:str(entry),old['verl_root']:str(owner)}
    if task=='AppWorld':replacements[old['loop_root']]=str(task_owner)
    inherited=[replacements.get(path,path) for path in prior['pythonpath'].split(':')]
    env['PYTHONPATH']=':'.join(dict.fromkeys([str(entry),str(owner),*prefix,*inherited]))
    env['DT_MAX_LENGTH']='32768'
    cpu=dict(os.environ,**env);cpu.update(CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='-1')
    inspect_code=r"""
import hashlib,importlib,inspect,json,os,pathlib,resource,sys,time
P=pathlib.Path;out=P(sys.argv[1]);task=sys.argv[2];output=P(sys.argv[3]);data=sys.argv[4]
import torch
assert not torch.cuda.is_initialized()
import owner_runtime_options as runtime
launcher=importlib.import_module('launch_textcraft_native' if task=='TextCraft' else 'launch_appworld_native')
options,sampling=(launcher.options_for(P(data),output) if task=='TextCraft' else launcher.options_for(output))
modules={name:importlib.import_module(name) for name in ['counterfactual','reward_readout','dt_training_batch',
 'owner_trajectory_batch','deltatrace_rollout','executed_target_spans','verl.trainer.ppo.ray_trainer',
 'verl.workers.actor.dp_actor','verl.utils.torch_functional']}
modules['task_bridge']=importlib.import_module('textcraft_owner_rollout' if task=='TextCraft' else 'loop_owner_rollout')
if task=='TextCraft':
 import site
 if os.environ.get('TEXTCRAFT_EXTRAS'):site.addsitedir(os.environ['TEXTCRAFT_EXTRAS'])
 modules['official_parser']=importlib.import_module('agentenv.envs.textcraft')
else:
 import site
 if os.environ.get('LOOP_EXTRAS'):site.addsitedir(os.environ['LOOP_EXTRAS'])
 modules['official_parser']=importlib.import_module('phi_agents.utils.appworld')
modules['launcher']=launcher
assert options['trainer.resume_mode']=='disable' and 'trainer.resume_from_path' not in options
assert options['actor_rollout_ref.model.lora_rank']==8 and options['actor_rollout_ref.model.lora_alpha']==16
assert options['actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu']==4
assert not torch.cuda.is_initialized()
source=lambda m:dict(path=inspect.getfile(m),sha256=hashlib.sha256(P(inspect.getfile(m)).read_bytes()).hexdigest())
record=dict(status='CPU_exact_owner_import_and_original_options_only',observed_unix=time.time(),
 options=options,sampling=sampling,imports={name:source(m) for name,m in modules.items()},
 owner_command=runtime.owner_command(options),cuda_initialized=False,
 max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
(out/'cpu-imports.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(task=task,imports=len(modules),max_rss_bytes=record['max_rss_bytes'])))
"""
    with (base/'cpu-imports.log').open('wb') as log:
        result=subprocess.run([env['VENV_PYTHON'],'-c',inspect_code,str(base),task,str(output),data or ''],
                              cwd=entry,env=cpu,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    assert result.returncode==0,(task,str(base/'cpu-imports.log'))
    inspection=read(base/'cpu-imports.json');options=inspection['options']
    old_options=read(P(old['output'])/'launch.json')['options']
    differences={key:dict(before=old_options.get(key),after=options.get(key))
                 for key in old_options.keys()|options.keys() if old_options.get(key)!=options.get(key)}
    allowed={'data.custom_cls.path','trainer.default_local_dir','trainer.rollout_data_dir',
             'trainer.validation_data_dir','+ray_init.runtime_env.env_vars.DT_WORKER_VISIBILITY_DIR','+env.loop'}
    assert set(differences)<=allowed,differences
    assert inspection['imports']['verl.workers.actor.dp_actor']['sha256']=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
    entry_sha={p.name:sha(p) for p in entry.iterdir() if p.is_file()}
    source=dict(prior,entry=str(entry),verl_root=str(owner),entry_sha256=entry_sha,
                source_scope='Real officially executed joint action targets; no synthetic reward labels; fresh base model',
                source_bindings={str(entry/name):digest for name,digest in entry_sha.items()},
                prepared_only=True,checkpoint_restore_requested=False,resume_mode='disable',
                target_semantics='native_joint_action_target',baseline_source=binding(old['source_receipt']),
                official_configuration_differences=differences,actual_CPU_imports=inspection['imports'],
                environment=env,pythonpath=env['PYTHONPATH'],startup_options=options,
                local_patch_commit='@COMMIT@')
    source['source_bindings'].update({str(owner/name):digest for name,digest in prior['owner_head_sha256'].items()})
    source['source_bindings'].update({str(P(old['dt_root'])/name):digest for name,digest in prior['dt_source_sha256'].items()})
    source['source_bindings'][str(parser)]=sha(parser)
    if task=='AppWorld':source['loop_root']=str(task_owner)
    argv=[env['VENV_PYTHON'],str(launcher)]
    if data:argv+=['--data',data]
    argv+=['--output',str(output)]
    plan=dict(task=task,devices=old['devices'],argv=argv,working_directory=str(output),
              environment=env,entry=str(entry),verl_root=str(owner),dt_root=old['dt_root'],
              budget=old['budget'],resume_mode='disable',checkpoint_restore_requested=False)
    for name,value in [('source-template.json',source),('launch-plan.json',plan)]:
        (base/name).write_text(json.dumps(value,indent=2)+'\n');os.chmod(base/name,0o600)
    receipt=dict(status='prepared_CPU_imports_passed_not_submitted',task=task,observed_unix=time.time(),
                 source_template=binding(base/'source-template.json'),launch_plan=binding(base/'launch-plan.json'),
                 CPU_imports=binding(base/'cpu-imports.json'),actual_parser=binding(parser),
                 source_bindings=source['source_bindings'],configuration_differences=differences,
                 task_owner=str(task_owner),entry=str(entry),devices=old['devices'],
                 scope='No GPU/model/environment episode/optimizer/checkpoint; numeric receipts separate')
    (base/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
    prepared.append(dict(task=task,prepared=binding(base/'prepared.json'),entry=str(entry),
                         output=str(output),devices=old['devices']))
(O/'prepared-pair.json').write_text(json.dumps(prepared,indent=2)+'\n')
print(json.dumps(prepared))
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--version', required=True, choices=('v2','v3'))
    args=parser.parse_args()
    out=stage.ROOT+'/candidates/direct-action-target-20261007-'+args.version
    files={name:stage.REPO/'experiments/rl'/name for name in ['counterfactual.py','dt_training_batch.py',
        'owner_trajectory_batch.py','reward_readout.py','executed_target_spans.py',
        'patch_executed_payload_metadata.py','textcraft_owner_rollout.py']}
    files['deltatrace_rollout.py']=HERE/'deltatrace_rollout.py'
    candidate=AUDIT/'direct-target-owner-mapping-20261007/implementation-v1/candidate/entry'
    for name in ('loop_owner_rollout.py','loop_owner_worker.py'):files[name]=candidate/name
    bundle=HERE/('direct-overlay-'+args.version+'.tar')
    with tarfile.open(bundle,'w') as tar:
        for name,path in files.items():tar.add(path,arcname=name)
    subprocess.run(stage.SSH+['mkdir','-p',out+'/overlay'],check=True)
    subprocess.run(stage.SCP+[str(bundle),stage.SSH[-1]+':'+out+'/direct-overlay.tar'],check=True)
    subprocess.run(stage.SSH+['tar','-xf',out+'/direct-overlay.tar','-C',out+'/overlay'],check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    code=CODE.replace('@ROOT@',repr(stage.ROOT)).replace('@OUT@',repr(out)).replace('@COMMIT@',commit)
    script='set -eu\nsource '+stage.ENTRY+'/metax-entry.env.sh\n"$VENV_PYTHON" - <<\'PY\'\n'+code+'\nPY\n'
    (HERE/('prepare-command-'+args.version+'.sh')).write_text(script,encoding='utf-8')
    completed=subprocess.run(stage.SSH+['bash','-s'],input=script.encode(),capture_output=True,timeout=300)
    (HERE/('prepare-'+args.version+'.stdout.txt')).write_bytes(completed.stdout)
    (HERE/('prepare-'+args.version+'.stderr.txt')).write_bytes(completed.stderr)
    print(completed.stdout.decode(errors='replace'));print(completed.stderr.decode(errors='replace'))
    completed.check_returncode()


if __name__=='__main__':main()
