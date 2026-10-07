"""Prepare an isolated fresh TextCraft entry using existing verified owners.

Source/configuration and CPU imports only: no service/model/DT/GPU/optimizer or
checkpoint operation, and no active manifest mutation or process submission.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
spec = importlib.util.spec_from_file_location('existing_stage', AUDIT/'stage_environment_entry.py')
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)
LOCAL = HERE/'fresh-row-prefix-v2'
ROOT = stage.ROOT
OUT = ROOT+'/candidates/textcraft-fresh-row-prefix-20261007-v2'
VENV = '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_qwen35_20260912/env/bin/python'

REMOTE = r'''
import ast, copy, difflib, hashlib, json, os, pathlib, subprocess, time
P=pathlib.Path
R=P('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922')
OUT=R/'candidates/textcraft-fresh-row-prefix-20261007-v2'
ENTRY=OUT/'entry'
OUTPUT=R/'runs/textcraft-fresh-row-prefix-20261007-v2/textcraft-dt'
DATA=R/'datasets/textcraft-native-20260930'
sha=lambda p:hashlib.sha256(P(p).read_bytes()).hexdigest()
read=lambda p:json.loads(P(p).read_bytes())
def binding(p): return dict(path=str(p),sha256=sha(p),bytes=P(p).stat().st_size)
ts=R/'runs/textcraft-official-whitening-20261006-v2/textcraft-dt/source.json'
assert sha(ts)=='175565cfe8c83ae146f7f06dfd93f62361705e95a0ac0853106deac10b182708'
prior=read(ts)
appsource=R/'runs/appworld-fresh-row-prefix-20261007-v1/appworld-dt/source.json'
assert sha(appsource)=='c83b96debba90d85476e58b56b2c8c7f3900901f26ce41094e8f60ac489c7789'
app=read(appsource)
prod=R/'candidates/appworld-row-cuts-finite-20261007-v1/production-wiring-v1'
assert sha(prod/'prepared.json')=='6c65c8c17f80575cbb643505e06cd8e6c8150d90ccdb23c0d49d4320aace2103'
pp=read(prod/'prepared.json')
assert sha(prod/'actual-imports-cpu.json')=='f1f481e8a76e2abaf84f0ca5c0ca7147a46b0d907cdba28a95882fb236c23a47'
old=R/'candidates/textcraft-official-whitening-formal-20261006-v2/entry'
old_owner=P(prior['verl_root']);dt=P(app['dt_root'])
for name,h in prior['entry_sha256'].items(): assert sha(old/name)==h, name
for name,h in prior['verl_sha256'].items(): assert sha(old_owner/name)==h, name
for name,h in app['dt_source_sha256'].items(): assert sha(dt/name)==h, name
for item in pp['import_resolution'].values(): assert sha(item['path'])==item['sha256'],item['path']
envpath=P(pp['environment']['path'])
assert sha(envpath)=='4ff007805297bfd5caddf7158f5b3198c42b41597951d7f226b0808f109b998e'
q35=read(envpath)['qwen35']
assert sha(q35['finite_library'])==q35['finite_library_sha256']=='4f42c391055afec0a0fee9ee698c0820163ff413e42f1c4909b2961ce81e5157'
assert q35['individual_prefixes'] and q35['boundary_row_storage']
assert not OUT.exists(), 'Preserve separately prepared candidates'
assert not OUTPUT.exists(), 'Fresh output must not exist'
OUT.mkdir();ENTRY.mkdir();owner=OUT/'verl';owner.mkdir()
actor_name='verl/workers/actor/dp_actor.py'
actor_reference=P(app['verl_root'])/actor_name
assert sha(actor_reference)=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
def method_AST(path,name):
 return ast.dump(next(n for n in ast.walk(ast.parse(P(path).read_bytes())) if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
assert method_AST(old_owner/actor_name,'update_policy')==method_AST(actor_reference,'update_policy')
selected_verl=dict(prior['verl_sha256']);selected_verl[actor_name]=sha(actor_reference)
selected_head=dict(prior['owner_head_sha256']);selected_head[actor_name]=sha(actor_reference)
for name in selected_head:
 p=owner/name;p.parent.mkdir(parents=True,exist_ok=True)
 target=actor_reference if name==actor_name else old_owner/name
 assert target.is_file(),str(target)
 p.symlink_to(target)
(OUT/'actor-owner.diff').write_text(''.join(difflib.unified_diff((old_owner/actor_name).read_text().splitlines(True),actor_reference.read_text().splitlines(True),fromfile=str(old_owner/actor_name),tofile=str(actor_reference))))
for item in [app['canonical_HF_owner'],*app['installed_verified_files']]: assert sha(item['path'])==item['sha256'],item['path']
for name in prior['entry_sha256']:
 p=ENTRY/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((old/name).read_bytes())
assets={'owner_environment_configs.json':'69af7f858cd25f2f76f8ccc4c6831d0e9eb51dc072817489257bcb84cd0faafd',
 'textcraft_qwen_template.json':'254d1693b6ee23781ec4aacdfb2d81f311b1a0d1f756e7808ea965dc865a0648'}
for name,digest in assets.items():
 assert sha(old/name)==digest,name
 (ENTRY/name).write_bytes((old/name).read_bytes())
for name in ['deltatrace_rollout.py','native_prefix_leases.py']:
 (ENTRY/name).write_bytes((prod/'entry'/name).read_bytes())
assert sha(ENTRY/'deltatrace_rollout.py')=='3e0c6feb2d55d54566f7e732f17a07f2e2b71333f326a34d90900b7daa13d011'
assert sha(ENTRY/'native_prefix_leases.py')=='b94756147cc6f8e59fb39baa1c2737aa655b0d9c6b87091969a65c9071ad0852'
launcher=ENTRY/'launch_textcraft_native.py'
original=(old/launcher.name).read_text()
assert original.count("'trainer.resume_mode': 'auto'")==1
launcher.write_text(original.replace("'trainer.resume_mode': 'auto'", "'trainer.resume_mode': 'disable'"))
changes={name:dict(before=h,after=sha(ENTRY/name)) for name,h in prior['entry_sha256'].items() if sha(ENTRY/name)!=h}
assert set(changes)=={'launch_textcraft_native.py','deltatrace_rollout.py'}
entry_sha={name:sha(ENTRY/name) for name in dict.fromkeys([*prior['entry_sha256'],*assets,'native_prefix_leases.py'])}
for name in changes:
 (OUT/(name+'.diff')).write_text(''.join(difflib.unified_diff((old/name).read_text().splitlines(True),(ENTRY/name).read_text().splitlines(True),fromfile=str(old/name),tofile=str(ENTRY/name))))
env=dict(prior['environment'])
env.update(VERL_ROOT=str(owner),DT_ROOT=str(dt),DT_ENTRY_ROOT=str(ENTRY),DT_ENVIRONMENT_JSON=str(envpath),
 CUDA_VISIBLE_DEVICES='2,3',MODEL_PATH=q35['checkpoint'],DT_TASK='TextCraft',DT_MAX_STEPS='30',
 DT_MAX_LENGTH='32768',DT_SAMPLING_JSON=json.dumps(prior['sampling']))
env.pop('MACA_VISIBLE_DEVICES',None)
env['PYTHONPATH']=':'.join(str(ENTRY)+p[len(str(old)):] if p==str(old) or p.startswith(str(old)+'/') else str(owner)+p[len(str(old_owner)):] if p==str(old_owner) or p.startswith(str(old_owner)+'/') else str(dt)+p[len(prior['dt_root']):] if p==prior['dt_root'] or p.startswith(prior['dt_root']+'/') else p for p in prior['pythonpath'].split(':'))
assert env['VERL_ROOT']==str(owner)
cpu=dict(os.environ,**env);cpu.update(CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='-1')
inspection_code=r"""
import ast,hashlib,importlib,inspect,json,os,pathlib,resource,sys,time
from types import SimpleNamespace
P=pathlib.Path;sha=lambda p:hashlib.sha256(P(p).read_bytes()).hexdigest();out=P(sys.argv[1])
import torch
assert not torch.cuda.is_initialized()
import launch_textcraft_native as launcher
import owner_runtime_options as runtime
import verl.trainer.ppo.ray_trainer as trainer
import verl.utils.torch_functional as functional
import verl.workers.actor.dp_actor as actor
import textcraft_environment_entry as task
import textcraft_owner_rollout as task_rollout
import reward_readout as readout
import deltatrace_rollout as producer
env=json.loads(P(os.environ['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
dt=P(os.environ['DT_ROOT']);sys.path[:0]=[str(dt),env['official_root'],str(dt/'clean/qwen35')]
modules={'launcher':launcher,'runtime_options':runtime,'trainer':trainer,'official_helper':functional,'actor':actor,
 'task_environment':task,'task_rollout':task_rollout,'readout':readout,'producer':producer}
for name in ['native_prefix_leases','qwen35_dense_finite_runner','qwen35_answer_finite','qwen35_native_prefix_artifacts','vendor_fa_finite_bf16_d256']:
 modules[name]=importlib.import_module(name)
options,sampling=launcher.options_for(P(sys.argv[2]),P(sys.argv[3]),resume_from=None)
assert options['trainer.resume_mode']=='disable' and 'trainer.resume_from_path' not in options
assert sha(readout.__file__)=='94a7afbc09da72b62572d31fd32a6534f6e8f3daf656fce1011cdfa68b3c3e2b'
leases=modules['native_prefix_leases'];factory=leases.prepare_native_prefix_leases
node=next(n for n in ast.walk(ast.parse(P(producer.__file__).read_bytes())) if isinstance(n,ast.If) and n.lineno==396)
instance=SimpleNamespace(readout_options={'prefix_lease_factory':factory})
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),producer.__file__+'::existing-factory-config-only','exec'),dict(self=instance,env=env,prepare_native_prefix_leases=factory))
configured=instance.readout_options['prefix_lease_factory']
assert configured.func is factory and configured.keywords==dict(individual_prefixes=True,boundary_row_storage=True)
inspect.signature(factory).bind_partial(None,[],minibatch_size=4,eos_token_id=248046,**configured.keywords)
inspect.signature(readout.EventRatioReadout).bind_partial(None,None,task='TextCraft',prefix_lease_factory=configured)
assert not torch.cuda.is_initialized()
mem=dict(line.split(':',1) for line in P('/proc/self/smaps_rollup').read_text().splitlines()[1:] if ':' in line)
record=dict(status='CPU_existing_owner_imports_and_factory_interface_passed',observed_unix=time.time(),options=options,
 sampling=sampling,owner_command=runtime.owner_command(options),imports={n:dict(path=m.__file__,resolved_path=str(P(m.__file__).resolve()),sha256=sha(m.__file__)) for n,m in modules.items()},
 factory=dict(owner_signature=str(inspect.signature(factory)),kwargs=configured.keywords,called=False),
 cuda_initialized=False,PSS_bytes=int(mem['Pss'].split()[0])*1024,RSS_bytes=int(mem['Rss'].split()[0])*1024,
 max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,model=False,DT=False,GPU=False,checkpoint=False,
 scope='Original owner imports, options_for and existing callback branch/signature only; no model/factory/capture/environment/rollout/trainer invocation')
(out/'native-interface-inspection.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(status=record['status'],PSS_bytes=record['PSS_bytes'],max_rss_bytes=record['max_rss_bytes'],imports=len(modules))))
"""
started=time.time()
with (OUT/'cpu-interface.log').open('wb') as log:
 completed=subprocess.run([env['VENV_PYTHON'],'-c',inspection_code,str(OUT),str(DATA),str(OUTPUT)],cwd=ENTRY,env=cpu,stdout=log,stderr=subprocess.STDOUT,timeout=120)
assert completed.returncode==0,(completed.returncode,str(OUT/'cpu-interface.log'))
inspection=read(OUT/'native-interface-inspection.json');options=inspection['options']
allowed={'trainer.default_local_dir','trainer.rollout_data_dir','trainer.validation_data_dir','trainer.resume_mode','trainer.resume_from_path'}
option_changes={k:dict(before=prior['startup_options'].get(k),after=options.get(k)) for k in prior['startup_options'].keys()|options.keys() if prior['startup_options'].get(k)!=options.get(k)}
assert set(option_changes)==allowed,option_changes
assert options['actor_rollout_ref.model.lora_rank']==8 and options['actor_rollout_ref.model.lora_alpha']==16
assert options['actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu']==4
assert options['actor_rollout_ref.actor.ppo_mini_batch_size']==64 and options['actor_rollout_ref.actor.ppo_epochs']==1
assert options['data.train_batch_size']==32 and options['env.rollout.n']==8 and options['env.max_steps']==30
assert options['trainer.total_epochs']==30
for name,item in inspection['imports'].items():
 if name in pp['import_resolution']: assert item['sha256']==pp['import_resolution'][name]['sha256'],name
assert inspection['imports']['trainer']['sha256']=='7366557b482e604d66f47bdc4841ea147ffaac7c80538fa000544bb4e92eb619'
assert inspection['imports']['actor']['sha256']=='3a65e173300be82a7a9e056a96227c4f746eabc3be778ef41c8d50138d52ce6c'
assert inspection['imports']['task_rollout']['sha256']=='d9bb65501ded52fb2fd7a5aebf06557496c22e4ea679a649bc46596b9ba420f5'
assert inspection['sampling']==prior['sampling']
assert not OUTPUT.exists()
argv=[env['VENV_PYTHON'],str(launcher),'--data',str(DATA),'--output',str(OUTPUT)]
plan=dict(argv=argv,working_directory=str(OUTPUT),entry=str(ENTRY),output=str(OUTPUT),environment=env,
 environment_source=prior['environment_source'],startup_options=options,sampling=inspection['sampling'],resume_mode='disable',checkpoint_restore_requested=False,
 budget=dict(total_epochs=30,expected_iterations=330,groups_per_iteration=32,trajectories_per_group=8,global_ppo_minibatch=64,effective_ppo_epochs=1),
 status='prepared_only_not_submitted',scope='Original launcher/main_ppo, task and official workload unchanged; isolated fresh resume configuration and verified general DT owner entry')
import psutil
service_path=R/'receipts/environment-only-20260930/entry/textcraft-service.json'
service_record=read(service_path);service_pid=service_record['pid']
service_observation=dict(receipt=binding(service_path),pid=service_pid,live=psutil.pid_exists(service_pid),scope='PID identity only; no service/API/environment call')
if service_observation['live']:
 proc=psutil.Process(service_pid);service_observation.update(pid_birth=proc.create_time(),status=proc.status())
source=copy.deepcopy(prior)
historical_keys=['prepared_receipt','prepared_receipt_sha256','launch_plan_receipt','launch_plan_sha256','effective_config_preparation','submission_script_sha256',
 'padding_comparison_receipt','padding_comparison_receipt_sha256','rollout_scope_comparison','rollout_scope_comparison_sha256','completed_overlay_receipt','completed_overlay_sha256',
 'prior_driver_observation','physical_before','host_available_bytes','service_observation']
baseline_verification={k:source.pop(k) for k in historical_keys if k in source}
for k in ['resume_from','resume_launcher','completed_checkpoint_marker','completed_checkpoint_proofs','checkpoint_files','data_loader_state_sha256','prior_driver_pid','prior_source_receipt','prior_source_sha256','actual_checkpoint_restore']:
 source.pop(k,None)
source.update(unix=time.time(),entry=str(ENTRY),verl_root=str(owner),verl_sha256=selected_verl,owner_head_sha256=selected_head,dt_root=str(dt),entry_sha256=entry_sha,dt_source_sha256=app['dt_source_sha256'],
 pythonpath=env['PYTHONPATH'],environment=env,startup_options=options,candidate_environment=binding(envpath),
 finite_library=binding(q35['finite_library']),fresh_base_model=q35['checkpoint'],resume_mode='disable',checkpoint_restore_requested=False,
 baseline_textcraft_source=binding(ts),generic_DT_reference_source=binding(appsource),entry_changes=changes,
 added_entry_sources={'native_prefix_leases.py':entry_sha['native_prefix_leases.py']},
 actual_CPU_imports=inspection['imports'],configuration_changes=option_changes,canonical_HF_owner=app['canonical_HF_owner'],installed_verified_files=app['installed_verified_files'],
 service_observation=service_observation,
 historical_baseline_verification_references=dict(scope='Historical verification provenance only; none is a requested checkpoint, current preparation or current launch',references=baseline_verification),
 actor_padding_sha256=selected_verl[actor_name],
 actor_owner_reuse=dict(reference=binding(actor_reference),baseline=binding(old_owner/actor_name),update_policy_AST_identical=True),
 prepared_only=True,submission_repository_commit='@COMMIT@',preparation_source_sha256='@SCRIPT_SHA@',
 source_scope='Fresh original TextCraft/AgentGym task and synchronous VERL owners; existing whole-batch masked_whiten; current verified generic row-prefix DT only',
 inherited_verification_scope='Generic DT actual B8 and exact32768/kernel receipts retain original scope; these CPU interface imports are not TextCraft training/numerical verification')
source['resource_environment']={k:env[k] for k in ['VERL_RELEASE_UNUSED_HOST_CACHE','VERL_TRIM_SHARED_PADDING','VERL_TRIM_RESPONSE_HEAD','CUDA_VISIBLE_DEVICES','DT_ROOT','DT_ENTRY_ROOT','DT_ENVIRONMENT_JSON']}
source['source_bindings']={str(ENTRY/name):digest for name,digest in entry_sha.items()}
source['source_bindings'].update({str(owner/name):digest for name,digest in selected_head.items()})
source['source_bindings'].update({str(dt/name):digest for name,digest in app['dt_source_sha256'].items()})
for item in [source['candidate_environment'],source['finite_library'],app['canonical_HF_owner'],*app['installed_verified_files']]:source['source_bindings'][item['path']]=item['sha256']
for name,data in [('run-env.json',env),('launch-plan.json',plan)]:
 (OUT/name).write_text(json.dumps(data,indent=2)+'\n')
source.update(launch_plan_receipt=str(OUT/'launch-plan.json'),launch_plan_sha256=sha(OUT/'launch-plan.json'),
 CPU_preparation_receipt=binding(OUT/'native-interface-inspection.json'))
(OUT/'source-template.json').write_text(json.dumps(source,indent=2)+'\n')
os.chmod(OUT/'run-env.json',0o600)
receipt=dict(status='prepared_only_CPU_interface_passed_not_submitted',observed_unix=time.time(),phase_wall_seconds=time.time()-started,
 remote_root=str(OUT),entry=str(ENTRY),output=str(OUTPUT),output_created=False,devices=[2,3],resume_mode='disable',checkpoint_restore_requested=False,
 source_template=binding(OUT/'source-template.json'),run_environment=binding(OUT/'run-env.json'),launch_plan=binding(OUT/'launch-plan.json'),
 interface=binding(OUT/'native-interface-inspection.json'),entry_changes=changes,added_entry_sources=source['added_entry_sources'],configuration_changes=option_changes,
 service_observation=service_observation,
 original_VERL_root=str(old_owner),selected_VERL_root=str(owner),original_VERL_source_count=len(prior['verl_sha256']),full_owner_file_count=len(selected_head),source_binding_count=len(source['source_bindings']),VERL_sha_changes={actor_name:dict(before=prior['verl_sha256'][actor_name],after=selected_verl[actor_name])},DT_root=str(dt),DT_source_count=len(app['dt_source_sha256']),
 model=False,DT=False,GPU=False,checkpoint=False,service=False,submit=False,manifest_changed=False,
 scope='Prepared source copy, exact recorded source hashes, unchanged original options and CPU imports/callback binding; no new experiment settings or numerical claims')
(OUT/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(prepared=binding(OUT/'prepared.json'),entry=str(ENTRY),argv=argv,phase_wall_seconds=receipt['phase_wall_seconds'],submit=False)))
'''

if __name__ == '__main__':
    LOCAL.mkdir(exist_ok=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=stage.REPO,text=True).strip()
    script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    script=REMOTE.replace('@COMMIT@',commit).replace('@SCRIPT_SHA@',script_sha)
    command='set -eu\nsource '+ROOT+'/receipts/environment-only-20260930/entry/metax-entry.env.sh\n'+VENV+" - <<'PY'\n"+script+'\nPY\n'
    (LOCAL/'prepare-command.sh').write_text(command,encoding='utf8')
    completed=subprocess.run(stage.SSH+['bash','-s'],input=command.encode(),capture_output=True,timeout=180)
    (LOCAL/'prepare.stdout.txt').write_bytes(completed.stdout)
    (LOCAL/'prepare.stderr.txt').write_bytes(completed.stderr)
    record=dict(returncode=completed.returncode,script_sha256=script_sha,remote_script_sha256=hashlib.sha256(command.encode()).hexdigest(),repository_commit=commit,remote_root=OUT,submit=False)
    (LOCAL/'execution.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
    print(completed.stdout.decode(errors='replace'));print(completed.stderr.decode(errors='replace'));print(json.dumps(record))
    if completed.returncode: raise SystemExit(completed.returncode)
