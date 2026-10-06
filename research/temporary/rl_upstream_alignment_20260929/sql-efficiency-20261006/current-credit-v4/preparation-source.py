"""Prepare frozen SQL with accepted credit seams; CPU only, no job submission."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess

from stage_environment_entry import AUDIT, ENTRY, REPO, ROOT, SCP, SSH


NAME = 'sql-native-current-credit-20261006-v4'
LOCAL = AUDIT / 'sql-efficiency-20261006' / 'current-credit-v4'
OUT = ROOT + '/receipts/sql-efficiency-20261006/current-credit-v4'
BASE = ROOT + '/candidates/' + NAME
PRIOR = ROOT + '/candidates/sql-native-host-cache-20261005-v2/prepared.json'
PRIOR_SHA = '630ed1f67bb7b6dc6a9d0b16a5c91c69773c0c47c5da11e9dfb59795cdf87c92'
ENTRY_SEAMS = {
    'owner_trajectory_batch.py': ('e42f5968f8cf62d447f6f30262c9672b9c9ddc52',
        'trajectory_credit', None,
        '2d3a93f03ba13b49056209c5be5488da6bea0fb9ba1f58d8549b358194b6aca3',
        '95042a7a610e4c7781982873b5730d4bb744145dbcabe61961d55621e83d8939'),
    'dt_training_batch.py': ('22bcfdbf271cd3278bcf3d2486c3e1dfaab4a329',
        'compute_training_credit', None,
        'da9b8a01c3bb9ba00fe3388fd95f93961d33c44bdf0eccf32e7b5846e2ffcae2',
        '3f4a5f70abc93778db2d2285e55f2bc75e0f62bac03a880f4c66350465c95d56'),
    'reward_readout.py': ('afe59dd', 'query_ids', 'RewardAlphabet',
        '228afbc7a10841d482c3d73def59dfe9ef192c057a97d76dca57369502a10137',
        '6f2a4e326d1200994112a798fd6799c6842d7f7807262197ee864da956c5f5df'),
}
QUERY_CLAUSES = (
    ('possibly ending inside an unfinished response. Do not treat this forecast request ',
     'after generation of the current response has ended, possibly at its token limit. Do not treat this forecast request '),
    ('as an environment action. Imagine completing that response and continuing with ',
     'as an environment action. Imagine the environment immediately processing that response as emitted and then continuing with '),
)
PINNED_FILES = {
    'patch_verl_agent2.py': (REPO / 'experiments/rl/patch_verl_agent2.py',
        '84155a172b4c41a4deea7f1a2dff6a6b2044f99035331841393c8ca210cf61b9'),
    'test_dt_official_whitening.py': (REPO / 'tests/test_dt_official_whitening.py',
        'f53a09962229704ba227d7d122ab57387c4896d129e3f6cbb0ae3e92d75dbfea'),
    'owner-pristine-trainer.py': (AUDIT / 'recipe-sources/verl-agent-20bd331/verl/trainer/ppo/ray_trainer.py',
        '69dab6ef8a0521704468c8158c6b19c782d5455da49a3a1582d4bcd6a2a51e30'),
    'test_distributed_credit.py': (AUDIT / 'appworld-efficiency-20261006/retained-action-v2/test_distributed_credit.py',
        'f1e90e6f24ed9e7906cb5b188491b3c5ba6df4d675769934e6258cc1ef4c7358'),
    'accepted-retained-prepared.json': (AUDIT / 'appworld-efficiency-20261006/retained-action-v2/prepared.json',
        '4ae2f06dd440a15625c636b916bcda90eb4bb8f263fdda5c99436ea21f4406b2'),
}


def main():
    source = Path(__file__).read_bytes()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    blobs = {}
    for name, (path, expected) in PINNED_FILES.items():
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, name
        blobs[name] = base64.b64encode(raw).decode('ascii')
    for name, (commit, _, _, _, expected) in ENTRY_SEAMS.items():
        raw = subprocess.check_output(['git', 'show', commit + ':experiments/rl/' + name], cwd=REPO)
        if name == 'reward_readout.py':
            assert hashlib.sha256(raw).hexdigest() == '94a7afbc09da72b62572d31fd32a6534f6e8f3daf656fce1011cdfa68b3c3e2b'
            blobs['accepted-query-source.py'] = base64.b64encode(raw).decode('ascii')
            raw = subprocess.check_output(['git', 'show', '99fb5c28:experiments/rl/reward_readout.py'], cwd=REPO)
            assert hashlib.sha256(raw).hexdigest() == ENTRY_SEAMS[name][3]
            for before, after in QUERY_CLAUSES:
                assert raw.count(before.encode()) == 1
                raw = raw.replace(before.encode(), after.encode(), 1)
        assert hashlib.sha256(raw).hexdigest() == expected, name
        blobs[name] = base64.b64encode(raw).decode('ascii')
    blobs['preparation-source.py'] = base64.b64encode(source).decode('ascii')
    code = r'''
import ast,base64,copy,hashlib,importlib.util,json,os,psutil,resource,shutil,subprocess,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
root=Path(@ROOT@);out=Path(@OUT@);base=Path(@BASE@);prior_path=Path(@PRIOR@)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
started=time.time();proc=psutil.Process()
assert sha(prior_path)==@PRIOR_SHA@
prior=read(prior_path);assert prior['status']=='prepared_not_launched'
old_entry=Path(prior['entry']);old_owner=Path(prior['verl_root'])
for directory,files in [(old_entry,prior['entry_sha256']),(old_owner,prior['owner_sha256']),
 (Path(prior['dt_root']),prior['dt_source_sha256'])]:
 for name,h in files.items():assert sha(directory/name)==h,name
assert prior['entry_sha256']['owner_trajectory_batch.py']==@ENTRY_SEAMS@['owner_trajectory_batch.py'][3]
assert prior['entry_sha256']['dt_training_batch.py']==@ENTRY_SEAMS@['dt_training_batch.py'][3]
assert prior['entry_sha256']['reward_readout.py']==@ENTRY_SEAMS@['reward_readout.py'][3]
trainer_rel='verl/trainer/ppo/ray_trainer.py'
assert prior['owner_sha256'][trainer_rel]=='8816ea4e5f9a95a1dfe1067349eee9101da8bf4ec916bcc28a5b03288976f0df'
assert prior['owner_sha256']['verl/utils/torch_functional.py']=='079a6d20696a861340687d6e61fb4162cc1d436837ce747ac9ef318b738c1701'
manifest_paths=[root/'active-training.json',root/'active-source.json']
manifests={str(p):sha(p) for p in manifest_paths}
assert not base.exists() and not out.exists(), 'Inspect existing immutable preparation'
out.mkdir(parents=True);sources=out/'source';sources.mkdir()
for name,encoded in @BLOBS@.items():
 with (sources/name).open('xb') as f:f.write(base64.b64decode(encoded,validate=True))
accepted=read(sources/'accepted-retained-prepared.json')
assert accepted['work_counters_source_commit']=='22bcfdbf271cd3278bcf3d2486c3e1dfaab4a329'
assert accepted['changes']['owner_trajectory_batch.py']['before_sha256']==prior['entry_sha256']['owner_trajectory_batch.py']
assert accepted['changes']['dt_training_batch.py']['before_sha256']==prior['entry_sha256']['dt_training_batch.py']
(out/'prior-prepared.json').write_bytes(prior_path.read_bytes())

def tree(directory):
 return {str(p.relative_to(directory)):({'symlink':os.readlink(p)} if p.is_symlink() else sha(p))
  for p in directory.rglob('*') if (p.is_file() or p.is_symlink()) and
  '.git' not in p.relative_to(directory).parts and '__pycache__' not in p.parts and p.suffix!='.pyc'}

def without_function(text,function,owner=None):
 parsed=ast.parse(text)
 body=parsed.body if owner is None else next(n for n in parsed.body if isinstance(n,ast.ClassDef) and n.name==owner).body
 selected=[n for n in body if isinstance(n,ast.FunctionDef) and n.name==function]
 assert len(selected)==1
 body.remove(selected[0])
 return ast.dump(parsed)

old_entry_tree=tree(old_entry);old_owner_tree=tree(old_owner)
base.mkdir();entry=base/'entry';owner=base/'verl'
shutil.copytree(old_entry,entry,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
shutil.copytree(old_owner,owner,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
assert tree(entry)==old_entry_tree and tree(owner)==old_owner_tree
changes={}
for name,(commit,function,cls,before,after) in @ENTRY_SEAMS@.items():
 assert sha(old_entry/name)==before and sha(sources/name)==after
 assert without_function((old_entry/name).read_text(),function,cls)==without_function((sources/name).read_text(),function,cls),name
 shutil.copy2(sources/name,entry/name)
 changes[name]={'before_sha256':before,'after_sha256':after,'function':function,'class':cls,'source_commit':commit,'outside_function_AST_equal':True}
 if name=='reward_readout.py':
  def query_method(path):
   c=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='RewardAlphabet')
   return next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='query_ids')
  assert ast.dump(query_method(entry/name))==ast.dump(query_method(sources/'accepted-query-source.py'))
  changes[name].update(accepted_method_source_sha256=sha(sources/'accepted-query-source.py'),
   accepted_query_method_AST_equal=True,scope='Only the two accepted clock clauses; SQL EventRatioReadout/prefix interface preserved')
spec=importlib.util.spec_from_file_location('accepted_patch',sources/'patch_verl_agent2.py')
patch=importlib.util.module_from_spec(spec);spec.loader.exec_module(patch)
trainer=owner/trainer_rel
candidate_text=patch.patch_dt_advantage_preprocessing(trainer.read_text())
trainer.write_text(candidate_text)
assert sha(trainer)=='7366557b482e604d66f47bdc4841ea147ffaac7c80538fa000544bb4e92eb619'
assert patch.patch_dt_advantage_preprocessing(candidate_text)==candidate_text
before=ast.parse((old_owner/trainer_rel).read_text());after=ast.parse(candidate_text)
for parsed in (before,after):
 parsed.body=[n for n in parsed.body if not (isinstance(n,ast.FunctionDef) and n.name=='compute_advantage')
  and not (isinstance(n,ast.Import) and len(n.names)==1 and n.names[0].name=='verl.utils.torch_functional' and n.names[0].asname=='verl_F')]
assert ast.dump(before)==ast.dump(after), 'Change outside DT advantage seam/import'
changes[trainer_rel]={'before_sha256':sha(old_owner/trainer_rel),'after_sha256':sha(trainer),
 'function':'compute_advantage','patch_function':'patch_dt_advantage_preprocessing',
 'patch_sha256':sha(sources/'patch_verl_agent2.py'),'outside_function_and_official_import_AST_equal':True}
new_entry_tree=tree(entry);new_owner_tree=tree(owner)
assert set(new_entry_tree)==set(old_entry_tree) and set(new_owner_tree)==set(old_owner_tree)
assert {n for n in new_entry_tree if new_entry_tree[n]!=old_entry_tree[n]}==set(@ENTRY_SEAMS@)
assert {n for n in new_owner_tree if new_owner_tree[n]!=old_owner_tree[n]}=={trainer_rel}
assert tree(old_entry)==old_entry_tree and tree(old_owner)==old_owner_tree

env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='-1',MACA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
 PYTHONDONTWRITEBYTECODE='1',VERL_ROOT=str(owner),DT_ENTRY_ROOT=str(entry),DT_ROOT=prior['dt_root'],
 DT_VERL_PRISTINE_TRAINER_SOURCE=str(sources/'owner-pristine-trainer.py'),
 DT_VERL_TRAINER_SOURCE=str(old_owner/trainer_rel),
 DT_VERL_TORCH_FUNCTIONAL_SOURCE=str(owner/'verl/utils/torch_functional.py'))
env['PYTHONPATH']=':'.join([str(entry),str(owner),env.get('PYTHONPATH','')])
nodes=['test_native_two_rank_dispatch_keeps_complete_future_returns','test_zero_batch_does_not_enter_fsdp_collectives',
 'test_training_slices_do_not_drop_later_executed_reward','test_empty_training_slice_selection_avoids_all_dt_calls',
 'test_trajectory_uses_native_slice_sources_after_complete_returns','test_owner_balance_restores_credit_after_padding_and_duplicate_rows',
 'test_native_denominators_stay_collectively_aligned_without_changing_credit']
tests=[];failed=False
for name,argv in [('transport',[env['VENV_PYTHON'],'-m','pytest','-q','--import-mode=importlib','-p','no:cacheprovider',
 *[str(sources/'test_distributed_credit.py')+'::'+n for n in nodes],'--junitxml='+str(out/'transport-tests.xml')]),
 ('whitening',[env['VENV_PYTHON'],'-m','unittest','discover','-s',str(sources),'-p','test_dt_official_whitening.py','-v'])]:
 tick=time.time();result=subprocess.run(argv,cwd=entry,env=env,capture_output=True)
 log=out/(name+'-tests.stdout.txt');log.write_bytes(result.stdout+result.stderr)
 report={'name':name,'argv':argv,'returncode':result.returncode,'wall_seconds':time.time()-tick,
  'stdout':str(log),'stdout_sha256':sha(log)}
 if name=='transport':
  xml=out/'transport-tests.xml';report['xml']=str(xml);report['xml_sha256']=sha(xml)
  suite=ET.parse(xml).getroot().find('testsuite');report['counts']={k:int(suite.get(k,'0')) for k in ('tests','errors','failures','skipped')}
 else:
  report['scope']='Actual pinned owner helper definitions and generated original compute_advantage; no model or full PPO numerical test'
 tests.append(report);failed|=result.returncode!=0

os.environ.update(env)
sys.path[:0]=[str(entry),str(owner)]
import torch
import verl.utils.torch_functional as functional
assert str(Path(functional.__file__).resolve())==str((owner/'verl/utils/torch_functional.py').resolve())
assert sha(Path(functional.__file__))==prior['owner_sha256']['verl/utils/torch_functional.py']
resource_record={'pid':proc.pid,'created_unix':proc.create_time(),'wall_seconds':time.time()-started,
 'parent_RSS_bytes':proc.memory_info().rss,'parent_max_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
 'child_max_RSS_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
 'torch_version':torch.__version__,'torch_path':torch.__file__,'CUDA_initialized':torch.cuda.is_initialized(),
 'CUDA_VISIBLE_DEVICES':env['CUDA_VISIBLE_DEVICES'],'MACA_VISIBLE_DEVICES':env['MACA_VISIBLE_DEVICES'],
 'OMP_NUM_THREADS':env['OMP_NUM_THREADS'],'MKL_NUM_THREADS':env['MKL_NUM_THREADS'],
 'official_helper_path':functional.__file__,'official_helper_sha256':sha(Path(functional.__file__))}
assert not resource_record['CUDA_initialized']
assert {str(p):sha(p) for p in manifest_paths}==manifests, 'Active manifests changed'
assert tree(old_entry)==old_entry_tree and tree(old_owner)==old_owner_tree
prepared=copy.deepcopy(prior)
prepared.update(prepared_unix=time.time(),status='prepared_only_CPU_interface_tests_passed' if not failed else 'prepared_only_CPU_interface_tests_failed',
 entry=str(entry),verl_root=str(owner),
 entry_sha256={n:sha(entry/n) for n in prior['entry_sha256']},
 owner_sha256={n:sha(owner/n) for n in prior['owner_sha256']},
 prior_prepared_receipt=str(prior_path),prior_prepared_sha256=sha(prior_path),
 supersedes_for_future_start=str(prior_path),preparation_repository_commit=@REVISION@,
 preparation_script_sha256=sha(sources/'preparation-source.py'),
 credit_source_changes=changes,
 credit_preparation_inputs={'accepted_retained_receipt':{'path':str(sources/'accepted-retained-prepared.json'),'sha256':sha(sources/'accepted-retained-prepared.json')},
  'source_files_sha256':{n:sha(sources/n) for n in @BLOBS@},
  'original_credit_CPU_tests':prior['cpu_tests'],'original_host_cache_CPU_tests':prior['host_boundary_cpu']},
 credit_CPU_tests=tests,credit_CPU_resources=resource_record,
 preserved={'entry_file_count':len(prior['entry_sha256']),'entry_files_unchanged':len(prior['entry_sha256'])-3,
  'VERL_changed_files':[trainer_rel],'DT_c9_source_sha256_unchanged':True,
  'actor_worker_core_transport_scope_host_cache_unchanged':True,
  'SQL_prefix_scope':'Existing pair-internal prefix only; no cross-response lease/bank interface introduced',
  'budget_sampling_LoRA8_16_B4_PPO_configuration_unchanged':True,
  'active_manifest_sha256':manifests,'old_candidates_unchanged':True},
 scope='Only accepted trainer whitening and three existing entry function seams. Raw d/Q/V/A/returns preserved; complete future returns before retained-source selection; original SQL workload, scope, host cache and kernels preserved. CPU contracts are not FA/FLA/DT numerical, performance or training acceptance.',
 deployment=False,stop_calls=0,launch_calls=0,GPU_calls=0,model_initializations=0,training_calls=0)
with (out/'prepared.json').open('x') as f:f.write(json.dumps(prepared,indent=2)+'\n')
summary={'prepared':str(out/'prepared.json'),'sha256':sha(out/'prepared.json'),'status':prepared['status'],
 'candidate':str(base),'changes':changes,'tests':tests,'resources':resource_record,'preserved':prepared['preserved']}
with (out/'composition-result.json').open('x') as f:f.write(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
assert not failed, 'Preserved CPU failure receipts; no deployment'
'''
    for key, value in {'ROOT': ROOT, 'OUT': OUT, 'BASE': BASE, 'PRIOR': PRIOR,
                       'PRIOR_SHA': PRIOR_SHA, 'ENTRY_SEAMS': ENTRY_SEAMS,
                       'BLOBS': blobs, 'REVISION': revision}.items():
        code = code.replace('@' + key + '@', repr(value))
    ast.parse(code)
    LOCAL.mkdir(parents=True, exist_ok=False)
    shell = f"set -e\nsource {ENTRY}/metax-entry.env.sh\nexport CUDA_VISIBLE_DEVICES=-1 MACA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\n\"$VENV_PYTHON\" - <<'PY'\n{code}\nPY\n"
    (LOCAL / 'prepare.sh').write_text(shell, encoding='utf-8')
    result = subprocess.run(SSH + ['bash', '-s'], input=shell.encode(), capture_output=True)
    (LOCAL / 'prepare.stdout.txt').write_bytes(result.stdout + result.stderr)
    for name in ('prepared.json', 'composition-result.json', 'transport-tests.xml',
                 'transport-tests.stdout.txt', 'whitening-tests.stdout.txt'):
        fetched = subprocess.run(SCP + [f'{SSH[-1]}:{OUT}/{name}', str(LOCAL / name)], capture_output=True)
        if fetched.returncode:
            (LOCAL / (name + '.fetch.stderr.txt')).write_bytes(fetched.stderr)
    result.check_returncode()
    print((LOCAL / 'composition-result.json').read_text())


if __name__ == '__main__':
    main()
