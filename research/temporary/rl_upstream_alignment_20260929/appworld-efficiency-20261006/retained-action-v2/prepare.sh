source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'

import ast,hashlib,json,os,psutil,shutil,subprocess,time
from pathlib import Path
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');receipt=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/appworld-efficiency-20261006/retained-action-v2');base=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/candidates/appworld-retained-credit-sources-20261006-v2')
old=root/'candidates/appworld-official-whitening-20261006-v1/entry'
run=root/'runs/appworld-official-whitening-20261006-v2/appworld-dt'
driver=psutil.Process(3592468);assert driver.create_time()==1791291997.14
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=json.loads((run/'source.json').read_bytes())
assert sha(run/'source.json')=='2827c5ae8785c498785b639573769d7db0208d3e3540087a4a0816b6968a34ac'
assert not base.exists(), 'Inspect existing candidate; do not replace a frozen entry'
assert not (receipt/'prepared.json').exists()
for name,digest in source['entry_sha256'].items():assert sha(old/name)==digest,name
entry=base/'entry'
shutil.copytree(old,entry,ignore=shutil.ignore_patterns('__pycache__'))
changes={}
for name,(function,before,after) in {'owner_trajectory_batch.py': ('trajectory_credit', '2d3a93f03ba13b49056209c5be5488da6bea0fb9ba1f58d8549b358194b6aca3', '95042a7a610e4c7781982873b5730d4bb744145dbcabe61961d55621e83d8939'), 'dt_training_batch.py': ('compute_training_credit', 'da9b8a01c3bb9ba00fe3388fd95f93961d33c44bdf0eccf32e7b5846e2ffcae2', '3f4a5f70abc93778db2d2285e55f2bc75e0f62bac03a880f4c66350465c95d56')}.items():
 assert sha(old/name)==before and sha(receipt/name)==after
 original=ast.parse((old/name).read_text());candidate=ast.parse((receipt/name).read_text())
 original.body=[node for node in original.body if not isinstance(node,ast.FunctionDef) or node.name!=function]
 candidate.body=[node for node in candidate.body if not isinstance(node,ast.FunctionDef) or node.name!=function]
 assert ast.dump(original)==ast.dump(candidate), 'Change outside accepted seam: '+name
 shutil.copy2(receipt/name,entry/name)
 changes[name]={'before_sha256':before,'after_sha256':after,'function':function,'source_commit':'e42f5968f8cf62d447f6f30262c9672b9c9ddc52'}
unchanged={}
for name,digest in source['entry_sha256'].items():
 if name not in changes:
  assert sha(entry/name)==digest,name
  unchanged[name]=digest
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='-1',MACA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
 VERL_ROOT=source['verl_root'],DT_ENTRY_ROOT=str(entry))
env['PYTHONPATH']=':'.join([str(entry),source['verl_root'],env.get('PYTHONPATH','')])
nodes=['test_native_two_rank_dispatch_keeps_complete_future_returns',
 'test_zero_batch_does_not_enter_fsdp_collectives',
 'test_training_slices_do_not_drop_later_executed_reward',
 'test_empty_training_slice_selection_avoids_all_dt_calls',
 'test_trajectory_uses_native_slice_sources_after_complete_returns',
 'test_owner_balance_restores_credit_after_padding_and_duplicate_rows',
 'test_native_denominators_stay_collectively_aligned_without_changing_credit']
xml=receipt/'cpu-tests.xml';started=time.time()
r=subprocess.run([env['VENV_PYTHON'],'-m','pytest','-q','--import-mode=importlib',
 *[str(receipt/'test_distributed_credit.py')+'::'+n for n in nodes],
 '--junitxml='+str(xml)],cwd=entry,env=env,capture_output=True)
(receipt/'cpu-tests.stdout.txt').write_bytes(r.stdout+r.stderr)
record={'status':'prepared_only_CPU_interface_tests_passed' if r.returncode==0 else 'prepared_only_CPU_tests_failed',
 'observed_unix':time.time(),'wall_seconds':time.time()-started,'preparation_repository_commit':'22bcfdbf271cd3278bcf3d2486c3e1dfaab4a329',
 'source_fix_commit':'e42f5968f8cf62d447f6f30262c9672b9c9ddc52','work_counters':True,
 'work_counters_source_commit':'22bcfdbf271cd3278bcf3d2486c3e1dfaab4a329' if True else None,
 'prior_entry':str(old),'entry':str(entry),'verl_root':source['verl_root'],
 'dt_root':source['dt_root'],'driver_pid':3592468,'driver_birth':driver.create_time(),
 'source_receipt':str(run/'source.json'),'source_receipt_sha256':sha(run/'source.json'),
 'changes':changes,'unchanged_entry_sha256':unchanged,
 'cpu_tests':{'returncode':r.returncode,'xml':str(xml),'sha256':sha(xml) if xml.exists() else None,
 'scope':'Actual VERL dispatch/pad/collect and Q/V transport with fixed ratio test doubles; no DT numerical or performance claim'},
 'scope':'Only select existing retained source identities after complete returns. No reward, native trajectory, sampling, PPO, whitening, LoRA, batch or tolerance change.',
 'deployment':False,'GPU_calls':0,'training_calls':0}
(receipt/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
r.check_returncode()

PY
