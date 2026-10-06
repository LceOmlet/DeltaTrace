/opt/conda/bin/python - <<'PY'
import ast,hashlib,json,pathlib,psutil,re,subprocess,time
out=pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-label-gradient-20261006-v2'); base=pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-minibatch-20261006-v4'); mode='prepare'; name='verify_textcraft_label_gradients.py'
assert not (out/'job.json').exists(), 'Inspect the existing job; never submit twice'
out.mkdir(exist_ok=True)
source=out/name; assert hashlib.sha256(source.read_bytes()).hexdigest()=='21919f3d6efa5ca498711d47be2e3018087439f03d947fad5b70f54d4f66cd76'
ast.parse(source.read_bytes())
extra_hashes={'observe_textcraft_label_gradients.py': '9ed52c51f5eac34a2b12d5425e386f4b3e5b5945f2c6b61e01b25aba837eddb7'}
pinned={'/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-adam-20261006-v1/verify_textcraft_native_adam.py': 'cdb45238843b685234f21b882f5635b22a1bdba072948354a92ab98360268901', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-adam-20261006-v1/observe_native_adam_update.py': '74b66bf8bb703c66ae17e2cbb8506a18fe7685b3b073fd94b8ca07a6e43be36c', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-adam-20261006-v1/effective-config.yaml': '9f663a7ddad06203edaaa59ec0237c1bdf64ac1ced88e9ce00d598f9a556b4dd', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-minibatch-20261006-v4/observe_native_optimizer_minibatch.py': '022466b2bac94617b8e627ea027fb3afdf4490dc4bc2bcaa11944ab444e36214', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-minibatch-20261006-v4/observe_native_actor_loss_gradients.py': '94219328ba644a5c0118f4a3bd2561b16f969643f2cd2915047202a7ff085047', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-minibatch-20261006-v4/observe_textcraft_native_batches.py': '429ca8254e54ed3ec4882027e72d74a59aba0d9069a92086943a5817102daa4d', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-readout-20261006-v2/verify_textcraft_native_readout.py': '1fd8d8981bf82f0992db78cce43251cd2040afb4a3f039d702eb528b8d8f69cb', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-minibatch-20261006-v4/map_textcraft_native64_readout.py': 'aa5be01fd020d7f1c52d6321aaa7b8483dfe76f5c69b9203d221aaff5715fd9a', '/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-minibatch-20261006-v4/analyze_textcraft_native_minibatch.py': 'ec4512f2ce66bce1ca262075b3026080bf837909d8f62854b90c90137877c341'}
for filename, expected in extra_hashes.items():
 p=out/filename; assert hashlib.sha256(p.read_bytes()).hexdigest()==expected; ast.parse(p.read_bytes())
for filename, expected in pinned.items():
 p=pathlib.Path(filename); assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,(filename,'Pinned owner/config mismatch')
prior=json.loads((base/'job.json').read_bytes()); prepared=json.loads((base/'prepared-diagnostic.json').read_bytes())
assert (base/'native-minibatch-completed.json').is_file()
roots={}
for basename,kind in [('textcraft_owner_rollout.py','entry'),('dp_actor.py','verl'),('qwen35_dense_finite_runner.py','dt')]:
 paths=[pathlib.Path(p) for p in prepared['sources'] if pathlib.Path(p).name==basename]
 assert len(paths)==1
 p=paths[0]; roots[kind]=p.parent if kind=='entry' else p.parents[3] if kind=='verl' else p.parents[2]
checks={}
for f,h in prepared['sources'].items():
 p=pathlib.Path(f); actual=hashlib.sha256(p.read_bytes()).hexdigest(); assert actual==h,(str(p),actual)
 checks[f]=h
cases=base/'native-optimizer-minibatch.pkl'
boundary=json.loads((base/'native-minibatch-update-boundary.json').read_bytes())
assert hashlib.sha256(cases.read_bytes()).hexdigest()==boundary['snapshot']['sha256']
parent=psutil.Process(prior['reused_environment_pid'])
assert abs(parent.create_time()-prior['reused_environment_pid_birth'])<.05
env=parent.environ(); env.pop('RAY_ADDRESS',None)
tail=env['PYTHONPATH']
root=base.parents[1]
tail=tail.replace(str(root/'candidates/appworld-eval-client-routing-resume-20261005-v1/entry'),str(roots['entry']))
tail=tail.replace(str(root/'candidates/appworld-native-prefix-resume-20261005-v2/verl'),str(roots['verl']))
tail=tail.replace(str(root/'candidates/appworld-native-prefix-resume-20261005-v2/deltatrace'),str(roots['dt']))
env.update(PYTHONPATH=str(out)+':'+str(base)+':/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-adam-20261006-v1:'+str(base.parent/'textcraft-native-readout-20261006-v2')+':'+tail,VERL_ROOT=str(roots['verl']),DT_ROOT=str(roots['dt']),
 DT_TASK='TextCraft',DT_MAX_STEPS='30',DT_MAX_LENGTH='32768',
 DT_TEXTCRAFT_READOUT_ROOT=str(out),DT_TEXTCRAFT_READOUT_BASE=str(base),
 DT_TEXTCRAFT_READOUT_CASES=str(cases),DT_TEXTCRAFT_CHECKPOINT=prior['checkpoint'],CUDA_VISIBLE_DEVICES='4,5')
env['DT_TEXTCRAFT_ADAM_RECIPE_ROOT']='/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-native-adam-20261006-v1'
env['DT_TEXTCRAFT_LABEL_GRADIENT_ROOT']=str(out)
launch=json.loads((base/'launch.json').read_bytes())
env['DT_SAMPLING_JSON']=json.dumps(dict(temperature=launch['options']['actor_rollout_ref.rollout.temperature'],
 max_tokens=launch['options']['data.max_response_length']))
for key in ['DT_TEXTCRAFT_GRADIENT_DIAGNOSTIC','DT_TEXTCRAFT_MATCHED_DIAGNOSTIC','DT_TEXTCRAFT_PROBE_ROOT',
            'DT_TEXTCRAFT_MINIBATCH_ROOT','DT_TEXTCRAFT_NATIVE_OBSERVATION_DIR']:
 env.pop(key,None)
receipt=dict(role='Prepared saved native64/checkpoint25; original PG/H then labels-only original DT recomputation and swapped PG/H plus cross-PG Gram; no rollout or optimizer/scheduler update',
 sources=checks,diagnostic_source=dict(path=str(source),sha256='21919f3d6efa5ca498711d47be2e3018087439f03d947fad5b70f54d4f66cd76'),
 diagnostic_dependencies={n:dict(path=str(out/n),sha256=h) for n,h in extra_hashes.items()},
 reused_diagnostic_sources=pinned,
 input=dict(path=str(cases),sha256=hashlib.sha256(cases.read_bytes()).hexdigest(),bytes=cases.stat().st_size),
 checkpoint=prior['checkpoint'],devices=[4,5],reused_environment_pid=parent.pid,
 reused_environment_pid_birth=parent.create_time(),observed_unix=time.time(),optimizer_steps=0,
 native_batch_per_call=4,planned_backward_passes_per_rank=4,planned_original_DT_recompute=True,CPU_inspection_finite_trace_calls=0,config_source=dict(path=str(base/'launch.json'),
 sha256=hashlib.sha256((base/'launch.json').read_bytes()).hexdigest()),
 runtime_environment={k:env.get(k) for k in ['VERL_ROOT','DT_ROOT','DT_ENVIRONMENT_JSON','CUDA_VISIBLE_DEVICES',
  'TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','PYTHONPATH','DT_SAMPLING_JSON']})
if mode=='prepare':
 code="import hashlib,importlib,json,pathlib,os,psutil,torch; names=('observe_native_optimizer_minibatch','observe_native_actor_loss_gradients','observe_textcraft_native_batches','observe_textcraft_label_gradients','verify_textcraft_native_adam'); modules={n:importlib.import_module(n) for n in names}; sources={n:dict(path=str(pathlib.Path(m.__file__).resolve()),sha256=hashlib.sha256(pathlib.Path(m.__file__).read_bytes()).hexdigest()) for n,m in modules.items()}; observer=modules['observe_native_optimizer_minibatch']; assert observer.observe_native_optimizer_minibatch.__globals__ is vars(observer); value=dict(sources=sources,pid=os.getpid(),pid_birth=psutil.Process().create_time(),rss_bytes=psutil.Process().memory_info().rss,cuda_initialized=torch.cuda.is_initialized(),distributed_initialized=torch.distributed.is_initialized(),scope='Fresh Python subprocess using worker PYTHONPATH only; no driver import/sys.path edits, model or loss calls.'); pathlib.Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/textcraft-label-gradient-20261006-v2/worker-callsite-import-inspection.json').write_text(json.dumps(value,indent=2)+chr(10))"
 with (out/'worker-callsite-import.stdout.txt').open('wb') as log:
  callsite=subprocess.run([env['VENV_PYTHON'],'-c',code],cwd=out,env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
 assert callsite.returncode==0, 'Recorded helper callsite import failure before GPU submission'
 imports=json.loads((out/'worker-callsite-import-inspection.json').read_bytes())
 for item in imports['sources'].values():
  expected=pinned.get(item['path'],extra_hashes.get(pathlib.Path(item['path']).name)); assert item['sha256']==expected, 'Helper callsite actual path/hash mismatch'
 assert not imports['cuda_initialized'] and not imports['distributed_initialized']
 receipt['worker_callsite_import_inspection']=dict(path=str(out/'worker-callsite-import-inspection.json'),sha256=hashlib.sha256((out/'worker-callsite-import-inspection.json').read_bytes()).hexdigest())
 with (out/'native-owner-inspection.stdout.txt').open('wb') as log:
  result=subprocess.run([env['VENV_PYTHON'],'-u',str(source),'--inspect-only'],cwd=out,
   env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0, 'Inspect CPU binding failure before any GPU submission'
 inspected=json.loads((out/'native-owner-inspection.json').read_bytes())
 assert not inspected['cuda_initialized']
 assert inspected['native_reader_inputs_sha256']==receipt['input']['sha256']
 prior_imports=json.loads((base/'native-minibatch-source-identity.json').read_bytes())
 assert inspected['worker_constructor']=={key:prior_imports['worker_constructor'][key] for key in ['path','sha256']}, 'Native owner path/hash mismatch'
 receipt['inspection']=dict(path=str(out/'native-owner-inspection.json'),
  sha256=hashlib.sha256((out/'native-owner-inspection.json').read_bytes()).hexdigest())
 (out/'prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(dict(status='prepared_cpu_owner_inspection',out=str(out),input_sha256=receipt['input']['sha256'])))
else:
 accepted=json.loads((out/'prepared.json').read_bytes())
 for key in ['sources','diagnostic_source','diagnostic_dependencies','reused_diagnostic_sources','input','config_source','checkpoint']:
  assert receipt[key]==accepted[key], 'Prepared and launch sources differ: '+key
 physical=subprocess.run(['mx-smi'],capture_output=True,text=True,check=True).stdout
 (out/'physical-before-start.txt').write_text(physical)
 occupied=[line for line in physical.splitlines() if re.match(r'^\|\s+[45]\s+\d+\s+',line)]
 assert not occupied,occupied
 available=psutil.virtual_memory().available; assert available>150*1024**3
 argv=[env['VENV_PYTHON'],'-u',str(source)]
 with (out/'diagnostic.log').open('wb') as log:
  child=subprocess.Popen(argv,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 receipt.update(pid=child.pid,pid_birth=psutil.Process(child.pid).create_time(),started_unix=time.time(),
  argv=argv,log=str(out/'diagnostic.log'),host_available_before_bytes=available)
 (out/'job.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(dict(status='native_label_gradient_submitted',out=str(out),pid=child.pid,pid_birth=receipt['pid_birth'],devices=[4,5])))
PY
