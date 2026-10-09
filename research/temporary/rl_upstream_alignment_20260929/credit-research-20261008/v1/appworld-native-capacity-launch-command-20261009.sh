source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
CUDA_VISIBLE_DEVICES=-1 "$VENV_PYTHON" - <<'PY'
import os,json,subprocess,time,hashlib
from pathlib import Path
r=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');out=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/appworld-memory-stable-composition-20261009-v1');source_path=r/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json';source=json.loads(source_path.read_bytes());memory=r/'candidates/direct-target-consumed-cache-release-20261007-v1';qwen=json.loads((memory/'environment.json').read_bytes())['qwen35']
env=dict(os.environ,**source['environment']);env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env.update(DT_TASK='AppWorld',DT_MAX_STEPS=str(source['startup_options']['env.max_steps']),DT_ROOT=str(memory/'deltatrace'),DT_ENVIRONMENT_JSON=str(memory/'environment.json'),CUDA_VISIBLE_DEVICES='-1')
old_helpers=r/'receipts/direct-target-memory-capacity-20261008-v1'
env['PYTHONPATH']=':'.join([str(out),str(old_helpers),str(memory/'deltatrace'),env.get('DT_OFFICIAL_ROOT') or qwen['official_root'],str(memory/'deltatrace/clean/qwen35'),source['pythonpath'],qwen['ft_extension_root']])
expected={'check_appworld_native_dt_capacity.py':'a5ba8729b4003e5f5916353456e0de8e653dfde45fe81d17d021d70e6923788f'}
for name,want in expected.items(): assert hashlib.sha256((out/name).read_bytes()).hexdigest()==want
for name,want in {'check_capacity_lifetime.py':'060070d993cef62fae79150b7e2e6ffc4d0521b7fd13e383181d285eac291d3a','profile_existing_offload.py':'732f434a1e3c6deea46e6c5c9fab68b481d15be59e477d4f03147e13ec2892c7','inspect_extreme_endpoint.py':'8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5'}.items(): assert hashlib.sha256((old_helpers/name).read_bytes()).hexdigest()==want
assert hashlib.sha256(source_path.read_bytes()).hexdigest()=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
assert hashlib.sha256((memory/'environment.json').read_bytes()).hexdigest()=='cd28a6e2140190ffcba4d64229b5c62549b85473645fea249899014a034d6db2'
assert hashlib.sha256((memory/'deltatrace/clean/qwen35/qwen35_dense_finite_runner.py').read_bytes()).hexdigest()=='7d6f57f61ecde7ce3506b8ef58d61268859fc04357c35de99a1892928c4ba7a8'
assert hashlib.sha256((memory/'deltatrace/clean/qwen35/qwen35_gdn_finite.py').read_bytes()).hexdigest()=='3f51f5f7569a4d6b410e7acef89635a50f2e6ecadfa43eb94d1fd4fdd655dfd1'
import psutil
formal=psutil.Process(982372);assert abs(formal.create_time()-1791553809.84)<.05
assert not (out/'native-capacity-v1').exists() and not (out/'launch-native-capacity-v1.json').exists()
env['CUDA_VISIBLE_DEVICES']='4,5'
argv=[env['VENV_PYTHON'],str(out/'check_appworld_native_dt_capacity.py'),'--source',str(source_path),'--failed-batch',str(r/'receipts/direct-target-native-mlp-memory-20261007-v3/failed-inputs'),'--output',str(out/'native-capacity-v1')]
log=out/'native-capacity-v1.log'
with log.open('wb') as stream:
 process=subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(unix=time.time(),pid=process.pid,birth=psutil.Process(process.pid).create_time(),argv=argv,log=str(log),devices=[4,5],source_commit='2c9daaed',script_sha256=expected['check_appworld_native_dt_capacity.py'],source_path=str(source_path),source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),DT_ROOT=env['DT_ROOT'],DT_ENVIRONMENT_JSON=env['DT_ENVIRONMENT_JSON'],PYTHONPATH=env['PYTHONPATH'],numerical_version='fla-early-output-scale-20261009-v1',numerical_source_commit='26bef6c8',status='launched_not_verified',operations=dict(DT_RPC=2,PPO_updates=0,rollout_requests=0,checkpoint_restore=False),formal_textcraft_unchanged=True,appworld_formal_restarted=False)
(out/'launch-native-capacity-v1.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
PY
