set -eu
source /mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/environment-only-20260930/entry/metax-entry.env.sh
"$VENV_PYTHON" - <<'PY'

import hashlib,json,os,re,subprocess,time
from pathlib import Path
import psutil
root=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922');out=Path('/mnt/si0021787ci2/default/lzq/deepresearch/deltatrace_rl_20260922/receipts/direct-target-existing-pv-rule-20261008-v1-gdn');hashes={'inspect_layer_effect.py': '315d5300c5ca445b20ab6a8afd41e8b0a3de4d341d96bb03990f5119fdca6c44', 'inspect_extreme_endpoint.py': '8a72887a32bd11533486ddee6dc0d36b4980bfb77ed94eadc7a9a13f031b36d5'}
for name,expected in hashes.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==expected
source_path=root/'runs/direct-target-prefix-runtime-20261007-v1/appworld/appworld-dt/source.json'
source_sha=hashlib.sha256(source_path.read_bytes()).hexdigest()
assert source_sha=='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0'
source=json.loads(source_path.read_bytes())
candidate=json.loads((root/'candidates/direct-target-consumed-cache-release-20261007-v1/preparation.json').read_bytes())
for key,expected in [('changed_file','changed_sha256'),('candidate_environment','candidate_environment_sha256')]:
 assert hashlib.sha256(Path(candidate[key]).read_bytes()).hexdigest()==candidate[expected]
native=root/'receipts/direct-target-prefix-runtime-20261007-v1/appworld-first-dt/rank1-readout-native-batch-16.pt'
assert hashlib.sha256(native.read_bytes()).hexdigest()=='3e902bc058ca1c06bec4c742be53523fd3e336b806d74ce19120682af2281a0a'
env=dict(os.environ,**source['environment']);env.pop('RAY_ADDRESS',None);env.pop('MACA_VISIBLE_DEVICES',None)
env.update(DT_TASK=source['startup_options']['env.env_name'],DT_MAX_STEPS=str(source['startup_options']['env.max_steps']),DT_ROOT=candidate['candidate_dt_root'],DT_ENVIRONMENT_JSON=candidate['candidate_environment'])
dt=Path(env['DT_ROOT']);qwen=json.loads(Path(env['DT_ENVIRONMENT_JSON']).read_bytes())['qwen35']
env['PYTHONPATH']=':'.join([str(out),str(dt),env.get('DT_OFFICIAL_ROOT') or qwen['official_root'],str(dt/'clean/qwen35'),source['pythonpath'],qwen['ft_extension_root']])
if not False:
 env['CUDA_VISIBLE_DEVICES']=''
 if False:
  script='''import json,torch,hashlib
from pathlib import Path
root=Path(ROOT);out=Path(OUT);native=Path(NATIVE)
original=torch.load(native,map_location='cpu',weights_only=False)
assert len(original['rows'])==4
record=dict(scope='Original real B4, only the saved attribution vector is replaced by the two already-measured existing-rule vectors for author-metric evaluation',ranks={})
for rank,mode in [(0,'original_content1'),(1,'existing_content0')]:
 trace=root/'receipts/direct-target-existing-pv-rule-20261008-v1/results'/('rank0-'+mode+'.pt')
 value=torch.load(trace,map_location='cpu',weights_only=False)
 assert value['native_sha256']==hashlib.sha256(native.read_bytes()).hexdigest()
 assert value['signed'].shape==original['native_signed'].shape
 other=torch.load(trace.with_name('rank1-'+mode+'.pt'),map_location='cpu',weights_only=False)
 assert torch.equal(value['signed'],other['signed'])
 hybrid=dict(original,native_signed=value['signed'],detail=value['detail'])
 hybrid['provenance']=dict(original_native_path=str(native),original_native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),attribution_path=str(trace),attribution_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),scope='Quality-evaluation carrier only, not a formal native capture')
 path=out/(mode+'-quality-input.pt');torch.save(hybrid,path)
 row=sorted(original['rows'],key=lambda x:x['batch_row'])[3]
 assert row['traj_uid']=='f0f85f5c-74b4-4670-a074-99a3a09dbb98' and row['trajectory_index']==64 and int(row['selected'][2883])==198
 case=dict(source_sha256='58209daa0fccfea4b70645465e96ea5d203f9d309187b7a64b405cbd9fd47da0',native_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),uid=row['traj_uid'],row=3,trajectory_index=64,response_slot=125,packed_slot=2883,token_id=198,rank=1,batch=16,native=str(path))
 record['ranks'][str(rank)]=dict(profile=mode,case=case,provenance=hybrid['provenance'],metric_source='Unchanged ft_ifr_improve.faithfulness_test_skip_tokens, k=20, signed RISE and positive-only MAS evaluation; original inspect_action_curve owner')
record['cuda_initialized']=torch.cuda.is_initialized();(out/'curve-inputs.json').write_text(json.dumps(record,indent=2)+'\\n');print(json.dumps(record))
'''.replace('ROOT',repr(str(root))).replace('OUT',repr(str(out))).replace('NATIVE',repr(str(native)))
 else:
  script='''import json,torch
from pathlib import Path
from reward_readout import DirectActionTargetReadout
d=torch.load(NATIVE,map_location='cpu',weights_only=False);result=[]
for item in sorted(d['rows'],key=lambda x:x['batch_row']):
 p=DirectActionTargetReadout._prepare_row(item['row'],0)
 assert torch.equal(p['selected'],item['selected']) and torch.equal(p['case']['target_ids'],item['case']['target_ids']) and p['target_offsets']==item['target_offsets']
 result.append(dict(uid=item['traj_uid'],length=p['selected'].numel(),targets=len(p['target_offsets'])))
print(json.dumps(dict(rows=result,cuda_initialized=torch.cuda.is_initialized(),scope='CPU input identity only, no model/DT/update')))
'''.replace('NATIVE',repr(str(native)))
 if True:
  case=dict(source_sha256=source_sha,native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),uid='f0f85f5c-74b4-4670-a074-99a3a09dbb98',row=3,trajectory_index=64,response_slot=125,packed_slot=2883,token_id=198,rank=1,batch=16,native=str(native))
  script+='\nfrom inspect_extreme_endpoint import load_request\nload_request('+repr(str(source_path))+','+repr(str(native))+','+repr(case)+')\nassert not torch.cuda.is_initialized()\n'
 r=subprocess.run([env['VENV_PYTHON'],'-c',script],env=env,cwd=out,capture_output=True)
 (out/'prepare.stdout.txt').write_bytes(r.stdout);(out/'prepare.stderr.txt').write_bytes(r.stderr);r.check_returncode()
 (out/'prepared.json').write_bytes(r.stdout);print(r.stdout.decode())
 if True:
  binding=dict(case=case,runner_sha256=candidate['changed_sha256'],scope='Current actual extreme token on original unchanged B4; passive original single/joint boundary diagnosis only')
  if True:binding.update(probe_decoder_layers=[31,30],stop_after_decoder=30)
  if True:binding.update(probe_gdn_layers=[30])
  (out/'case.json').write_text(json.dumps(binding,indent=2)+'\n')
else:
 assert (out/'prepared.json').exists() and not (out/'launch.json').exists()
 assert psutil.Process(2833207).create_time()==1791370325.16
 held=root/'receipts/direct-target-prefix-runtime-20261007-v1/textcraft-first-dt'
 assert not any((held/('rank'+str(i)+'-release-update')).exists() for i in (0,1))
 physical=subprocess.run(['mx-smi'],capture_output=True,check=True).stdout;(out/'before-physical.txt').write_bytes(physical)
 assert not any(re.match(r'^\|\s*[45]\s+\d+\s+\S',line) for line in physical.decode(errors='replace').splitlines())
 env['CUDA_VISIBLE_DEVICES']='4,5'
 program='inspect_layer_effect.py' if True else ('compare_existing_pv_curves.py' if False else 'compare_existing_pv_rule.py')
 argv=[env['VENV_PYTHON'],str(out/program),'--source',str(source_path)]
 if True:argv+=['--case',str(out/'case.json')]
 elif not False:argv+=['--native',str(native)]
 argv+=['--output',str(out/'results')]
 with (out/'driver.log').open('xb') as log:p=subprocess.Popen(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 record=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),launched_unix=time.time(),code_commit='a2242a88d03d91209628768a78b4b2746873d459',argv=argv,scripts=hashes,source_sha256=source_sha,native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),memory_candidate=candidate,devices=[4,5],DT_calls_per_rank=0 if False else 2,native_forwards_per_rank=42 if False else 0,scope=('Original passive single/joint layer comparison, current largest negative token' if True else 'Existing content1/content0 diagnostic only')+'; no profile deployment, no parameter/optimizer/scheduler updates, no rollout or checkpoint restore',formal_restart=False,credit_repaired=False,text_update_released=False)
 if True:record['case_binding']=dict(path=str(out/'case.json'),sha256=hashlib.sha256((out/'case.json').read_bytes()).hexdigest(),value=json.loads((out/'case.json').read_bytes()))
 if True:record.update(partial_DT_only=True,stop_after_decoder=30,full_signed_vector_produced=False)
 (out/'launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))

PY
